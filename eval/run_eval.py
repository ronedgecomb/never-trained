#!/usr/bin/env python3
"""Runs the evaluation matrix (policies x protocols) and writes logs + summaries.

    python eval/run_eval.py --phase dev               # development seeds
    python eval/run_eval.py --phase final             # held-out seeds; requires eval/FROZEN.json
    python eval/run_eval.py --phase final --verify    # re-run and compare with the stored logs

Every episode is one row in results/<phase>/episodes.csv. Summaries are derived
from those rows only. The final phase refuses to run unless the authored weights
match the hash recorded in FROZEN.json, and refuses to overwrite existing
final results.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
RLFUN_HOME = Path(os.environ.get("RLFUN_HOME", Path.home() / ".cache" / "rl-fun"))
PUFFER_ROOT = RLFUN_HOME / "PufferLib"
BK_EVAL = RLFUN_HOME / "bin" / "bk_eval"
FROZEN = REPO / "eval" / "FROZEN.json"

MAX_STEPS = 30_000          # agent steps per episode (90,000 frames = 25 min of game time)
MAX_SCORE = 864

POLICIES = {
    "authored": {
        "args": ["--policy", "net", "--weights", str(REPO / "policy/weights/authored_breakout.bin")],
        "how": "Weights written directly by Claude (policy/gen_weights.py). No training.",
    },
    "naive_tracker": {
        "args": ["--policy", "net", "--weights", str(REPO / "policy/weights/naive_tracker.bin")],
        "how": "Authored baseline: linear 'follow the ball' rule in the same architecture.",
    },
    "pufferai_pretrained": {
        "args": ["--policy", "net", "--weights", str(RLFUN_HOME / "reference_weights/breakout_weights.bin")],
        "how": "Checkpoint published by PufferAI (puffer.ai repo, docs/assets/models/"
               "breakout_weights.bin), trained with PufferLib's RL trainer.",
    },
    "random": {
        "args": ["--policy", "random"],
        "how": "Uniformly random action each agent step.",
    },
}

PROTOCOLS = {
    "standard": {
        "args": [],
        "what": "PufferLib Breakout exactly as configured in config/breakout.ini.",
    },
    "sticky25": {
        "args": ["--sticky", "0.25"],
        "what": "Stress test: each agent step, with probability 0.25 the previously "
                "applied action is repeated instead of the policy's choice.",
    },
    "greedy": {
        "args": ["--argmax"],
        "what": "Standard environment, argmax action instead of PufferLib's softmax sampling.",
        "skip": ["random"],
    },
}

PHASES = {
    "dev": {"seed_base": 1, "episodes": 20},
    # Wider development check of the authored policy alone (looking for rare misses).
    "dev300": {"seed_base": 1, "episodes": 300, "policies": ["authored"], "protocols": ["standard"]},
    "final": {"seed_base": 1001, "episodes": 100},
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit(path: Path) -> str:
    out = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True, text=True)
    return out.stdout.strip()


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator,
        resamples: int = 10_000) -> tuple[float, float]:
    idx = rng.integers(0, len(values), size=(resamples, len(values)))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def summarize(rows: list[dict]) -> dict:
    score = np.array([int(r["score"]) for r in rows], dtype=float)
    frames = np.array([int(r["frames"]) for r in rows], dtype=float)
    lost = np.array([int(r["balls_lost"]) for r in rows], dtype=float)
    cleared = int(sum(int(r["cleared"]) for r in rows))
    n = len(rows)
    rng = np.random.default_rng(12345)   # analysis only: bootstrap resampling
    lo, hi = bootstrap_mean_ci(score, rng)
    wl, wh = wilson(cleared, n)
    cleared_frames = frames[[int(r["cleared"]) == 1 for r in rows]]
    return {
        "episodes": n,
        "score_mean": float(score.mean()),
        "score_std": float(score.std(ddof=1)) if n > 1 else 0.0,
        "score_min": float(score.min()),
        "score_median": float(np.median(score)),
        "score_max": float(score.max()),
        "score_mean_ci95": [lo, hi],
        "perf_mean": float(score.mean() / MAX_SCORE),
        "cleared": cleared,
        "clear_rate": cleared / n,
        "clear_rate_ci95": [wl, wh],
        "balls_lost_mean": float(lost.mean()),
        "balls_lost_max": int(lost.max()),
        "zero_loss_episodes": int((lost == 0).sum()),
        "frames_mean": float(frames.mean()),
        "frames_to_clear_mean": float(cleared_frames.mean()) if len(cleared_frames) else None,
        "truncated": int(sum(int(r["truncated"]) for r in rows)),
        "distinct_trajectories": len({(r["action_hash"], r["frames"]) for r in rows}),
    }


def run(label: str, policy_args: list[str], protocol_args: list[str],
        seed_base: int, episodes: int, log) -> list[dict]:
    cmd = [str(BK_EVAL), "--puffer-root", str(PUFFER_ROOT), *policy_args,
        "--episodes", str(episodes), "--seed-base", str(seed_base),
        "--max-steps", str(MAX_STEPS), "--label", label, *protocol_args]
    log.write("$ " + " ".join(cmd) + "\n")
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    log.write(proc.stderr)
    log.flush()
    return list(csv.DictReader(proc.stdout.splitlines()))


def matrix(phase: dict):
    for protocol, pspec in PROTOCOLS.items():
        if protocol not in phase.get("protocols", PROTOCOLS):
            continue
        for policy, spec in POLICIES.items():
            if policy in pspec.get("skip", []) or policy not in phase.get("policies", POLICIES):
                continue
            yield protocol, pspec, policy, spec


def verify(name: str, phase: dict, out_dir: Path) -> int:
    """Re-runs the phase with the current harness build and compares every
    episode row with the stored log. Writes reproduction_check.json; never
    touches episodes.csv or summary.json."""
    stored = list(csv.DictReader(open(out_dir / "episodes.csv")))
    fresh: list[dict] = []
    with open(os.devnull, "w") as log:
        for protocol, pspec, policy, spec in matrix(phase):
            rows = run(f"{policy}/{protocol}", spec["args"], pspec["args"],
                phase["seed_base"], phase["episodes"], log)
            for r in rows:
                r["policy"], r["protocol"] = policy, protocol
            fresh.extend(rows)
    keys = [k for k in stored[0] if k not in ("policy", "protocol")]
    mismatches = [i for i, (a, b) in enumerate(zip(stored, fresh))
        if any(a[k] != b[k] for k in keys) or a["policy"] != b["policy"]
        or a["protocol"] != b["protocol"]]
    identical = len(stored) == len(fresh) and not mismatches
    record = {
        "phase": name,
        "checked_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "episodes_compared": len(stored),
        "identical": identical,
        "mismatching_rows": mismatches[:20],
        "harness_sha256": {
            "bk_eval.c": sha256(REPO / "harness/bk_eval.c"),
            "bk_common.h": sha256(REPO / "harness/bk_common.h"),
        },
        "note": "Re-run of the stored evaluation with the harness sources as they are now.",
    }
    (out_dir / "reproduction_check.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"{name}: {len(stored)} stored rows, {len(fresh)} re-run rows, "
        f"{'IDENTICAL' if identical else 'DIFFERENT'}")
    return 0 if identical else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--verify", action="store_true",
        help="re-run the phase and compare every episode row with the stored episodes.csv")
    args = parser.parse_args()
    phase = PHASES[args.phase]
    out_dir = REPO / "eval" / "results" / args.phase
    authored = REPO / "policy/weights/authored_breakout.bin"

    if args.verify:
        return verify(args.phase, phase, out_dir)

    if args.phase == "final":
        if not FROZEN.exists():
            print("eval/FROZEN.json missing: freeze the policy first (eval/freeze.py)")
            return 1
        frozen = json.loads(FROZEN.read_text())
        if sha256(authored) != frozen["authored_weights_sha256"]:
            print("authored weights differ from the frozen hash; refusing to run")
            return 1
        if (out_dir / "episodes.csv").exists():
            print(f"{out_dir}/episodes.csv exists; final results are never overwritten")
            return 1
    out_dir.mkdir(parents=True, exist_ok=True)

    started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    all_rows: list[dict] = []
    summary: dict = {}
    with open(out_dir / "harness.log", "w") as log:
        for protocol, pspec, policy, spec in matrix(phase):
            label = f"{policy}/{protocol}"
            rows = run(label, spec["args"], pspec["args"],
                phase["seed_base"], phase["episodes"], log)
            for r in rows:
                r["policy"], r["protocol"] = policy, protocol
            all_rows.extend(rows)
            summary[label] = summarize(rows)
            s = summary[label]
            print(f"{label:34s} score {s['score_mean']:7.1f} ± {s['score_std']:6.1f}  "
                f"cleared {s['cleared']:3d}/{s['episodes']}  "
                f"balls lost {s['balls_lost_mean']:.2f}  frames {s['frames_mean']:8.0f}")

    fields = ["policy", "protocol"] + [k for k in all_rows[0] if k not in ("policy", "protocol", "label")]
    with open(out_dir / "episodes.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    meta = {
        "phase": args.phase,
        "started_utc": started,
        "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "seeds": [phase["seed_base"], phase["seed_base"] + phase["episodes"] - 1],
        "episodes_per_cell": phase["episodes"],
        "max_agent_steps_per_episode": MAX_STEPS,
        "environment": {
            "library": "PufferLib 5.0 (Ocean Breakout)",
            "pufferlib_commit": git_commit(PUFFER_ROOT),
            "config": "config/default.ini + config/breakout.ini, no overrides",
        },
        "weights_sha256": {
            "authored": sha256(authored),
            "naive_tracker": sha256(REPO / "policy/weights/naive_tracker.bin"),
            "pufferai_pretrained": sha256(RLFUN_HOME / "reference_weights/breakout_weights.bin"),
        },
        "harness_sha256": {
            "bk_eval.c": sha256(REPO / "harness/bk_eval.c"),
            "bk_common.h": sha256(REPO / "harness/bk_common.h"),
        },
        "host": {"platform": platform.platform(), "python": sys.version.split()[0],
            "numpy": np.__version__},
        "policies": {k: v["how"] for k, v in POLICIES.items()},
        "protocols": {k: v["what"] for k, v in PROTOCOLS.items()},
        "summary": summary,
    }
    (out_dir / "summary.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {out_dir}/episodes.csv, summary.json, harness.log")
    return 0


if __name__ == "__main__":
    sys.exit(main())
