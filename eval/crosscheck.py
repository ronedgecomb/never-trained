#!/usr/bin/env python3
"""Cross-checks the NumPy reference forward pass against PufferLib's C code.

Runs the C harness with a per-step trace (observation + decoder outputs as
computed by PufferLib's forward_puffernet), replays the same observations
through policy/puffernet.py and compares.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "policy"))
from puffernet import OBS_SIZE, PufferNet  # noqa: E402

RLFUN_HOME = Path(os.environ.get("RLFUN_HOME", Path.home() / ".cache" / "rl-fun"))

TRACE_DTYPE = np.dtype([
    ("episode", "<i4"), ("step", "<i4"), ("tick", "<i4"),
    ("action_policy", "<i4"), ("action_applied", "<i4"),
    ("logits", "<f4", 4), ("obs", "<f4", OBS_SIZE),
    ("paddle_x", "<f4"), ("ball_x", "<f4"), ("ball_y", "<f4"),
    ("ball_vx", "<f4"), ("ball_vy", "<f4"), ("paddle_width", "<f4"),
    ("score", "<i4"), ("num_balls", "<i4"),
])


def check(name: str, weights: Path) -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        trace_path = Path(tmp) / "trace.bin"
        subprocess.run([
            str(RLFUN_HOME / "bin/bk_eval"), "--puffer-root", str(RLFUN_HOME / "PufferLib"),
            "--policy", "net", "--weights", str(weights), "--episodes", "2",
            "--seed-base", "1", "--trace", str(trace_path), "--trace-episodes", "2",
            "--out", str(Path(tmp) / "episodes.csv"),
        ], check=True, capture_output=True)
        trace = np.fromfile(trace_path, dtype=TRACE_DTYPE)

    # The two implementations round differently (C: sequential float32 sums with
    # fused multiply-add; NumPy: BLAS), and the authored logits are differences
    # of large terms, so agreement is judged on the chosen action and on the
    # logit difference relative to the logit scale.
    net = PufferNet.load(str(weights))
    worst_rel = 0.0
    worst_abs = 0.0
    argmax_mismatch = 0
    sampled_differs = 0
    for i, rec in enumerate(trace):
        new_episode = i == 0 or rec["episode"] != trace[i - 1]["episode"]
        ref = net.forward(rec["obs"], terminal=new_episode)
        c = rec["logits"]
        diff = float(np.abs(ref[:3] - c[:3]).max())
        worst_abs = max(worst_abs, diff)
        worst_rel = max(worst_rel, diff / max(1.0, float(np.abs(c[:3]).max())))
        if int(np.argmax(ref[:3])) != int(np.argmax(c[:3])):
            argmax_mismatch += 1
        if int(np.argmax(c[:3])) != int(rec["action_policy"]):
            sampled_differs += 1
    n = len(trace)
    print(f"{name}: {n} agent steps")
    print(f"  largest logit difference NumPy vs C: {worst_abs:.3g} absolute, "
        f"{worst_rel:.2e} relative to the logit scale")
    print(f"  steps where the two argmax actions differ: {argmax_mismatch}")
    print(f"  steps where PufferLib's sampled action != argmax: {sampled_differs} "
        f"({100.0 * sampled_differs / n:.2f}%)")
    return worst_rel < 5e-3 and argmax_mismatch <= n // 1000


def main() -> int:
    ok = check("authored_breakout", REPO / "policy/weights/authored_breakout.bin")
    ok &= check("pufferai_pretrained", RLFUN_HOME / "reference_weights/breakout_weights.bin")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
