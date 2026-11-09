#!/usr/bin/env python3
"""Looks at every ball the authored policy loses under the sticky-action stress
test on the development seeds, and at what the sticky actions did just before.

Uses per-step traces from the C harness (policy's choice vs. applied action).
Writes eval/results/dev/sticky_failures.txt.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "eval"))
from crosscheck import TRACE_DTYPE  # noqa: E402

RLFUN_HOME = Path(os.environ.get("RLFUN_HOME", Path.home() / ".cache" / "rl-fun"))
EPISODES = 20
LOOKBACK = 8        # agent steps before the ball passes the paddle


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        trace_path = Path(tmp) / "trace.bin"
        subprocess.run([
            str(RLFUN_HOME / "bin/bk_eval"), "--puffer-root", str(RLFUN_HOME / "PufferLib"),
            "--policy", "net", "--weights", str(REPO / "policy/weights/authored_breakout.bin"),
            "--episodes", str(EPISODES), "--seed-base", "1", "--sticky", "0.25",
            "--trace", str(trace_path), "--trace-episodes", str(EPISODES),
            "--out", str(Path(tmp) / "episodes.csv"),
        ], check=True, capture_output=True)
        trace = np.fromfile(trace_path, dtype=TRACE_DTYPE)

    lines = []
    losses = 0
    overridden_counts = []
    extreme_angle = 0
    for ep in range(EPISODES):
        e = trace[trace["episode"] == ep]
        balls = e["num_balls"]
        for i in np.flatnonzero(balls[1:] < balls[:-1]) + 1:
            # Find the step where the ball dropped below the paddle line (y > 280).
            j = i - 1
            while j > 0 and e["ball_y"][j] > 280.0:
                j -= 1
            window = e[max(0, j - LOOKBACK + 1): j + 1]
            overridden = int((window["action_policy"] != window["action_applied"]).sum())
            speed = float(np.hypot(e["ball_vx"][j], e["ball_vy"][j]))
            angle = float(np.degrees(np.arctan2(abs(e["ball_vx"][j]), abs(e["ball_vy"][j])))) if speed else 0.0
            pw = float(e["paddle_width"][j])
            miss = float((e["ball_x"][j] + 16.0) - (e["paddle_x"][j] + pw / 2.0))
            losses += 1
            overridden_counts.append(overridden)
            extreme_angle += angle > 70.0
            lines.append(f"seed {ep + 1:2d} step {int(e['step'][i]):5d}: paddle {pw:.0f}px, "
                f"ball angle {angle:4.0f} deg from vertical, ball-paddle offset {miss:+7.1f}px, "
                f"{overridden} of the last {LOOKBACK} actions overridden")

    counts = np.array(overridden_counts)
    summary = [
        f"Authored policy, sticky25 protocol, development seeds 1-{EPISODES}",
        f"balls lost: {losses}",
        f"  preceded by >= 2 overridden actions in the last {LOOKBACK} steps: {int((counts >= 2).sum())}",
        f"  preceded by exactly 1 overridden action: {int((counts == 1).sum())}",
        f"  preceded by none: {int((counts == 0).sum())}",
        f"  ball travelling more than 70 degrees from vertical when lost: {int(extreme_angle)}",
        "",
    ]
    out = REPO / "eval/results/dev/sticky_failures.txt"
    out.write_text("\n".join(summary + lines) + "\n")
    print("\n".join(summary))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
