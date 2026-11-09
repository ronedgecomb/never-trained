#!/usr/bin/env python3
"""Checks that the generated weights compute the intended formulas.

No environment rollouts are involved: random *states* are drawn (fixed seed),
turned into observations, pushed through the NumPy PufferNet, and compared with
controller_spec.decide(). States within a small margin of a decision boundary
are excluded, since there the two may legitimately differ by rounding.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import controller_spec as spec
import gen_weights as g
from puffernet import OBS_SIZE, PufferNet

WEIGHTS = Path(__file__).parent / "weights" / "authored_breakout.bin"


def make_obs(p, pw, x, y, vx, vy) -> np.ndarray:
    obs = np.zeros(OBS_SIZE, np.float32)
    obs[0] = p / 576.0
    obs[1] = 312.0 / 330.0
    obs[2] = x / 576.0
    obs[3] = y / 330.0
    obs[4] = vx / 512.0
    obs[5] = vy / 512.0
    obs[6] = 0.2
    obs[8] = 1.0
    obs[9] = pw / 62.0
    return obs


def main() -> int:
    rng = np.random.default_rng(0)
    net = PufferNet.load(str(WEIGHTS))
    n = 200_000
    checked = disagreements = 0
    worst_px = 0.0
    for _ in range(n):
        pw = 62.0 if rng.random() < 0.5 else 31.0
        p = rng.uniform(0.0, 576.0 - pw)
        x = rng.uniform(0.0, 544.0)
        y = rng.uniform(0.0, 300.0)
        speed = rng.choice([256.0, 320.0, 384.0, 448.0]) / 60.0
        angle = np.radians(rng.uniform(-60.0, 60.0))
        vx = speed * np.sin(angle)
        vy = speed * np.cos(angle) * (1.0 if rng.random() < 0.5 else -1.0)

        action, mode, error = spec.decide(p, pw, x, y, vx, vy)
        trace: dict = {}
        logits = net.forward(make_obs(p, pw, x, y, vx, vy), terminal=True, trace=trace)[:3]

        # Recover the network's own error estimate (px) from its RIGHT score.
        score_r = trace["layer4"][g.CH["TR"]] - g.SCORE_OFFSET - 0.5
        if mode == "intercept":
            net_error = score_r / vy + g.DEADBAND
        else:
            net_error = score_r / g.TRACK_GAIN + g.DEADBAND
        # Within half a pixel of the brick line the layer-4 switch is mid-way
        # between its two inputs, so the comparison is not meaningful there.
        if abs(y - g.BRICK_BOTTOM) < 0.5:
            continue
        worst_px = max(worst_px, abs(net_error - error))
        if abs(abs(error) - g.DEADBAND) < 0.5:
            continue
        checked += 1
        if int(np.argmax(logits)) != action:
            disagreements += 1
        margin = np.sort(logits)[-1] - np.sort(logits)[-2]
        assert margin > 10.0, (margin, mode, error)

    print(f"states checked: {checked} of {n} (rest within 0.5 px of a boundary)")
    print(f"action disagreements with the specification: {disagreements}")
    print(f"largest |network error - specified error|: {worst_px:.4f} px")

    # The pre-serve observation (ball at rest above the paddle) must give NOOP.
    logits = net.forward(make_obs(257.0, 62.0, 272.0, 135.0, 0.0, 0.0), terminal=True)[:3]
    print(f"pre-serve logits (NOOP, LEFT, RIGHT): {logits}")
    ok = disagreements == 0 and worst_px < 1.0 and int(np.argmax(logits)) == 0
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
