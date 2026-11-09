#!/usr/bin/env python3
"""Lists every non-zero weight of the authored network by name.

Writes policy/weights/authored_breakout_nonzero.txt. Each line is one row of a
weight matrix, written as the linear combination it computes. Reading the file
top to bottom is reading the whole policy.

Usage: python policy/describe_weights.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import gen_weights as G

OBS_NAMES = {0: "paddle_x", 1: "paddle_y", 2: "ball_x", 3: "ball_y", 4: "ball_vx",
    5: "ball_vy", 6: "balls_fired", 7: "score", 8: "num_balls", 9: "paddle_w"}
ROWS = ["candidate", "gate", "highway"]


def term(coef: float, name: str) -> str:
    return f"{coef:+.6g}*{name}"


def main() -> None:
    b = G.build_authored()
    names = G.CHANNELS + [f"unused{i}" for i in range(len(G.CHANNELS), G.HIDDEN)]
    out = ["Authored Breakout policy: every non-zero weight", ""]
    total = 0

    out.append("ENCODER  channel = sum(weight * observation)")
    for j in range(G.HIDDEN):
        nz = np.flatnonzero(b.encoder[j])
        if len(nz):
            total += len(nz)
            out.append(f"  {names[j]:4s} = " + "  ".join(
                term(b.encoder[j, i], OBS_NAMES.get(i, f"brick{i - 10}")) for i in nz))

    for layer, W in enumerate(b.layers, 1):
        out.append("")
        out.append(f"LAYER {layer}  (candidate | gate | highway rows; inputs are channels)")
        closed = 0
        for block, label in enumerate(ROWS):
            for j in range(G.HIDDEN):
                row = W[block * G.HIDDEN + j]
                nz = np.flatnonzero(row)
                if not len(nz):
                    continue
                total += len(nz)
                if block == 2 and len(nz) == 1 and nz[0] == G.CH["ONE"] and row[nz[0]] == -G.BIG:
                    closed += 1
                    continue
                out.append(f"  {label:9s}[{names[j]:3s}] = " + "  ".join(
                    term(row[i], names[i]) for i in nz))
        out.append(f"  highway[*] = {-G.BIG:+g}*ONE for the other {closed} units "
            "(gate closed: the unit copies its input)")

    out.append("")
    out.append("DECODER  logit = sum(weight * channel)")
    for r, label in enumerate(["NOOP", "LEFT", "RIGHT", "value"]):
        nz = np.flatnonzero(b.decoder[r])
        total += len(nz)
        out.append(f"  {label:5s} = " + ("  ".join(term(b.decoder[r, i], names[i]) for i in nz)
            if len(nz) else "0"))

    out.append("")
    out.append(f"non-zero weights: {total} of {b.flat().size}")
    path = Path(__file__).parent / "weights" / "authored_breakout_nonzero.txt"
    path.write_text("\n".join(out) + "\n")
    print("\n".join(out[-1:]))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
