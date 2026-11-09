#!/usr/bin/env python3
"""Deterministic weight generation for the authored Breakout policy.

Every number written to the checkpoint comes from a named constant or a
closed-form expression below. Nothing here is fitted, searched, sampled or
learned: there is no data, no loss and no random number generator.

The target is PufferLib 5.0's default policy network ("PufferNet") in the exact
shape its Breakout config uses, so the file is a drop-in replacement for a
trained checkpoint:

    obs[118] -> encoder (32x118, no bias)
             -> 4 x MinGRU-with-highway layer (96x32 each, no bias)
             -> decoder (4x32, no bias): logits NOOP, LEFT, RIGHT + value

One layer computes, per unit j, from its input vector x:

    cand, gate, hw = W @ x                      (three bias-free projections)
    g(c)   = c + 0.5 if c >= 0 else sigmoid(c)
    m      = state + sigmoid(gate) * (g(cand) - state);   state <- m
    out_j  = sigmoid(hw) * m + (1 - sigmoid(hw)) * x_j    (highway)

How the design uses those pieces (details at each layer below):

  * no bias terms exist, so a constant-1 channel is derived from obs[1]
    (paddle_y / height, which never changes);
  * gate = +BIG removes the memory (m = g(cand)), giving a feed-forward unit;
  * g() is linear for c >= 0 and ~0 for c << 0, i.e. a ReLU up to a 0.5 offset;
  * hw = -BIG closes the highway gate: the unit copies its input (a wire);
  * hw = +BIG * s is a switch on the sign of s (a multiplexer);
  * hw = small * v puts the gate in its linear range, sigmoid(z) ~ 1/2 + z/4,
    so the unit outputs (1/2 + z/4) * m: a multiplication of two inputs.

Usage:  python policy/gen_weights.py            (writes policy/weights/*.bin)
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from puffernet import OBS_SIZE, pack, weight_count

REVISION = 1

HIDDEN = 32
LAYERS = 4

# ---------------------------------------------------------------------------
# Environment facts, read from PufferLib ocean/breakout/breakout.h and
# config/breakout.ini (units: pixels, frames at 60 Hz).
# ---------------------------------------------------------------------------
FIELD_W = 576.0
FIELD_H = 330.0
BALL_SIZE = 32.0
PADDLE_Y = FIELD_H - 8.0 - 10.0          # 312: top edge of the paddle
PADDLE_W_FULL = 62.0                     # shrinks to 31 after the ball touches the ceiling
BRICK_BOTTOM = 50.0 + 6 * 12.0           # 122: Y_OFFSET + rows * brick_height
FRAMESKIP = 3
PADDLE_STEP = 620.0 / 60.0 * FRAMESKIP   # 31 px: paddle travel per agent step
OBS_V_SCALE = 512.0                      # velocities are observed as px/frame / 512

# Observation index and the factor that converts it back to physical units.
PHYS = {
    "one": (1, FIELD_H / PADDLE_Y),   # obs[1] = paddle_y/height = 312/330, a constant
    "p": (0, FIELD_W),                # paddle left edge
    "x": (2, FIELD_W),                # ball left edge
    "y": (3, FIELD_H),                # ball top edge
    "vx": (4, OBS_V_SCALE),           # ball velocity, px/frame
    "vy": (5, OBS_V_SCALE),
    "pw": (9, PADDLE_W_FULL),         # current paddle width
}

# ---------------------------------------------------------------------------
# Design constants (chosen by reasoning; see the comment on each).
# ---------------------------------------------------------------------------
# The ball can travel no further right than this (left-edge coordinate).
X_MAX = FIELD_W - BALL_SIZE                       # 544

# Ball-top height at which its bottom edge meets the paddle is 312 - 32 = 280.
# The env computes the bounce angle from the ball position at the START of the
# contact frame, on average half a frame (vy/2 ~ 3 px) before contact, so the
# landing point is predicted for a line 3 px higher.
LAND_Y = PADDLE_Y - BALL_SIZE - 3.0               # 277

# Stop moving once the paddle is within half a step of the target: any larger
# dead band wastes accuracy, any smaller one makes the paddle oscillate.
DEADBAND = PADDLE_STEP / 2.0                      # 15.5 px

# While tracking, aim at where the ball will be one agent step from now.
LEAD_FRAMES = float(FRAMESKIP)                    # 3 frames

# Offsets that keep the multiplicands positive, where g() is exactly linear.
X_OFF = 50.0                                      # ball x and paddle target are >= -0.5
H_OFF = 60.0                                      # remaining height 277 - y is > -50

# Highway-gate multiplier slope. |v| <= 7.5 px/frame so |z| = K_MUL*|v| <= 0.03,
# where sigmoid(z) = 1/2 + z/4 holds to a relative error of z^2/12 < 1e-4.
K_MUL = 0.004

# Gain inside the ReLU units; makes g()'s 0.5 offset negligible (0.5/8 px^2/frame).
RELU_GAIN = 8.0

# Added to both action scores so the multiplexer inputs stay in g()'s linear
# range (scores are bounded by 7.5 px/frame * 576 px ~ 4300 in magnitude).
SCORE_OFFSET = 8192.0

# Converts the tracking error (px) to the same units as the interception
# score (px * px/frame); 4 px/frame is a typical vertical ball speed.
TRACK_GAIN = 4.0

# Logit gain. PufferLib samples actions from softmax(logits); with this gain a
# 1 px difference in the decision variable is > 24 logits, so sampling and
# argmax coincide except within a hair of a tie.
LOGIT_GAIN = 12.0

# Saturates a sigmoid: sigmoid(100) == 1 and sigmoid(-100) == 0 in float32.
BIG = 100.0

# ---------------------------------------------------------------------------
# Hidden channels. The same index means the same quantity in every layer.
# ---------------------------------------------------------------------------
CHANNELS = [
    "ONE",   # constant 1
    "VX",    # ball vx (px/frame)
    "VY",    # ball vy (px/frame, positive = falling)
    "BX",    # ball x + X_OFF
    "BH",    # height left to fall + H_OFF = (LAND_Y + H_OFF) - y
    "BQ",    # q + X_OFF, where q = the ball x that would be centred on the paddle
    "ZY",    # BRICK_BOTTOM - y: positive while the ball is still among/above bricks
    "TR",    # score for RIGHT (starts as the tracking score, see layers 3 and 4)
    "TL",    # score for LEFT
    "TR2",   # tracking score for RIGHT minus 0.5 (candidate input for layer 4)
    "TL2",   # tracking score for LEFT minus 0.5
    "PC",    # product: sigmoid(K*vy) * (x + X_OFF)
    "PQ",    # product: sigmoid(K*vy) * (q + X_OFF)
    "PB",    # product: sigmoid(K*vx) * (height left + H_OFF)
    "R1",    # relu(-U):            landing point is left of the left wall
    "R2",    # relu(-U - X_MAX*vy): ... by more than one field width
    "R3",    # relu(U - X_MAX*vy):  landing point is right of the right wall
    "R4",    # relu(U - 2*X_MAX*vy): ... by more than one field width
]
CH = {name: i for i, name in enumerate(CHANNELS)}
assert len(CHANNELS) <= HIDDEN


class Builder:
    """Bias-free tensors addressed by channel name."""

    def __init__(self) -> None:
        self.encoder = np.zeros((HIDDEN, OBS_SIZE), np.float64)
        self.decoder = np.zeros((4, HIDDEN), np.float64)
        self.layers = [np.zeros((3 * HIDDEN, HIDDEN), np.float64) for _ in range(LAYERS)]
        # Default for every unit in every layer: highway gate closed, i.e. copy
        # the input channel unchanged. Channels nobody writes stay exactly 0.
        for W in self.layers:
            for j in range(HIDDEN):
                W[2 * HIDDEN + j, CH["ONE"]] = -BIG

    def encode(self, channel: str, **phys: float) -> None:
        """channel = sum(coef * physical variable), expressed in obs weights."""
        for name, coef in phys.items():
            index, scale = PHYS[name]
            self.encoder[CH[channel], index] += coef * scale

    def _row(self, layer: int, block: int, channel: str, terms: dict[str, float]) -> None:
        row = self.layers[layer - 1][block * HIDDEN + CH[channel]]
        row[:] = 0.0
        for name, coef in terms.items():
            row[CH[name]] += coef

    def candidate(self, layer: int, channel: str, **terms: float) -> None:
        self._row(layer, 0, channel, terms)

    def gate(self, layer: int, channel: str, **terms: float) -> None:
        self._row(layer, 1, channel, terms)

    def highway(self, layer: int, channel: str, **terms: float) -> None:
        self._row(layer, 2, channel, terms)

    def feed_forward(self, layer: int, channel: str, **terms: float) -> None:
        """out = g(sum(terms)): no memory, highway fully open."""
        self.candidate(layer, channel, **terms)
        self.gate(layer, channel, ONE=BIG)
        self.highway(layer, channel, ONE=BIG)

    def decode(self, row: int, **terms: float) -> None:
        for name, coef in terms.items():
            self.decoder[row, CH[name]] += coef

    def flat(self) -> np.ndarray:
        return pack(self.encoder, self.decoder, self.layers)


def scaled(terms: dict[str, float], factor: float) -> dict[str, float]:
    return {name: coef * factor for name, coef in terms.items()}


def add(*parts: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for part in parts:
        for name, coef in part.items():
            out[name] = out.get(name, 0.0) + coef
    return out


def encode_tracking_scores(b: Builder, lead_frames: float) -> None:
    """Tracking: keep the paddle under the ball.

    e  = ball centre - paddle centre = (x + 16) - (p + pw/2)
    E  = e + lead_frames * vx              (where the ball will be shortly)
    RIGHT is good when E > DEADBAND, LEFT when E < -DEADBAND. These are linear
    in the observation, so the encoder computes them directly.
    """
    e = {"x": 1.0, "p": -1.0, "pw": -0.5, "one": BALL_SIZE / 2.0, "vx": lead_frames}
    right = add(scaled(e, TRACK_GAIN), {"one": -TRACK_GAIN * DEADBAND + SCORE_OFFSET})
    left = add(scaled(e, -TRACK_GAIN), {"one": -TRACK_GAIN * DEADBAND + SCORE_OFFSET})
    b.encode("TR", **add(right, {"one": 0.5}))
    b.encode("TL", **add(left, {"one": 0.5}))
    b.encode("TR2", **right)
    b.encode("TL2", **left)


def decode_scores(b: Builder) -> None:
    """logit(NOOP) = 0, logit(LEFT/RIGHT) = gain * (score - offset)."""
    b.decode(1, TL=LOGIT_GAIN, ONE=-LOGIT_GAIN * (SCORE_OFFSET + 0.5))
    b.decode(2, TR=LOGIT_GAIN, ONE=-LOGIT_GAIN * (SCORE_OFFSET + 0.5))


def build_authored() -> Builder:
    b = Builder()

    # ---- Encoder: physical quantities, linear in the observation -----------
    b.encode("ONE", one=1.0)
    b.encode("VX", vx=1.0)
    b.encode("VY", vy=1.0)
    b.encode("BX", x=1.0, one=X_OFF)
    b.encode("BH", y=-1.0, one=LAND_Y + H_OFF)
    # q = p + pw/2 - 16: ball x at which ball centre == paddle centre.
    b.encode("BQ", p=1.0, pw=0.5, one=-BALL_SIZE / 2.0 + X_OFF)
    b.encode("ZY", y=-1.0, one=BRICK_BOTTOM)
    encode_tracking_scores(b, LEAD_FRAMES)

    # ---- Layer 1: three multiplications ------------------------------------
    # A falling ball lands (ignoring walls) at x + vx * (LAND_Y - y) / vy.
    # Multiplying through by vy > 0 removes the division:
    #     U = vy * x + vx * (LAND_Y - y)          (= vy * landing x)
    # so the network needs velocity-times-position products. Unit: candidate
    # = B - 0.5 >= 0 so m = g(.) = B; highway gate sigmoid(K*v) ~ 1/2 + K*v/4;
    # input channel is 0. Output = B/2 + (K/4) * v * B.
    b.candidate(1, "PC", BX=1.0, ONE=-0.5)
    b.gate(1, "PC", ONE=BIG)
    b.highway(1, "PC", VY=K_MUL)

    b.candidate(1, "PQ", BQ=1.0, ONE=-0.5)
    b.gate(1, "PQ", ONE=BIG)
    b.highway(1, "PQ", VY=K_MUL)

    b.candidate(1, "PB", BH=1.0, ONE=-0.5)
    b.gate(1, "PB", ONE=BIG)
    b.highway(1, "PB", VX=K_MUL)

    # Recover the products from the layer-1 outputs (linear combinations):
    #   v * B = (4/K) * P - (2/K) * B
    vy_bx = {"PC": 4.0 / K_MUL, "BX": -2.0 / K_MUL}        # vy * (x + X_OFF)
    vy_bq = {"PQ": 4.0 / K_MUL, "BQ": -2.0 / K_MUL}        # vy * (q + X_OFF)
    vx_bh = {"PB": 4.0 / K_MUL, "BH": -2.0 / K_MUL}        # vx * (h + H_OFF)
    vx_h = add(vx_bh, {"VX": -H_OFF})                      # vx * h
    U = add(vy_bx, {"VY": -X_OFF}, vx_h)                   # vy * x + vx * h

    # ---- Layer 2: fold the trajectory at the side walls --------------------
    # The ball's left edge lives in [0, X_MAX]; a wall bounce mirrors the path.
    # With u = U / vy the landing point is the triangle wave
    #   fold(u) = u + 2 relu(-u) - 2 relu(-u - X) - 2 relu(u - X) + 2 relu(u - 2X)
    # (exact for up to two bounces on either side). Scaled by vy it only needs
    # ReLUs of linear combinations of U and vy.
    b.feed_forward(2, "R1", **scaled(U, -RELU_GAIN))
    b.feed_forward(2, "R2", **scaled(add(U, {"VY": X_MAX}), -RELU_GAIN))
    b.feed_forward(2, "R3", **scaled(add(U, {"VY": -X_MAX}), RELU_GAIN))
    b.feed_forward(2, "R4", **scaled(add(U, {"VY": -2.0 * X_MAX}), RELU_GAIN))

    # D = vy * (landing x - q): how far the paddle target is from the landing
    # point, times vy. The X_OFF offsets cancel in vy*BX - vy*BQ.
    fold = scaled({"R1": 1.0, "R2": -1.0, "R3": -1.0, "R4": 1.0}, 2.0 / RELU_GAIN)
    D = add(vy_bx, scaled(vy_bq, -1.0), vx_h, fold)

    # ---- Layer 3: falling ball -> intercept, rising ball -> track ----------
    # Interception scores (px * px/frame): RIGHT if D > DEADBAND * vy, etc.
    # Highway gate sigmoid(BIG * vy) switches between the unit's own candidate
    # (interception score) and its input channel (tracking score).
    right = add(D, {"VY": -DEADBAND, "ONE": SCORE_OFFSET})
    left = add(scaled(D, -1.0), {"VY": -DEADBAND, "ONE": SCORE_OFFSET})
    b.candidate(3, "TR", **right)
    b.gate(3, "TR", ONE=BIG)
    b.highway(3, "TR", VY=BIG)
    b.candidate(3, "TL", **left)
    b.gate(3, "TL", ONE=BIG)
    b.highway(3, "TL", VY=BIG)

    # ---- Layer 4: but keep tracking while the ball is among the bricks -----
    # Above BRICK_BOTTOM a brick can still deflect a falling ball, so the
    # landing prediction is not trustworthy yet: stay under the ball, from
    # where any landing point is reachable in time (paddle 10.3 px/frame vs.
    # at most 5.3 px/frame of horizontal ball speed).
    b.candidate(4, "TR", TR2=1.0)
    b.gate(4, "TR", ONE=BIG)
    b.highway(4, "TR", ZY=BIG)
    b.candidate(4, "TL", TL2=1.0)
    b.gate(4, "TL", ONE=BIG)
    b.highway(4, "TL", ZY=BIG)

    decode_scores(b)
    return b


def build_naive_tracker() -> Builder:
    """Baseline: a linear ball follower in the same architecture.

    The encoder computes "ball centre minus paddle centre"; all four layers are
    wires; the decoder turns the sign into LEFT/RIGHT. No prediction, no lead.
    """
    b = Builder()
    b.encode("ONE", one=1.0)
    encode_tracking_scores(b, lead_frames=0.0)
    decode_scores(b)
    return b


def summarize(name: str, b: Builder, flat: np.ndarray) -> dict:
    tensors = {"encoder": b.encoder, "decoder": b.decoder}
    tensors.update({f"mingru{i + 1}": W for i, W in enumerate(b.layers)})
    return {
        "name": name,
        "revision": REVISION,
        "hidden_size": HIDDEN,
        "num_layers": LAYERS,
        "parameters": int(flat.size),
        "nonzero_parameters": int(np.count_nonzero(flat)),
        "nonzero_by_tensor": {k: int(np.count_nonzero(v)) for k, v in tensors.items()},
        "sha256": hashlib.sha256(flat.astype("<f4").tobytes()).hexdigest(),
        "channels": CHANNELS,
        "constants": {
            "X_MAX": X_MAX, "LAND_Y": LAND_Y, "DEADBAND": DEADBAND,
            "LEAD_FRAMES": LEAD_FRAMES, "X_OFF": X_OFF, "H_OFF": H_OFF,
            "K_MUL": K_MUL, "RELU_GAIN": RELU_GAIN, "SCORE_OFFSET": SCORE_OFFSET,
            "TRACK_GAIN": TRACK_GAIN, "LOGIT_GAIN": LOGIT_GAIN, "BIG": BIG,
            "BRICK_BOTTOM": BRICK_BOTTOM, "PADDLE_STEP": PADDLE_STEP,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "weights")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {}
    for name, build in [("authored_breakout", build_authored),
            ("naive_tracker", build_naive_tracker)]:
        b = build()
        flat = b.flat()
        assert flat.size == weight_count(HIDDEN, LAYERS), flat.size
        assert np.all(np.isfinite(flat))
        flat.astype("<f4").tofile(args.out_dir / f"{name}.bin")
        manifest[name] = summarize(name, b, flat)
        info = manifest[name]
        print(f"{name}.bin  {info['parameters']} parameters, "
            f"{info['nonzero_parameters']} nonzero, sha256 {info['sha256'][:16]}...")
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
