# How the authored weights play Breakout

The policy is a file of 16,192 float32 numbers in PufferLib's checkpoint
format. 233 of them are non-zero. This note explains what those 233 do. The
generator is [`gen_weights.py`](gen_weights.py); the complete list of non-zero
weights, by name, is in
[`weights/authored_breakout_nonzero.txt`](weights/authored_breakout_nonzero.txt).

## The network it has to fit into

PufferLib's Breakout policy ("PufferNet", `src/puffercpu.c`), in the shape its
config uses:

```
obs[118] ──encoder 32×118──► x[32] ──4 × layer 96×32──► x[32] ──decoder 4×32──► 3 logits + value
```

There are no bias terms anywhere. One layer computes, for each of its 32
units *j*:

```
cand, gate, hw = W @ x                       three linear projections of the layer input
g(c)   = c + 0.5        if c >= 0
         sigmoid(c)     otherwise
m      = state + sigmoid(gate) * (g(cand) - state)          (state <- m)
out_j  = sigmoid(hw) * m + (1 - sigmoid(hw)) * x_j          (highway connection)
```

PufferLib then samples the action from `softmax(logits)`.

## Five tricks

Everything below is built from five ways of using one unit.

| Need | How |
|---|---|
| **A constant** (there are no biases) | `obs[1]` is `paddle_y / height`, which never changes (312/330). One encoder weight turns it into a channel that is always 1. |
| **A wire** (carry a value to the next layer) | `hw = −100·ONE`. The highway gate is exactly 0, so the unit outputs its input channel unchanged. This is the default for every unit. |
| **A ReLU** | `gate = +100·ONE`, `hw = +100·ONE`: no memory, no bypass, so the output is `g(cand)`: equal to `cand + 0.5` when positive and essentially 0 when clearly negative. |
| **A switch** | `hw = +100·s`. The output is the unit's own value when `s > 0` and its input channel when `s < 0`. |
| **A multiplier** | `hw = 0.004·v`. For small arguments `sigmoid(z) ≈ ½ + z/4`, so the unit outputs `(½ + 0.001·v)·m`. The next layer subtracts the `½·m` part (it has `m` on a wire) and is left with `v × m`. With `|v| ≤ 7.5` px/frame the approximation is good to 1 part in 10,000. |

The recurrent state is never used: every unit that computes something has its
gate saturated open, so the policy is purely feed-forward.

## What it computes

Units: pixels and frames. `x, y` are the ball's top-left corner, `vx, vy` its
velocity per frame, `p` the paddle's left edge, `pw` its width.

**Encoder: physical quantities.** Each observation is scaled back to pixels.

| Channel | Value |
|---|---|
| `ONE` | 1 |
| `VX`, `VY` | ball velocity |
| `BX` | `x + 50` |
| `BH` | `337 − y` = height left to fall + 60 |
| `BQ` | `q + 50`, where `q = p + pw/2 − 16` is the ball position that would be centred on the paddle |
| `ZY` | `122 − y`: positive while the ball is level with or above the bricks |
| `TR`, `TL` | "follow the ball" scores for RIGHT and LEFT (see below) |

The offsets (50, 60) keep the quantities positive, where `g` is exactly linear.

**Layer 1: multiply.** A falling ball, ignoring walls, lands at
`x + vx·(277 − y)/vy`. (277 rather than 280 because the env computes the
bounce from the ball's position at the start of the contact frame, about half
a frame early.) Multiplying through by `vy > 0` removes the division:

```
U = vy·x + vx·(277 − y)          U / vy is the landing position
```

Three multiplier units produce `vy·(x+50)`, `vy·(q+50)` and `vx·(337−y)`.

**Layer 2: fold.** The ball's left edge lives in `[0, 544]`, and a wall bounce
mirrors the path. The landing position is a triangle wave of `U/vy`, which is
a sum of ReLUs. Scaled by `vy`, the kinks sit at multiples of `544·vy`, so
four ReLU units of linear combinations of `U` and `vy` are enough for up to
two bounces on either side:

```
fold(U) = U + 2·relu(−U) − 2·relu(−U − 544·vy) − 2·relu(U − 544·vy) + 2·relu(U − 2·544·vy)
```

**Layer 3: falling → intercept.** `D = fold(U) − vy·q` is how far the landing
point is from where the paddle is centred, times `vy`. The paddle moves 31 px
per agent step, so the right dead band is half of that:

```
score_right = D − 15.5·vy        score_left = −D − 15.5·vy
```

The two score units are switches on the sign of `vy`: a falling ball selects
these interception scores; a rising ball keeps the follow scores that arrived
on the same channels from the encoder,

```
E = (x − q) + 3·vx               ball minus paddle, one agent step ahead
score_right = 4·(E − 15.5)       score_left = 4·(−E − 15.5)
```

(A constant 8192 is added to all four so the switch inputs stay positive; the
decoder removes it.)

**Layer 4: but not while among the bricks.** A falling ball above `y = 122` can
still be deflected by a brick, so its predicted landing point is not yet
reliable. A second switch, on the sign of `ZY`, puts the follow scores back in
that case. Staying under the ball is safe: the paddle moves 10.3 px per frame
and the ball at most 5.3 px sideways, so from under the ball any landing point
is reachable in time.

**Decoder.** `logit(NOOP) = 0`, `logit(LEFT) = 12·score_left`,
`logit(RIGHT) = 12·score_right`. At most one score is positive. The gain of 12
makes a 1 px difference worth more than 24 logits, so sampling from the
softmax picks the largest logit except within a hair of a tie.

## Why it does not miss

- The landing prediction is exact up to float rounding (checked against the
  formulas to 0.08 px), because the env's bounce handling keeps the ball on the
  mirrored straight-line path.
- The paddle can always get there: from under the ball, the furthest landing
  point is 158·tan(45°) = 158 px away with at least 30 frames to cover it at
  10.3 px per frame.
- It arrives early and stops. With a dead band of half a step, the paddle
  settles on the reachable position nearest the landing point, so the ball
  meets the paddle within 15.5 px of its centre. On the 31 px paddle that keeps
  the bounce within about 45° of vertical.

## What it does not do

It ignores the 108 brick inputs and never aims. Its only skill is to never
miss; the bricks fall because the ball keeps coming back.

## Checks

```
python policy/gen_weights.py        # writes the checkpoints (deterministic)
python policy/test_weights.py       # 200,000 states: network vs. plain-math specification
python policy/describe_weights.py   # every non-zero weight, by name
python eval/crosscheck.py           # NumPy forward pass vs. PufferLib's C forward pass
```

[`controller_spec.py`](controller_spec.py) states the intended behaviour in
ten lines of ordinary arithmetic. It is the specification the weights are
tested against, and is not used when the policy runs.
