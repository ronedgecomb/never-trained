# Never trained: weights for Breakout, written by an LLM

A neural network that plays PufferLib's Breakout perfectly, whose weights were
written directly by Claude from the environment's source code. No gradient
descent, no reinforcement learning, no search, no training data.

The checkpoint has the same architecture and size as PufferLib's own trained
Breakout policy (16,192 parameters) and is loaded and run by PufferLib's own
inference code. 233 of its weights are non-zero.

## Result

100 held-out games per policy (seeds 1001–1100), PufferLib 5.0 Breakout,
default configuration. Maximum score 864.

| Policy | Mean score | Games cleared | Balls lost per game |
|---|---|---|---|
| **Authored** (written by Claude, no training) | **864.0** | **100 / 100** | **0.00** |
| PufferAI's pretrained checkpoint (trained with RL) | 853.1 | 92 / 100 | 3.23 |
| Naive ball-follower (authored baseline) | 864.0 | 100 / 100 | 0.89 |
| Random actions | 0.6 | 0 / 100 | 6.00 |

Stress test with sticky actions (25% of agent steps repeat the previous
action; neither network was built for it):

| Policy | Mean score | Games cleared |
|---|---|---|
| **Authored** | **846.4** | **84 / 100** |
| Naive ball-follower | 574.8 | 7 / 100 |
| PufferAI's pretrained checkpoint | 297.2 | 0 / 100 |

Full tables, confidence intervals, development results, failure cases and
limitations: [`RESULTS.md`](RESULTS.md). Per-episode logs:
[`eval/results/`](eval/results/).

Two things to know before quoting these numbers:

- The standard environment is nearly deterministic (the only randomness is the
  serve direction), so "100 of 100" mostly means the policy is flawless on the
  two games that environment can produce. The sticky-action test is where
  every seed is a different game.
- Score saturates: even the naive ball-follower clears every standard game.
  The differences are in balls lost and in robustness.

## What this is

An analytically designed controller, expressed as the weights of a standard
network. Claude read `breakout.h`, worked out how to predict where the ball
will land, and wrote the numbers that make PufferLib's bias-free
MinGRU-with-highway network compute it:

1. **Layer 1 multiplies.** A sigmoid gate whose input `z` stays near zero is
   approximately `½ + z/4`, which turns one unit into a multiplier: ball
   velocity × position.
2. **Layer 2 folds.** Four ReLU-like units mirror the predicted path at the
   side walls.
3. **Layers 3–4 choose.** Rising ball, or still among the bricks: stay under
   it. Falling with a clear path: go to the predicted landing point.
4. **The decoder** turns the sign of the error into LEFT / STAY / RIGHT.

Details: [`policy/DESIGN.md`](policy/DESIGN.md). Every non-zero weight, by
name: [`policy/weights/authored_breakout_nonzero.txt`](policy/weights/authored_breakout_nonzero.txt).

It is not a learned policy, it reads the simulator's state vector rather than
pixels, and it works because the game's physics are simple and fully
specified. It ignores the bricks and never aims; it wins by never missing.

**Development.** The version evaluated is the first version written; no weight
changed after its first rollout. Development rollouts on separate seeds were
used to validate it and to decide to stop, so this is described as an
unrevised first draft, not as zero-shot. The full log is in
[`REVISIONS.md`](REVISIONS.md); the evaluation protocol, fixed before the
final run, is in [`eval/PROTOCOL.md`](eval/PROTOCOL.md).

## Reproduce

Tested on Ubuntu 26.04 (x86-64, WSL2). Needs `git`, `curl`, `unzip`, `clang`
and [`uv`](https://docs.astral.sh/uv/). No GPU and no root access.

```bash
scripts/setup.sh                     # ~1.5 min: PufferLib at a pinned commit, raylib, venv, harness
source scripts/env.sh
PY=$RLFUN_VENV/bin/python

$PY policy/gen_weights.py            # rebuild the checkpoints (byte-identical to policy/weights/)
$PY policy/test_weights.py           # weights vs. a plain-math statement of the controller
$PY eval/crosscheck.py               # NumPy forward pass vs. PufferLib's C forward pass
$PY eval/run_eval.py --phase final --verify   # re-run all 1,100 final episodes, compare with the logs
eval/stock_check.sh                  # PufferLib's own evaluator: perf=1.000000
$PY eval/report.py                   # regenerate RESULTS.md from the logs
```

Watch it play in PufferLib's own viewer (needs a display):

```bash
cd $PUFFER_ROOT && ./breakout $RLFUN_REPO/policy/weights/authored_breakout.bin
```

`--verify` re-runs the stored evaluation and reports whether every episode row
matches. It does on the machine above and from a fresh `setup.sh` install. The
simulation is chaotic, so a different compiler or math library may produce
different individual trajectories.

Everything fetched or built lives under `$RLFUN_HOME` (default
`~/.cache/rl-fun`), outside this repository.

### Versions

| Component | Version |
|---|---|
| PufferLib | 5.0 branch, commit `ea7f989371270d2948f96728bda8aec5e7634a4c` (2026-10-14) |
| raylib | 5.5 (fetched by PufferLib's `build.sh`) |
| Reference checkpoint | `PufferAI/puffer.ai` @ `4856d59`, `docs/assets/models/breakout_weights.bin`, sha256 `859f57a4…` |
| Compiler | clang 21.1.8 |
| Python | 3.12.16, numpy 2.5.4 ([`requirements.txt`](requirements.txt)) |

## Layout

```
policy/     gen_weights.py (the generator), puffernet.py (NumPy inference), controller_spec.py,
            test_weights.py, describe_weights.py, DESIGN.md, weights/ (the checkpoints)
harness/    bk_eval.c, bk_common.h: seeding and logging around PufferLib's unmodified
            environment and inference code
eval/       PROTOCOL.md, run_eval.py, freeze.py, crosscheck.py, failure_analysis.py, report.py,
            stock_check.sh, FROZEN.json, results/ (per-episode logs and summaries)
scripts/    setup.sh, env.sh
RESULTS.md  REVISIONS.md
```

## Credits and licence

[PufferLib](https://puffer.ai) provides the environment and the inference
code, and the reference checkpoint is PufferAI's; `scripts/setup.sh` fetches
all three. `policy/puffernet.py` is a NumPy port of PufferLib's forward pass.
PufferLib is MIT-licensed: [`licenses/PufferLib.txt`](licenses/PufferLib.txt).

This repository is licensed under the [Apache License 2.0](LICENSE).
