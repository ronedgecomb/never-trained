# Evaluation protocol

Written before the final evaluation was run. The policy was frozen by hash
(`eval/FROZEN.json`) before any final seed was used, and `eval/run_eval.py`
refuses to run the final phase against different weights or to overwrite
existing final results.

## Environment

- PufferLib 5.0, Ocean `breakout`, commit `ea7f989371270d2948f96728bda8aec5e7634a4c`.
- Configuration: `config/default.ini` + `config/breakout.ini`, no overrides.
  Field 576×330, 18×6 bricks, ball 32×32, paddle 62 px wide (31 px after the
  ball first touches the ceiling), ball speed 256–448 px/s, paddle speed
  620 px/s, `frameskip = 3`, discrete actions NOOP / LEFT / RIGHT.
- One agent step = 3 simulated frames (the training-time semantics: the env's
  own frameskip). Observation: 118 floats (paddle, ball, counters, 108 brick flags).
- Reward = points for each brick destroyed (7, 4 or 1 by row). Clearing all 108
  bricks brings them back once; maximum score 864.
- Episode ends at 864 points or when the sixth ball is lost. The environment
  has no time limit of its own; the harness truncates at 30,000 agent steps
  (90,000 frames, 25 minutes of game time) and reports truncations.
- The environment's only randomness is the serve direction (left/right),
  drawn from a per-episode seed.

## Policies

| name | what it is |
|---|---|
| `authored` | Weights written directly by Claude; `policy/gen_weights.py`. No training. |
| `naive_tracker` | Authored baseline: a linear "follow the ball" rule in the same architecture. |
| `pufferai_pretrained` | PufferAI's published Breakout checkpoint (trained with PufferLib's RL trainer). |
| `random` | Uniformly random action each agent step. |

All three network policies use the same architecture (PufferNet, hidden 32,
4 layers, 16,192 parameters) and run through PufferLib's own inference code
(`forward_puffernet` in `src/puffercpu.c`).

## Protocols

| name | description |
|---|---|
| `standard` | The environment as configured. Actions are sampled from softmax(logits) exactly as PufferLib's inference code does. |
| `sticky25` | Stress test. Each agent step, with probability 0.25 the previously applied action is repeated instead of the policy's choice. Neither network was built or trained for this. |
| `greedy` | Standard environment with argmax actions instead of sampling (network policies only). |

`standard` is the primary protocol.

## Seeds

- Development: seeds 1–20 for every policy and protocol, plus seeds 1–300 for
  `authored` under `standard`.
- Final: seeds 1001–1100, 100 episodes per policy and protocol. Not run or
  inspected before the freeze.

The seed sets the environment's serve-direction generator, PufferLib's action
sampling generator, and (independently) the sticky-action generator.

## Reported metrics

Per policy and protocol: mean, standard deviation, minimum and median of the
episode score (which equals the episode return); 95% bootstrap interval of the
mean; number of episodes that reach 864 with a 95% Wilson interval; mean balls
lost per episode; mean episode length in frames; truncations; number of
distinct trajectories.

## Known property of the standard protocol

A policy that never loses a ball sees only one serve per episode, so a
deterministic policy has two possible trajectories. Sampling from the softmax
adds a little variation for the authored policy (it is near-deterministic) and
a lot for the pretrained one. The `sticky25` protocol exists partly because it
produces genuinely different episodes for every seed.
