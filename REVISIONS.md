# Development log

What was looked at, in what order, and what was (and was not) changed. The
short version:

- **Revisions: 1.** The version evaluated is the first version written. No
  weight changed after its first rollout.
- **Rollout feedback was used**, for validation and for the decision to stop.
  That is why this is described as a *first draft that was not revised*, not as
  "zero-shot".
- **No training of any kind**: no gradients, no reinforcement learning, no
  evolutionary or random search, no fitting to rollouts, no distillation, no
  pretrained weights. No constant was tuned against results.

## 1. Before any weights existed

Sources read (PufferLib 5.0, commit `ea7f989`):

- `src/puffercpu.c`: the inference code. This fixed the architecture
  (bias-free encoder → MinGRU-with-highway layers → bias-free decoder), the
  checkpoint layout, and the fact that actions are sampled from the softmax.
- `ocean/breakout/breakout.h`: the environment. Observation layout, paddle and
  ball motion, wall/brick/paddle collisions, the bounce-angle rule, scoring,
  the paddle shrinking after the first ceiling hit, the brick refresh at 432.
- `config/default.ini`, `config/breakout.ini`: sizes, speeds, `frameskip = 3`,
  network shape (hidden 32, 4 layers).
- `src/pufferenv.h`, `src/ini.h`, `build.sh`: to build and drive the env.

Rollouts run before the policy existed (none involve the authored weights):

- PufferLib's stock binary with no weights (smoke test; scores 0).
- PufferAI's pretrained checkpoint with the stock binary (20 episodes, mean
  837.4), and with the harness on seeds 1–5 (864 each, 3–4 balls lost).
- Random and no-op policies on seeds 1–5 (0–2 points).

These established how strong the reference is. They did not feed into the
weights, which came from the source code: the design needs the env's geometry
and physics, and those are stated exactly in `breakout.h`.

Environments considered and set aside: Tetris (a placement-evaluation circuit
would be far larger and its result less certain), Pong, Space Invaders,
SlimeVolley. Breakout was chosen for its clear win condition, a published
trained reference in the same architecture, and physics simple enough to
reason about exactly.

## 2. Version 1

Written in one pass as `policy/gen_weights.py` (network design in
[`policy/DESIGN.md`](policy/DESIGN.md)). Every constant has its reasoning in a
comment next to it; all were fixed at this point.

Checked without the environment: `policy/test_weights.py` pushes 200,000
synthetic states through the network and compares with a plain-math statement
of the intended controller (`policy/controller_spec.py`). Result: no action
disagreements; the network's internal landing estimate is within 0.08 px of
the formula.

One correction was made here, to the test and not to the weights: the first
run of the test reported a failure because it also compared states inside the
half-pixel band where the layer-4 switch is mid-way between its two inputs.
The test now skips that band. The weight file was already
`sha256 2a3c22fb…` before that fix and is the file evaluated.

## 3. First rollout

Development seeds 1–20, standard environment: 864 in all 20 games, no ball
lost. The naive ball-follower (written at the same time as a baseline) also
cleared all 20, losing about one ball per game.

## 4. Further development rollouts (all on development seeds)

| What | Seeds | Why |
|---|---|---|
| All four policies, standard / sticky actions / greedy | 1–20 | The development table in `RESULTS.md` |
| Authored, standard | 1–300 | Looking for rare misses (none: 300/300, no ball lost) |
| Random-action starts (first ≤30 steps random) | 1–20 | Explored as a stress protocol, then dropped: the first serve lands after 20 steps, so longer random prefixes make the first ball unreachable for any policy. Authored 864 (0.30 balls lost), pretrained 864 (2.70) |
| Brick-count-over-time traces, authored and pretrained | 1–6 | To see where game time goes (the last bricks of each wall) |
| Per-step traces of lost balls under sticky actions | 1–20 | Failure analysis in `RESULTS.md` |
| PufferLib's stock binary, 100 episodes per checkpoint | its own seeding | Independent check (`eval/results/stock_binary.txt`) |
| NumPy forward pass vs. PufferLib's C forward pass | 1–2 | `eval/crosscheck.py` |

## 5. Changes considered and not made

- **Aiming at the remaining bricks.** The traces showed that both the authored
  and the trained policy spend a large share of each wall on its last few
  bricks. A design was worked out on paper (find the leftmost surviving column
  pair from the 108 brick inputs, bias the contact point toward it, tighten
  the dead band in the final frames), and estimated to shorten games by
  roughly a fifth. It was not built. Version 1 already never missed, and the
  change would have been a revision driven by rollout results, with more
  moving parts near the moment of contact.
- **Hardening against sticky actions.** The losses under the stress test come
  from runs of repeated moves carrying the paddle past the landing point. No
  change was made in response; the stress-test numbers are for the same
  untouched weights.

## 6. Freeze and final evaluation

- Frozen `2026-11-09T21:56:28Z` (`eval/FROZEN.json`): `eval/freeze.py`
  regenerated the weights from the script, confirmed they were byte-identical,
  and recorded the hashes.
- Final evaluation started `2026-11-09T21:56:31Z` on seeds 1001–1100, which had
  not been run before. Results in `eval/results/final/`.
- Nothing about the policy was changed afterwards.

## 7. After the final evaluation

- The harness was reorganised (the episode runner moved into
  `harness/bk_common.h`), and a `--verify` mode was added to
  `eval/run_eval.py`. Re-running all 1,100 final episodes with the reorganised
  harness reproduces the stored log row for row
  (`eval/results/final/reproduction_check.json`). The same check passes from a
  fresh `scripts/setup.sh` install.

## Honest framing

The weights are an analytically designed controller, written as numbers in the
shape PufferLib's Breakout policy uses. Claude chose every non-zero weight by
reasoning from the simulator's source code. What makes that possible is that
the task's physics are fully specified and simple; this is not evidence that
weights can be hand-written for problems where they are not.
