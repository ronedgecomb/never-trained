#!/bin/bash
# Runs PufferLib's own stock evaluator (the ./breakout binary built by
# `./build.sh breakout --cpu`) on each checkpoint. This is not the seeded
# harness: the stock binary uses its own fixed seeding and plays episodes back
# to back, and in this mode it re-evaluates the policy right after any reward.
# It is here as an independent check that the checkpoints load and score in
# unmodified PufferLib.
#
# Usage: source scripts/env.sh && eval/stock_check.sh
set -euo pipefail
: "${PUFFER_ROOT:?source scripts/env.sh first}"
OUT="$RLFUN_REPO/eval/results/stock_binary.txt"
cd "$PUFFER_ROOT"
{
    echo "PufferLib commit $(git rev-parse HEAD)"
    echo "command: ./breakout WEIGHTS --headless --eval_episodes=100"
    for name in authored_breakout naive_tracker; do
        echo "== $name"
        ./breakout "$RLFUN_REPO/policy/weights/$name.bin" --headless --eval_episodes=100 | tail -1
    done
    echo "== pufferai_pretrained"
    ./breakout "$REFERENCE_WEIGHTS" --headless --eval_episodes=100 | tail -1
} > "$OUT"
cat "$OUT"
