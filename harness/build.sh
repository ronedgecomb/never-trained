#!/bin/bash
# Builds the evaluation harness against the PufferLib checkout.
# Usage: source scripts/env.sh && harness/build.sh
set -euo pipefail

: "${PUFFER_ROOT:?source scripts/env.sh first}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAYLIB="$PUFFER_ROOT/raylib-5.5_linux_amd64"
mkdir -p "$RLFUN_BIN"

FLAGS=(
    -O2 -Wall -Wno-unused-function -Wno-narrowing -Wno-unreachable-code
    -mavx2 -mfma
    -I"$PUFFER_ROOT" -I"$PUFFER_ROOT/src" -I"$PUFFER_ROOT/ocean/breakout"
    -I"$PUFFER_ROOT/vendor" -I"$RAYLIB/include"
    -DPLATFORM_DESKTOP
)
LIBS=("$RAYLIB/lib/libraylib.a" -lGL -lm -lpthread -ldl)

echo "Compiling bk_eval..."
clang "${FLAGS[@]}" "$HERE/bk_eval.c" "${LIBS[@]}" -o "$RLFUN_BIN/bk_eval"
echo "Built into $RLFUN_BIN"
