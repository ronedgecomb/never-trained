#!/bin/bash
# One-time setup. Everything fetched or built goes under $RLFUN_HOME
# (default ~/.cache/rl-fun); the repository itself is not modified.
#
# Usage: scripts/setup.sh
#
# Needs: git, curl, unzip, clang and uv (https://docs.astral.sh/uv/). No root
# access is required.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"
mkdir -p "$RLFUN_HOME" "$RLFUN_BIN" "$RLFUN_HOME/lib"

echo "[1/6] PufferLib at commit ${PUFFER_COMMIT:0:12}"
if [ ! -d "$PUFFER_ROOT/.git" ]; then
    git init -q "$PUFFER_ROOT"
    git -C "$PUFFER_ROOT" remote add origin https://github.com/PufferAI/PufferLib.git
fi
if [ "$(git -C "$PUFFER_ROOT" rev-parse -q --verify HEAD 2>/dev/null || true)" != "$PUFFER_COMMIT" ]; then
    git -C "$PUFFER_ROOT" fetch -q --depth 1 origin "$PUFFER_COMMIT"
    git -C "$PUFFER_ROOT" checkout -q FETCH_HEAD
fi

echo "[2/6] link-time libraries (libGL, OpenMP runtime)"
# PufferLib's build links -lGL and, through clang -fopenmp, -lomp. A stock
# Ubuntu install has the runtime libraries but not always the link names, so
# fall back to user-space copies (env.sh puts them on LIBRARY_PATH).
links() { echo 'int main(void){return 0;}' | clang -x c - "$@" -o /dev/null 2>/dev/null; }
if ! links -lGL; then
    gl="$(ldconfig -p | awk '/libGL\.so\.1 /{print $NF; exit}')"
    [ -n "$gl" ] || { echo "libGL.so.1 not found (install libgl1)"; exit 1; }
    ln -sf "$gl" "$RLFUN_HOME/lib/libGL.so"
fi
if ! links -fopenmp; then
    dev="$(apt-cache depends libomp-dev | awk '/Depends: libomp/{print $2; exit}')"
    run="$(apt-cache depends "$dev" | awk '/Depends: libomp/{print $2; exit}')"
    mkdir -p "$RLFUN_HOME/debs" "$RLFUN_HOME/sysroot"
    (cd "$RLFUN_HOME/debs" && apt-get download "$dev" "$run" >/dev/null)
    for deb in "$RLFUN_HOME"/debs/*.deb; do dpkg -x "$deb" "$RLFUN_HOME/sysroot"; done
    links -fopenmp || { echo "could not provide libomp (install libomp-dev)"; exit 1; }
fi

echo "[3/6] PufferLib's stock Breakout binary (fetches raylib 5.5 on first run)"
(cd "$PUFFER_ROOT" && ./build.sh breakout --cpu >/dev/null)

echo "[4/6] reference checkpoint published by PufferAI"
mkdir -p "$(dirname "$REFERENCE_WEIGHTS")"
if [ ! -f "$REFERENCE_WEIGHTS" ]; then
    curl -fsSL -o "$REFERENCE_WEIGHTS" \
        "https://raw.githubusercontent.com/PufferAI/puffer.ai/4856d5943797f8c136286e0cdcf4168f0c0763be/docs/assets/models/breakout_weights.bin"
fi
echo "859f57a45d5b51a39ca94e1601cc9ff572f5b26ea5c4ecec7853ca45a258dea3  $REFERENCE_WEIGHTS" \
    | sha256sum -c --quiet -

echo "[5/6] Python environment"
[ -x "$RLFUN_VENV/bin/python" ] || uv venv -q --python 3.12 "$RLFUN_VENV"
VIRTUAL_ENV="$RLFUN_VENV" uv pip install -q -r "$RLFUN_REPO/requirements.txt"

echo "[6/6] evaluation harness"
"$RLFUN_REPO/harness/build.sh" >/dev/null

echo "Ready. Next:  source scripts/env.sh && \$RLFUN_VENV/bin/python eval/run_eval.py --phase final --verify"
