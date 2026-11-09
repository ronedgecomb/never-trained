# Source this file:  source scripts/env.sh
# Shared paths for setup, build and evaluation.
#
# RLFUN_HOME holds everything that is fetched or built (PufferLib checkout,
# raylib, the Python venv, binaries). It defaults to a directory on the Linux
# filesystem because builds and venvs are slow on /mnt/c under WSL2.

export RLFUN_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RLFUN_HOME="${RLFUN_HOME:-$HOME/.cache/rl-fun}"
export PUFFER_ROOT="$RLFUN_HOME/PufferLib"
export PUFFER_COMMIT="ea7f989371270d2948f96728bda8aec5e7634a4c"
export RLFUN_BIN="$RLFUN_HOME/bin"
export RLFUN_VENV="$RLFUN_HOME/venv"
export REFERENCE_WEIGHTS="$RLFUN_HOME/reference_weights/breakout_weights.bin"

# User-space copies of link-time libraries that a stock Ubuntu install lacks
# (see scripts/setup.sh). Harmless if the system already provides them.
export LIBRARY_PATH="$RLFUN_HOME/lib:$RLFUN_HOME/sysroot/usr/lib/x86_64-linux-gnu${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$RLFUN_HOME/sysroot/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
