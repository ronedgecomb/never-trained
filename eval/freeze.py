#!/usr/bin/env python3
"""Freezes the authored policy before the final evaluation.

Regenerates the weights from policy/gen_weights.py into a temporary directory,
checks they are byte-identical to the committed files (so the generator is the
true source of the weights), and records hashes in eval/FROZEN.json.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FROZEN = REPO / "eval" / "FROZEN.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    if FROZEN.exists():
        print("eval/FROZEN.json already exists; the policy is frozen")
        return 1
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, str(REPO / "policy/gen_weights.py"),
            "--out-dir", tmp], check=True, capture_output=True)
        for name in ["authored_breakout.bin", "naive_tracker.bin"]:
            if (Path(tmp) / name).read_bytes() != (REPO / "policy/weights" / name).read_bytes():
                print(f"{name}: regenerated weights differ from policy/weights/{name}")
                return 1
    record = {
        "frozen_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "authored_weights_sha256": sha256(REPO / "policy/weights/authored_breakout.bin"),
        "naive_tracker_weights_sha256": sha256(REPO / "policy/weights/naive_tracker.bin"),
        "gen_weights_py_sha256": sha256(REPO / "policy/gen_weights.py"),
        "puffernet_py_sha256": sha256(REPO / "policy/puffernet.py"),
        "revision": 1,
        "note": "Weights regenerate byte-identically from policy/gen_weights.py. "
                "Final seeds 1001-1100 had not been run when this file was written.",
    }
    FROZEN.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
