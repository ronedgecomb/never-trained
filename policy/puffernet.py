"""NumPy reference implementation of PufferLib 5.0's PufferNet inference.

This mirrors `forward_puffernet` in PufferLib's src/puffercpu.c line for line:

    encoder (bias-free linear)  ->  N x MinGRU-with-highway  ->  decoder (bias-free linear)

It exists so the network can be read, tested and inspected without the C
toolchain. Evaluation itself runs PufferLib's own C code; `tests` in
eval/crosscheck.py confirm that the two implementations agree.

Weight file format (PufferLib `.bin` checkpoint): flat little-endian float32,
tensors in this order, each padded to a multiple of 8 floats:

    encoder   [hidden, obs]
    decoder   [num_logits + 1, hidden]      (last row is the value head)
    mingru[l] [3 * hidden, hidden]          rows: candidate | gate | highway
"""

from __future__ import annotations

import numpy as np

OBS_SIZE = 118
NUM_LOGITS = 3  # NOOP, LEFT, RIGHT


def _align8(n: int) -> int:
    return (n + 7) & ~7


def weight_count(hidden: int, layers: int, obs: int = OBS_SIZE, logits: int = NUM_LOGITS) -> int:
    n = _align8(hidden * obs)
    n = _align8(n + (logits + 1) * hidden)
    for _ in range(layers):
        n = _align8(n + 3 * hidden * hidden)
    return n


def pack(encoder: np.ndarray, decoder: np.ndarray, mingru: list[np.ndarray]) -> np.ndarray:
    """Flatten tensors into PufferLib's checkpoint layout."""
    chunks = []
    for tensor in [encoder, decoder, *mingru]:
        flat = np.asarray(tensor, dtype=np.float32).ravel()
        pad = _align8(flat.size) - flat.size
        chunks.append(np.concatenate([flat, np.zeros(pad, np.float32)]))
    return np.concatenate(chunks)


def unpack(flat: np.ndarray, hidden: int, layers: int, obs: int = OBS_SIZE,
        logits: int = NUM_LOGITS):
    flat = np.asarray(flat, dtype=np.float32)
    idx = 0

    def take(rows, cols):
        nonlocal idx
        n = rows * cols
        out = flat[idx:idx + n].reshape(rows, cols)
        idx = _align8(idx + n)
        return out

    encoder = take(hidden, obs)
    decoder = take(logits + 1, hidden)
    mingru = [take(3 * hidden, hidden) for _ in range(layers)]
    return encoder, decoder, mingru


def _sigmoid(x: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        return (np.float32(1.0) / (np.float32(1.0) + np.exp(-x))).astype(np.float32)


class PufferNet:
    """Single-agent PufferNet. Call `forward(obs)` once per agent step."""

    def __init__(self, flat_weights: np.ndarray, hidden: int = 32, layers: int = 4):
        self.hidden = hidden
        self.layers = layers
        self.encoder, self.decoder, self.mingru = unpack(flat_weights, hidden, layers)
        self.state = np.zeros((layers, hidden), np.float32)

    @classmethod
    def load(cls, path: str, hidden: int = 32, layers: int = 4) -> "PufferNet":
        flat = np.fromfile(path, dtype="<f4")
        assert flat.size == weight_count(hidden, layers), (flat.size, weight_count(hidden, layers))
        return cls(flat, hidden, layers)

    def reset(self) -> None:
        self.state[:] = 0

    def forward(self, obs: np.ndarray, terminal: bool = False, trace: dict | None = None) -> np.ndarray:
        """Returns the 4 decoder outputs: logits for NOOP/LEFT/RIGHT, then value."""
        H = self.hidden
        if terminal:
            self.reset()
        x = self.encoder @ np.asarray(obs, np.float32)
        if trace is not None:
            trace["encoder"] = x.copy()
        for layer, W in enumerate(self.mingru):
            combined = W @ x
            candidate, gate, highway = combined[:H], combined[H:2 * H], combined[2 * H:]
            s = self.state[layer]
            gate_s = _sigmoid(gate)
            # g(x) from the minGRU paper: linear above zero, sigmoid below.
            h_tilde = np.where(candidate >= 0, candidate + np.float32(0.5), _sigmoid(candidate))
            mingru_out = (s + gate_s * (h_tilde - s)).astype(np.float32)
            hw_s = _sigmoid(highway)
            out = (hw_s * mingru_out + (np.float32(1.0) - hw_s) * x).astype(np.float32)
            self.state[layer] = mingru_out
            if trace is not None:
                trace[f"layer{layer + 1}"] = out.copy()
                trace[f"layer{layer + 1}_highway_gate"] = hw_s.copy()
            x = out
        result = self.decoder @ x
        if trace is not None:
            trace["decoder"] = result.copy()
        return result

    def act(self, obs: np.ndarray, terminal: bool = False) -> int:
        """Greedy action. PufferLib samples from softmax(logits); with this
        policy's logit margins the two agree except within a hair of a tie."""
        return int(np.argmax(self.forward(obs, terminal)[:NUM_LOGITS]))
