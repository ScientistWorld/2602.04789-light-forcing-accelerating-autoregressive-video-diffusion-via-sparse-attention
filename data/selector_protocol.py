"""Client side of the sparse-attention mask-selector protocol.

A *method* in this benchmark is a mask selector: a program that the evaluator
starts as a subprocess and asks, for every attention call of an autoregressive
rollout, which (query block, key block) tiles to compute. The evaluator owns the
model, the inputs, the attention computation, the tile accounting and all
metrics; the selector only ever sees block-pooled summaries.

Usage (see method/ and baseline/ for complete selectors)::

    from data.selector_protocol import serve

    class MySelector:
        def begin_rollout(self, meta: dict) -> None: ...
        def select(self, call) -> np.ndarray: ...   # bool/uint8 [H, nqb, nkb]
        def end_rollout(self) -> None: ...          # optional

    if __name__ == "__main__":
        serve(MySelector())

Register it so the evaluators run it (scripts/register_method.sh):
``outputs/methods/<name>.json`` = ``{"command": ["python", "-m", "my.module"]}``.

``call`` attributes:
    chunk, step, sigma          AR chunk index, denoising step index, noise level
    q            float32 [H, nqb, d]  block means of the current chunk's queries
    k            float32 [H, nkb, d]  block means of all keys (cached chunks + current)
    frames_cached, current_frame_start, blocks_per_frame, frames_per_chunk
    budget_tiles, budget_used   rollout-level tile budget and what is spent so far
    probe(tiles) -> float32 [H, nqb, nkb]  exact log-sum-exp of the logits of the
                 requested tiles (NaN elsewhere); each probed tile costs
                 ``meta["probe_cost"]`` tiles of budget.

Attention logits are ``q . k / sqrt(d)``; key block ``b`` belongs to frame
``b // blocks_per_frame``; within a frame, blocks follow the row-major grid of
``grid // block_side`` x ``grid // block_side`` spatial tiles. Tokens inside a
block are a ``block_side x block_side`` spatial patch.

Wire format (both directions): b"SSP1", uint32 little-endian header length,
UTF-8 JSON header with an ``arrays`` list of {name, dtype, shape}, then the raw
C-ordered bytes of each array in that order.
"""

from __future__ import annotations

import json
import os
import struct
import sys
from typing import Any, BinaryIO

import numpy as np

MAGIC = b"SSP1"


def _read_exact(stream: BinaryIO, n: int) -> bytes | None:
    chunks = []
    remaining = n
    while remaining > 0:
        data = stream.read(remaining)
        if not data:
            return None
        chunks.append(data)
        remaining -= len(data)
    return b"".join(chunks)


def read_message(stream: BinaryIO) -> tuple[dict, dict[str, np.ndarray]] | None:
    head = _read_exact(stream, 8)
    if head is None:
        return None
    if head[:4] != MAGIC:
        raise ValueError("bad protocol magic")
    (length,) = struct.unpack("<I", head[4:])
    header = json.loads(_read_exact(stream, length).decode("utf-8"))
    arrays: dict[str, np.ndarray] = {}
    for spec in header.get("arrays", []):
        dtype = np.dtype(spec["dtype"])
        shape = tuple(int(s) for s in spec["shape"])
        nbytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        raw = _read_exact(stream, nbytes) if nbytes else b""
        arrays[spec["name"]] = np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
    return header, arrays


def write_message(stream: BinaryIO, header: dict, arrays: dict[str, np.ndarray] | None = None) -> None:
    arrays = arrays or {}
    header = dict(header)
    specs = []
    payloads = []
    for name, arr in arrays.items():
        arr = np.ascontiguousarray(arr)
        specs.append({"name": name, "dtype": arr.dtype.str.lstrip("<|="), "shape": list(arr.shape)})
        payloads.append(arr.tobytes())
    header["arrays"] = specs
    body = json.dumps(header).encode("utf-8")
    stream.write(MAGIC + struct.pack("<I", len(body)) + body)
    for payload in payloads:
        stream.write(payload)
    stream.flush()


class SelectCall:
    """One attention call (see module docstring)."""

    def __init__(self, header: dict, arrays: dict[str, np.ndarray], meta: dict,
                 inp: BinaryIO, out: BinaryIO) -> None:
        self.meta = meta
        self.chunk = int(header["chunk"])
        self.step = int(header["step"])
        self.sigma = float(header["sigma"])
        self.frames_cached = int(header["frames_cached"])
        self.current_frame_start = int(header["current_frame_start"])
        self.budget_tiles = float(header["budget_tiles"])
        self.budget_used = float(header["budget_used"])
        self.blocks_per_frame = int(meta["blocks_per_frame"])
        self.frames_per_chunk = int(meta["frames_per_chunk"])
        self.q = arrays["q"]
        self.k = arrays["k"]
        self.heads, self.nqb, self.dim = self.q.shape
        self.nkb = self.k.shape[1]
        self._inp = inp
        self._out = out

    def probe(self, tiles: np.ndarray) -> np.ndarray:
        tiles = np.asarray(tiles).astype(np.uint8)
        write_message(self._out, {"type": "probe"}, {"tiles": tiles})
        msg = read_message(self._inp)
        if msg is None:
            raise EOFError("evaluator closed the channel during a probe")
        return msg[1]["lse"]


def serve(selector: Any) -> None:
    """Run the request loop until the evaluator shuts the selector down."""

    inp = sys.stdin.buffer
    out = os.fdopen(os.dup(sys.stdout.fileno()), "wb")
    # Anything the selector prints goes to stderr, never into the protocol stream.
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    meta: dict = {}
    while True:
        msg = read_message(inp)
        if msg is None:
            return
        header, arrays = msg
        kind = header.get("type")
        if kind == "begin":
            meta = header["meta"]
            if hasattr(selector, "begin_rollout"):
                selector.begin_rollout(meta)
            write_message(out, {"type": "ack"})
        elif kind == "select":
            call = SelectCall(header, arrays, meta, inp, out)
            mask = np.asarray(selector.select(call))
            write_message(out, {"type": "mask"}, {"mask": mask.astype(np.uint8)})
        elif kind == "end":
            if hasattr(selector, "end_rollout"):
                selector.end_rollout()
            write_message(out, {"type": "ack"})
        elif kind == "shutdown":
            return
        else:
            raise ValueError(f"unknown request type: {kind}")
