"""Evaluator side of the mask-selector protocol (hidden test copy).

Starts a registered selector command as a subprocess, sends it block-pooled
summaries for every attention call and validates every reply. The selector
never receives raw queries, keys, values, world parameters or reference
outputs, and nothing it sends back is trusted beyond a 0/1 tile mask of the
exact expected shape.
"""

from __future__ import annotations

import json
import os
import select
import signal
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

MAGIC = b"SSP1"
MAX_HEADER = 1 << 20
# Hang guard only: an honest selector answers in milliseconds.
CALL_TIMEOUT_S = 1800.0
MAX_PROBE_ROUNDS = 4


class SelectorError(RuntimeError):
    pass


def resolve_command(command: list) -> list[str]:
    if not isinstance(command, list) or not command or not all(isinstance(c, str) for c in command):
        raise SelectorError("registration 'command' must be a non-empty list of strings")
    if command[0] in ("python", "python3"):
        return [sys.executable, *command[1:]]
    return list(command)


class SelectorProcess:
    def __init__(self, command: list[str], root: Path, log_path: Path, meta: dict) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(log_path, "wb")
        env = dict(os.environ)
        env["PYTHONPATH"] = str(root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        try:
            self.proc = subprocess.Popen(
                command, cwd=str(root), env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=self._log, close_fds=True, start_new_session=True,
            )
        except OSError as exc:
            self._log.close()
            raise SelectorError(f"cannot start selector: {exc}") from exc
        self._in = self.proc.stdout.fileno()
        self._out = self.proc.stdin.fileno()
        self.meta = meta

    # ----------------------------------------------------------------- framing
    def _write(self, data: bytes, deadline: float) -> None:
        view = memoryview(data)
        while view:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise SelectorError("timed out writing to selector")
            _, writable, _ = select.select([], [self._out], [], remaining)
            if not writable:
                continue
            try:
                n = os.write(self._out, view[: 1 << 16])
            except (BrokenPipeError, OSError) as exc:
                raise SelectorError(f"selector closed its input: {exc}") from exc
            view = view[n:]

    def _read(self, n: int, deadline: float) -> bytes:
        chunks = []
        remaining = n
        while remaining > 0:
            left = deadline - time.monotonic()
            if left <= 0:
                raise SelectorError("timed out waiting for selector")
            readable, _, _ = select.select([self._in], [], [], left)
            if not readable:
                continue
            data = os.read(self._in, min(remaining, 1 << 20))
            if not data:
                raise SelectorError(f"selector exited (code {self.proc.poll()})")
            chunks.append(data)
            remaining -= len(data)
        return b"".join(chunks)

    def send(self, header: dict, arrays: dict[str, np.ndarray] | None = None) -> None:
        deadline = time.monotonic() + CALL_TIMEOUT_S
        specs, payloads = [], []
        for name, arr in (arrays or {}).items():
            arr = np.ascontiguousarray(arr)
            specs.append({"name": name, "dtype": arr.dtype.str.lstrip("<|="), "shape": list(arr.shape)})
            payloads.append(arr.tobytes())
        header = dict(header, arrays=specs)
        body = json.dumps(header).encode("utf-8")
        self._write(MAGIC + struct.pack("<I", len(body)) + body + b"".join(payloads), deadline)

    def receive(self, expected: dict[str, tuple[int, ...]]) -> tuple[dict, dict[str, np.ndarray]]:
        """Read one reply; arrays must be 0/1 uint8/bool with exactly the expected shapes."""

        deadline = time.monotonic() + CALL_TIMEOUT_S
        head = self._read(8, deadline)
        if head[:4] != MAGIC:
            raise SelectorError("malformed reply (bad magic)")
        (length,) = struct.unpack("<I", head[4:])
        if length > MAX_HEADER:
            raise SelectorError("reply header too large")
        try:
            header = json.loads(self._read(length, deadline).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SelectorError(f"malformed reply header: {exc}") from exc
        if not isinstance(header, dict) or not isinstance(header.get("arrays", []), list):
            raise SelectorError("malformed reply header")
        arrays: dict[str, np.ndarray] = {}
        for spec in header.get("arrays", []):
            if not isinstance(spec, dict):
                raise SelectorError("malformed array spec")
            name = spec.get("name")
            if name not in expected or name in arrays:
                raise SelectorError(f"unexpected array in reply: {name!r}")
            if spec.get("dtype") not in ("u1", "b1"):
                raise SelectorError(f"array {name} must be uint8 or bool, got {spec.get('dtype')!r}")
            shape = spec.get("shape")
            if not isinstance(shape, list) or tuple(shape) != tuple(expected[name]):
                raise SelectorError(f"array {name} has shape {shape}, expected {list(expected[name])}")
            nbytes = int(np.prod(expected[name], dtype=np.int64))
            raw = np.frombuffer(self._read(nbytes, deadline), dtype=np.uint8).reshape(expected[name])
            if raw.max(initial=0) > 1:
                raise SelectorError(f"array {name} must contain only 0/1 values")
            arrays[name] = raw.astype(bool)
        return header, arrays

    # ------------------------------------------------------------------ requests
    def begin(self, meta: dict) -> None:
        self.send({"type": "begin", "meta": meta})
        header, _ = self.receive({})
        if header.get("type") != "ack":
            raise SelectorError("selector did not acknowledge begin_rollout")

    def end(self) -> None:
        self.send({"type": "end"})
        header, _ = self.receive({})
        if header.get("type") != "ack":
            raise SelectorError("selector did not acknowledge end_rollout")

    def select(self, call, heads: int, budget_tiles: float, state) -> np.ndarray:
        shape = (heads, call.nqb, call.nkb)
        self.send(
            {
                "type": "select",
                "chunk": call.chunk,
                "step": call.step,
                "sigma": call.sigma,
                "frames_cached": call.frames_cached,
                "current_frame_start": call.current_frame_start,
                "budget_tiles": budget_tiles,
                "budget_used": state.cost,
            },
            {
                "q": call.q_pooled.numpy().astype(np.float32),
                "k": call.k_pooled.numpy().astype(np.float32),
            },
        )
        for _ in range(MAX_PROBE_ROUNDS + 1):
            header, arrays = self.receive({"mask": shape, "tiles": shape})
            kind = header.get("type")
            if kind == "mask" and set(arrays) == {"mask"}:
                return arrays["mask"]
            if kind == "probe" and set(arrays) == {"tiles"}:
                lse = call.probe(arrays["tiles"])
                self.send({"type": "probe_result"}, {"lse": lse})
                continue
            raise SelectorError(f"invalid reply type {kind!r} with arrays {sorted(arrays)}")
        raise SelectorError("too many probe rounds in one call")

    def close(self) -> None:
        try:
            if self.proc.poll() is None:
                try:
                    self.send({"type": "shutdown"})
                except SelectorError:
                    pass
                try:
                    self.proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
        finally:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            for stream in (self.proc.stdin, self.proc.stdout):
                try:
                    stream.close()
                except Exception:
                    pass
            self._log.close()
