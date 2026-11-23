#!/usr/bin/env python3
"""Shortcut selectors replayed through the hardened hidden test.

Each mode tries one way to score without solving the task better:

  dense          all tiles every call (brute force; ignores the budget)
  probe_all      probes every tile, then keeps the exact top-mass tiles (oracle
                 selection with the probe cost counted)
  negative       mask entries of -1 (the old "impossible active fraction")
  float_mask     raw protocol reply with a float32 mask of 0.5 values
  extra_outputs  reference mask plus an extra "outputs"/"stats" array
                 (self-reported outputs or statistics)
  crash_long     reference behaviour, but refuses the long-context experiment
                 (the old "drop an experiment" shortcut)
  frontload      spends the whole budget densely in the earliest chunks, then
                 sends empty masks (budget gaming within the rules)
  read_hidden    tries to read eval/test/ and the evaluator's memory to recover
                 the hidden instances; behaves like the reference otherwise
  write_scores   reference behaviour, but overwrites scoring/scores_test.json
                 with perfect numbers while it runs
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/user")

from data.selector_protocol import serve, write_message  # noqa: E402
from method.light_forcing_selector import LightForcingSelector, parse_args  # noqa: E402


def log(msg: str) -> None:
    print(f"[shortcut] {msg}", file=sys.stderr, flush=True)


class Shortcut(LightForcingSelector):
    def __init__(self, mode: str) -> None:
        super().__init__(parse_args([]))
        self.mode = mode

    def begin_rollout(self, meta: dict) -> None:
        super().begin_rollout(meta)
        if self.mode == "crash_long" and int(meta["chunks"]) > 7:
            raise RuntimeError("refusing the long-context experiment")
        if self.mode == "read_hidden" and int(meta["rollout_index"]) == 0:
            for target in ("eval/test/evaluate.py", "eval/test/world.py"):
                try:
                    Path(target).read_text()
                    log(f"READ {target}: hidden test code is readable")
                except OSError as exc:
                    log(f"blocked reading {target}: {exc!r}")
            try:
                os.listdir("eval/test")
                log("listed eval/test")
            except OSError as exc:
                log(f"blocked listing eval/test: {exc!r}")
            ppid = os.getppid()
            for target in (f"/proc/{ppid}/mem", f"/proc/{ppid}/environ", f"/proc/{ppid}/maps"):
                try:
                    with open(target, "rb") as fh:
                        fh.read(16)
                    log(f"READ {target}")
                except OSError as exc:
                    log(f"blocked reading {target}: {exc!r}")
        if self.mode == "write_scores":
            fake = {"experiments": {e: {"results": {"write_scores": {
                "mean_generation_rel_error": 0.0, "final_chunk_rel_error": 0.0, "revisit_rel_error": 0.0,
                "mean_attention_cosine": 1.0, "tile_cost_fraction": 0.0}}} for e in
                ("ar_rollout_core", "ar_rollout_long_context", "ar_rollout_high_sparsity")}}
            Path("scoring/scores_test.json").write_text(json.dumps(fake))
            log("wrote fake scoring/scores_test.json")

    def select(self, call):
        shape = (call.heads, call.nqb, call.nkb)
        if self.mode == "dense":
            return np.ones(shape, dtype=bool)
        if self.mode == "negative":
            return -np.ones(shape, dtype=np.int8)
        if self.mode == "probe_all" and call.chunk > 0:
            lse = call.probe(np.ones(shape, dtype=bool))
            n = int(self.reference_mask(call).sum(axis=-1).max())
            top = np.argsort(-lse, axis=-1)[..., :n]
            mask = np.zeros(shape, dtype=bool)
            np.put_along_axis(mask, top, True, axis=-1)
            return mask
        if self.mode == "float_mask":
            write_message(call._out, {"type": "mask"}, {"mask": np.full(shape, 0.5, dtype=np.float32)})
            raise SystemExit(0)
        if self.mode == "extra_outputs":
            mask = self.reference_mask(call)
            write_message(call._out, {"type": "mask"}, {
                "mask": mask.astype(np.uint8),
                "outputs": np.zeros((4, 4), dtype=np.uint8),
                "stats": np.zeros(1, dtype=np.uint8),
            })
            raise SystemExit(0)
        if self.mode == "frontload":
            remaining = call.budget_tiles - call.budget_used
            if remaining >= np.prod(shape):
                return np.ones(shape, dtype=bool)
            return np.zeros(shape, dtype=bool)
        return self.reference_mask(call)

    def reference_mask(self, call):
        return super().select(call)


if __name__ == "__main__":
    serve(Shortcut(sys.argv[1]))
