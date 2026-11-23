#!/usr/bin/env python3
"""Baseline mask selectors for the autoregressive sparse-attention benchmark.

All baselines keep the first chunk dense and give every later chunk the same
fraction ``1 - target_sparsity`` of its key blocks (chunk-agnostic budget), the
evaluator-fixed operating point.

* ``one_d_sparse``: dynamic 1D block-sparse attention in the style of
  SpargeAttn / the paper's "+1D Sparse Attention" row: per query block, the key
  blocks with the largest pooled logit over the whole KV cache.
* ``local_window``: static spatio-temporal neighbourhood (STA / sliding-window
  style): same and adjacent spatial tiles, most recent frames first.
* ``random_tiles``: uniformly random key blocks (sanity floor for retrieval).
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from data.selector_protocol import serve


def n_tiles(chunk: int, target: float, nkb: int) -> int:
    if chunk == 0:
        return nkb
    return max(1, int(math.floor((1.0 - target) * nkb + 1e-9)))


def top_rows(scores: np.ndarray, n: int) -> np.ndarray:
    n = min(n, scores.shape[-1])
    top = np.argsort(-scores, axis=-1, kind="stable")[..., :n]
    mask = np.zeros(scores.shape, dtype=bool)
    np.put_along_axis(mask, top, True, axis=-1)
    return mask


class Baseline:
    def __init__(self, kind: str, seed: int) -> None:
        self.kind = kind
        self.seed = seed
        self.meta: dict = {}

    def begin_rollout(self, meta: dict) -> None:
        self.meta = meta

    def select(self, call) -> np.ndarray:
        n = n_tiles(call.chunk, float(self.meta["target_sparsity"]), call.nkb)
        shape = (call.heads, call.nqb, call.nkb)
        if call.chunk == 0:
            return np.ones(shape, dtype=bool)
        if self.kind == "one_d_sparse":
            scores = np.einsum("hqd,hkd->hqk", call.q.astype(np.float64), call.k.astype(np.float64))
            return top_rows(scores, n)
        if self.kind == "random_tiles":
            rng = np.random.default_rng(
                [self.seed, int(self.meta["rollout_index"]), call.chunk, call.step]
            )
            return top_rows(rng.random(shape), n)
        if self.kind == "local_window":
            return top_rows(np.broadcast_to(self._window_scores(call), shape), n)
        raise ValueError(self.kind)

    def _window_scores(self, call) -> np.ndarray:
        bpf = call.blocks_per_frame
        side = int(round(math.sqrt(bpf)))
        r = np.arange(call.nqb)
        c = np.arange(call.nkb)
        tq = call.current_frame_start + r // bpf
        tk = c // bpf
        qi, qj = np.divmod(r % bpf, side)
        ki, kj = np.divmod(c % bpf, side)
        dt = np.abs(tq[:, None] - tk[None, :]).astype(np.float64)
        cheb = np.maximum(np.abs(qi[:, None] - ki[None, :]), np.abs(qj[:, None] - kj[None, :]))
        near = cheb <= 1
        return np.where(near, -(dt + 0.01 * cheb), -(1000.0 + 10.0 * dt + cheb))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", choices=["one_d_sparse", "local_window", "random_tiles"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    serve(Baseline(args.baseline, args.seed))


if __name__ == "__main__":
    main()
