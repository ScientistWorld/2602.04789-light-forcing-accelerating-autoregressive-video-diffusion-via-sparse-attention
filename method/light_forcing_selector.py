#!/usr/bin/env python3
"""Light Forcing mask selector: Chunk-Aware Growth + Hierarchical Sparse Attention.

Implements the paper's computational steps on the benchmark's selector protocol:

* Chunk-Aware Growth (Sec. 4.1, Eq. 5-6): the first chunk stays dense (App. A);
  chunk i (1-indexed) gets sparsity ``s_i = s_base - alpha_i * beta`` with
  ``alpha_i = 1/sqrt(i)`` (noise level reached after i chunks of T steps, up to
  a constant), and ``beta`` solved so that the FLOPs of chunks 2..n equal those
  of a uniform ``s_target``. ``s_target`` is the operating point the evaluator
  fixes; ``s_base = 0.98`` as in the paper.
* Hierarchical Sparse Attention (Sec. 4.2, Eq. 7-13): block-mean queries and keys
  (given by the protocol), frame-wise keys = mean of a frame's block keys; each
  query block retrieves its top-k past frames (k = 6) by <q~_r, k^_frame> and
  always keeps the current chunk's frames (Eq. 9); inside every selected frame
  it keeps the top blocks by <q~_r, k~_j> (Eq. 10-12). The number of blocks per
  query block is ``(1 - s_i) * (#key blocks)`` (App. A), split evenly over the
  selected frames; leftover slots go to the best remaining blocks of those frames.

Options exist only for ablations and variants (allocation, selection rule,
frame score, extra sparsity).
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from data.selector_protocol import serve


def cag_sparsities(chunks: int, s_target: float, s_base: float) -> list[float]:
    """Eq. 5-6 with a dense first chunk; chunk cost is proportional to its KV length."""

    costs = [float(i + 1) for i in range(chunks)]
    alphas = [1.0 / math.sqrt(i + 1) for i in range(chunks)]
    adjustable = range(1, chunks)
    num = sum((s_base - s_target) * costs[i] for i in adjustable)
    den = sum(alphas[i] * costs[i] for i in adjustable)
    beta = num / max(den, 1e-12)
    return [0.0] + [min(0.99, max(0.0, s_base - alphas[i] * beta)) for i in adjustable]


def tiles_per_query_block(chunk: int, sparsity: float, nkb: int) -> int:
    if chunk == 0:
        return nkb
    return max(1, int(math.floor((1.0 - sparsity) * nkb + 1e-9)))


def hsa_select(
    q: np.ndarray,
    k: np.ndarray,
    *,
    chunk: int,
    n_tiles: int,
    frames_per_chunk: int,
    blocks_per_frame: int,
    topk_frames: int = 6,
    per_frame: bool = True,
    frame_score: str = "mean",
) -> np.ndarray:
    """Two-stage frame -> block selection; returns bool [H, nqb, nkb]."""

    q = q.astype(np.float64)
    k = k.astype(np.float64)
    heads, nqb, dim = q.shape
    nkb = k.shape[1]
    bpf = blocks_per_frame
    frames = nkb // bpf
    scores = np.einsum("hqd,hkd->hqk", q, k)
    cur0 = chunk * frames_per_chunk
    frame_sel = np.zeros((heads, nqb, frames), dtype=bool)
    frame_sel[:, :, cur0:] = True
    if cur0 > 0:
        if frame_score == "mean":
            k_frame = k[:, : cur0 * bpf].reshape(heads, cur0, bpf, dim).mean(axis=2)
            frame_logits = np.einsum("hqd,hfd->hqf", q, k_frame)
        elif frame_score == "max":
            frame_logits = scores[:, :, : cur0 * bpf].reshape(heads, nqb, cur0, bpf).max(axis=-1)
        else:
            raise ValueError(frame_score)
        kk = min(topk_frames, cur0)
        top = np.argsort(-frame_logits, axis=-1, kind="stable")[..., :kk]
        np.put_along_axis(frame_sel, top, True, axis=-1)
    n_frames = int(frame_sel[0, 0].sum())
    n_tiles = min(n_tiles, n_frames * bpf)
    s4 = scores.reshape(heads, nqb, frames, bpf)
    sel = np.zeros(s4.shape, dtype=bool)
    used = 0
    if per_frame:
        k_per = min(n_tiles // n_frames, bpf)
        if k_per > 0:
            order = np.argsort(-s4, axis=-1, kind="stable")
            ranks = np.empty_like(order)
            np.put_along_axis(ranks, order, np.broadcast_to(np.arange(bpf), order.shape), axis=-1)
            sel = frame_sel[..., None] & (ranks < k_per)
            used = k_per * n_frames
    rem = n_tiles - used
    if rem > 0:
        cand = np.where(frame_sel[..., None] & ~sel, s4, -np.inf).reshape(heads, nqb, nkb)
        top = np.argsort(-cand, axis=-1, kind="stable")[..., :rem]
        flat = sel.reshape(heads, nqb, nkb).copy()
        np.put_along_axis(flat, top, True, axis=-1)
        sel = flat.reshape(s4.shape)
    return sel.reshape(heads, nqb, nkb)


def global_select(q: np.ndarray, k: np.ndarray, n_tiles: int) -> np.ndarray:
    """1D block selection: top blocks by pooled logit over the whole KV cache."""

    scores = np.einsum("hqd,hkd->hqk", q.astype(np.float64), k.astype(np.float64))
    heads, nqb, nkb = scores.shape
    n_tiles = min(n_tiles, nkb)
    top = np.argsort(-scores, axis=-1, kind="stable")[..., :n_tiles]
    mask = np.zeros((heads, nqb, nkb), dtype=bool)
    np.put_along_axis(mask, top, True, axis=-1)
    return mask


def adaptive_rows(q: np.ndarray, k: np.ndarray, candidates: np.ndarray, total: int, floor_per_row: int) -> np.ndarray:
    """Variant: share the call's tile total across rows by estimated tile probability."""

    q = q.astype(np.float64)
    k = k.astype(np.float64)
    heads, nqb, dim = q.shape
    logits = np.einsum("hqd,hkd->hqk", q, k) / math.sqrt(dim)
    logits = np.where(candidates, logits, -np.inf)
    logits -= logits.max(axis=-1, keepdims=True)
    prob = np.exp(logits)
    prob /= prob.sum(axis=-1, keepdims=True)
    nkb = prob.shape[-1]
    floor_per_row = min(floor_per_row, nkb)
    order = np.argsort(-prob, axis=-1, kind="stable")
    mask = np.zeros(prob.shape, dtype=bool)
    np.put_along_axis(mask, order[..., :floor_per_row], True, axis=-1)
    rest = total - int(mask.sum())
    if rest > 0:
        flat = np.where(mask | ~candidates, -1.0, prob).ravel()
        rest = min(rest, int((flat >= 0).sum()))
        if rest > 0:
            idx = np.argpartition(-flat, rest - 1)[:rest]
            flat_mask = mask.reshape(-1)
            flat_mask[idx] = True
            mask = flat_mask.reshape(prob.shape)
    return mask


class LightForcingSelector:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.sparsities: list[float] = []
        self.meta: dict = {}

    def begin_rollout(self, meta: dict) -> None:
        self.meta = meta
        s_target = min(0.99, float(meta["target_sparsity"]) + self.args.extra_sparsity)
        chunks = int(meta["chunks"])
        if self.args.allocation == "cag":
            self.sparsities = cag_sparsities(chunks, s_target, self.args.s_base)
        else:
            self.sparsities = [0.0] + [s_target] * (chunks - 1)

    def select(self, call) -> np.ndarray:
        n_tiles = tiles_per_query_block(call.chunk, self.sparsities[call.chunk], call.nkb)
        if call.chunk == 0:
            return np.ones((call.heads, call.nqb, call.nkb), dtype=bool)
        if self.args.selection == "global":
            return global_select(call.q, call.k, n_tiles)
        mask = hsa_select(
            call.q, call.k, chunk=call.chunk, n_tiles=n_tiles,
            frames_per_chunk=call.frames_per_chunk, blocks_per_frame=call.blocks_per_frame,
            topk_frames=self.args.topk_frames, per_frame=self.args.selection == "hsa",
            frame_score=self.args.frame_score,
        )
        if self.args.adaptive:
            candidates = hsa_select(
                call.q, call.k, chunk=call.chunk, n_tiles=call.nkb,
                frames_per_chunk=call.frames_per_chunk, blocks_per_frame=call.blocks_per_frame,
                topk_frames=self.args.topk_frames, per_frame=False, frame_score=self.args.frame_score,
            )
            mask = adaptive_rows(call.q, call.k, candidates, int(mask.sum()), max(1, n_tiles // 3))
        return mask


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allocation", choices=["cag", "uniform"], default="cag")
    parser.add_argument("--selection", choices=["hsa", "hsa_global", "global"], default="hsa",
                        help="hsa: per-frame top-k in retrieved frames (paper Eq. 11-12); "
                             "hsa_global: best blocks anywhere in the retrieved frames; "
                             "global: 1D top blocks over the whole cache (no frame stage)")
    parser.add_argument("--frame-score", choices=["mean", "max"], default="mean")
    parser.add_argument("--topk-frames", type=int, default=6)
    parser.add_argument("--s-base", type=float, default=0.98)
    parser.add_argument("--extra-sparsity", type=float, default=0.0,
                        help="weaker variant: spend fewer tiles than the budget allows")
    parser.add_argument("--adaptive", action="store_true",
                        help="variant: share each call's tiles across rows by estimated mass")
    return parser.parse_args(argv)


def main() -> None:
    serve(LightForcingSelector(parse_args()))


if __name__ == "__main__":
    main()
