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
  (given by the protocol), frame-wise keys = mean of a frame's block keys (Eq. 7);
  each query block retrieves its top-k past frames (k = 6) by <q~_r, k^_frame>
  (Eq. 8-9) and always keeps the current chunk's frames; it then keeps the
  blocks with the largest <q~_r, k~_j> inside the selected frames (Eq. 10-12).
  The number of blocks per query block is fixed by the chunk's sparsity,
  ``(1 - s_i) * (#key blocks)``, and the blocks are chosen within the frames
  picked by the frame stage (App. A). ``--selection hsa_per_frame`` instead
  splits that number evenly over the selected frames (a literal per-frame
  reading of Eq. 11); it is kept as a documented alternative.

The remaining options are ablations and variants used to probe the benchmark.
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from data.selector_protocol import serve


def cag_sparsities(chunks: int, s_target: float, s_base: float, alpha_power: float = 0.5) -> list[float]:
    """Eq. 5-6 with a dense first chunk; chunk cost is proportional to its KV length."""

    costs = [float(i + 1) for i in range(chunks)]
    alphas = [float(i + 1) ** (-alpha_power) for i in range(chunks)]
    adjustable = range(1, chunks)
    num = sum((s_base - s_target) * costs[i] for i in adjustable)
    den = sum(alphas[i] * costs[i] for i in adjustable)
    beta = num / max(den, 1e-12)
    return [0.0] + [min(0.99, max(0.0, s_base - alphas[i] * beta)) for i in adjustable]


def tiles_per_query_block(chunk: int, sparsity: float, nkb: int) -> int:
    if chunk == 0:
        return nkb
    return max(1, int(math.floor((1.0 - sparsity) * nkb + 1e-9)))


def frame_candidates(q: np.ndarray, k: np.ndarray, scores: np.ndarray, *, chunk: int,
                     frames_per_chunk: int, blocks_per_frame: int, topk_frames: int,
                     frame_score: str) -> np.ndarray:
    """Frame stage: bool [H, nqb, frames] = top-k past frames + current-chunk frames."""

    heads, nqb, dim = q.shape
    bpf = blocks_per_frame
    frames = k.shape[1] // bpf
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
    return frame_sel


def hsa_select(
    q: np.ndarray,
    k: np.ndarray,
    *,
    chunk: int,
    n_tiles: int,
    frames_per_chunk: int,
    blocks_per_frame: int,
    topk_frames: int = 6,
    per_frame: bool = False,
    frame_score: str = "mean",
) -> np.ndarray:
    """Two-stage frame -> block selection; returns bool [H, nqb, nkb]."""

    q = q.astype(np.float64)
    k = k.astype(np.float64)
    heads, nqb, _ = q.shape
    nkb = k.shape[1]
    bpf = blocks_per_frame
    frames = nkb // bpf
    scores = np.einsum("hqd,hkd->hqk", q, k)
    frame_sel = frame_candidates(q, k, scores, chunk=chunk, frames_per_chunk=frames_per_chunk,
                                 blocks_per_frame=bpf, topk_frames=topk_frames, frame_score=frame_score)
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


def add_random_tiles(mask: np.ndarray, extra: int, rng: np.random.Generator) -> np.ndarray:
    """Variant: add ``extra`` uniformly random unselected tiles to every row."""

    if extra <= 0:
        return mask
    keys = np.where(mask, -1.0, rng.random(mask.shape))
    top = np.argsort(-keys, axis=-1, kind="stable")[..., :extra]
    out = mask.copy()
    np.put_along_axis(out, top, True, axis=-1)
    return out


class LightForcingSelector:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.sparsities: list[float] = []
        self.meta: dict = {}
        weights = [float(w) for w in args.step_weights.split(",")] if args.step_weights else []
        self.step_weights = [w * len(weights) / sum(weights) for w in weights] if weights else []

    def begin_rollout(self, meta: dict) -> None:
        self.meta = meta
        s_target = min(0.99, float(meta["target_sparsity"]) + self.args.extra_sparsity)
        chunks = int(meta["chunks"])
        if self.args.allocation == "cag":
            self.sparsities = cag_sparsities(chunks, s_target, self.args.s_base, self.args.alpha_power)
        else:
            self.sparsities = [0.0] + [s_target] * (chunks - 1)

    def select(self, call) -> np.ndarray:
        if call.chunk == 0:
            return np.ones((call.heads, call.nqb, call.nkb), dtype=bool)
        n_tiles = tiles_per_query_block(call.chunk, self.sparsities[call.chunk], call.nkb)
        if self.step_weights:
            n_tiles = max(1, int(math.floor(self.step_weights[call.step] * n_tiles + 1e-9)))
        extra = int(math.floor(self.args.explore * n_tiles))
        n_main = n_tiles - extra
        if self.args.selection == "global":
            mask = global_select(call.q, call.k, n_main)
        else:
            mask = hsa_select(
                call.q, call.k, chunk=call.chunk, n_tiles=n_main,
                frames_per_chunk=call.frames_per_chunk, blocks_per_frame=call.blocks_per_frame,
                topk_frames=self.args.topk_frames, per_frame=self.args.selection == "hsa_per_frame",
                frame_score=self.args.frame_score,
            )
        if extra:
            rng = np.random.default_rng([int(self.meta.get("rollout_index", 0)), call.chunk, call.step, 7])
            mask = add_random_tiles(mask, extra, rng)
        return mask


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allocation", choices=["cag", "uniform"], default="cag")
    parser.add_argument("--selection", choices=["hsa", "hsa_per_frame", "global"], default="hsa",
                        help="hsa: best blocks within the retrieved frames (App. A); "
                             "hsa_per_frame: equal per-frame split (literal Eq. 11); "
                             "global: 1D top blocks over the whole cache (no frame stage)")
    parser.add_argument("--frame-score", choices=["mean", "max"], default="mean")
    parser.add_argument("--topk-frames", type=int, default=6)
    parser.add_argument("--s-base", type=float, default=0.98)
    parser.add_argument("--alpha-power", type=float, default=0.5,
                        help="alpha_i = i^-p; the paper's 1/sqrt(t) scaling is p = 0.5")
    parser.add_argument("--extra-sparsity", type=float, default=0.0,
                        help="variant: shift the operating point (positive = fewer tiles)")
    parser.add_argument("--step-weights", default="",
                        help="variant: comma-separated per-denoising-step tile multipliers")
    parser.add_argument("--explore", type=float, default=0.0,
                        help="variant: fraction of each row's tiles drawn at random")
    return parser.parse_args(argv)


def main() -> None:
    serve(LightForcingSelector(parse_args()))


if __name__ == "__main__":
    main()
