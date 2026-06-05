#!/usr/bin/env python3
"""Simple sparse attention baselines for the Light Forcing benchmark."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from data.attention_inputs import make_structured_qkv


def block_mask_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor, block: int) -> torch.Tensor:
    token_mask = mask.repeat_interleave(block, 0).repeat_interleave(block, 1)
    token_mask = token_mask[: q.shape[0], : k.shape[0]]
    scores = q @ k.T / math.sqrt(q.shape[-1])
    scores = scores.masked_fill(~token_mask, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v


def _block_mean(x: torch.Tensor, block_size: int) -> torch.Tensor:
    blocks = math.ceil(x.shape[0] / block_size)
    padded = torch.zeros(blocks * block_size, x.shape[1], dtype=x.dtype, device=x.device)
    padded[: x.shape[0]] = x
    return padded.view(blocks, block_size, x.shape[1]).mean(dim=1)


def one_d_sparse_artifact(args: argparse.Namespace) -> dict:
    """Chunk-agnostic 1D sparse attention baseline."""

    q_chunks, k_chunks, v_chunks = make_structured_qkv(
        chunks=args.chunks,
        frames_per_chunk=args.frames_per_chunk,
        tokens_per_frame=args.tokens_per_frame,
        dim=args.dim,
        seed=args.seed,
    )
    outputs = []
    active_blocks = 0
    total_blocks = 0
    past_k: list[torch.Tensor] = []
    past_v: list[torch.Tensor] = []
    q_blocks_per_chunk = math.ceil(args.frames_per_chunk * args.tokens_per_frame / args.block_size)
    for chunk, (q, k, v) in enumerate(zip(q_chunks, k_chunks, v_chunks)):
        past_k.append(k)
        past_v.append(v)
        keys = torch.cat(past_k, dim=0)
        values = torch.cat(past_v, dim=0)
        kv_blocks = math.ceil(keys.shape[0] / args.block_size)
        if chunk == 0 and args.dense_first_chunk:
            mask = torch.ones(q_blocks_per_chunk, kv_blocks, dtype=torch.bool)
        else:
            q_summary = _block_mean(q, args.block_size)
            k_summary = _block_mean(keys, args.block_size)
            logits = q_summary @ k_summary.T
            selected = max(1, int(round((1.0 - args.sparsity) * kv_blocks)))
            top_idx = torch.topk(logits, k=min(selected, kv_blocks), dim=1).indices
            mask = torch.zeros(q_blocks_per_chunk, kv_blocks, dtype=torch.bool)
            mask.scatter_(1, top_idx, True)
        outputs.append(block_mask_attention(q, keys, values, mask, args.block_size))
        active_blocks += int(mask.sum().item())
        total_blocks += int(mask.numel())
    return {
        "outputs": outputs,
        "stats": {
            "active_blocks": active_blocks,
            "total_blocks": total_blocks,
            "baseline": "one_d_sparse",
            "sparsity": args.sparsity,
            "dense_first_chunk": args.dense_first_chunk,
        },
    }


def sliding_window_artifact(args: argparse.Namespace) -> dict:
    q_chunks, k_chunks, v_chunks = make_structured_qkv(
        chunks=args.chunks,
        frames_per_chunk=args.frames_per_chunk,
        tokens_per_frame=args.tokens_per_frame,
        dim=args.dim,
        seed=args.seed,
    )
    outputs = []
    active_blocks = 0
    total_blocks = 0
    past_k: list[torch.Tensor] = []
    past_v: list[torch.Tensor] = []
    blocks_per_frame = math.ceil(args.tokens_per_frame / args.block_size)
    q_blocks_per_chunk = math.ceil(args.frames_per_chunk * args.tokens_per_frame / args.block_size)
    for chunk, (q, k, v) in enumerate(zip(q_chunks, k_chunks, v_chunks)):
        past_k.append(k)
        past_v.append(v)
        keys = torch.cat(past_k, dim=0)
        values = torch.cat(past_v, dim=0)
        kv_blocks = math.ceil(keys.shape[0] / args.block_size)
        mask = torch.zeros(q_blocks_per_chunk, kv_blocks, dtype=torch.bool)
        if chunk == 0:
            mask[:] = True
        else:
            last_frames = min(args.window_frames, (chunk + 1) * args.frames_per_chunk)
            first_frame = (chunk + 1) * args.frames_per_chunk - last_frames
            first_block = first_frame * blocks_per_frame
            mask[:, first_block:] = True
        outputs.append(block_mask_attention(q, keys, values, mask, args.block_size))
        active_blocks += int(mask.sum().item())
        total_blocks += int(mask.numel())
    return {
        "outputs": outputs,
        "stats": {
            "active_blocks": active_blocks,
            "total_blocks": total_blocks,
            "baseline": "sliding_window",
            "window_frames": args.window_frames,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/sliding_window.pt"))
    parser.add_argument("--chunks", type=int, default=7)
    parser.add_argument("--frames-per-chunk", type=int, default=3)
    parser.add_argument("--tokens-per-frame", type=int, default=256)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--window-frames", type=int, default=6)
    parser.add_argument("--baseline", choices=["sliding_window", "one_d_sparse"], default="sliding_window")
    parser.add_argument("--sparsity", type=float, default=0.90)
    parser.add_argument("--dense-first-chunk", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=260204789)
    args = parser.parse_args()
    if args.baseline == "sliding_window":
        artifact = sliding_window_artifact(args)
    else:
        artifact = one_d_sparse_artifact(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
