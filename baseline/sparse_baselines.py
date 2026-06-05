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
    parser.add_argument("--tokens-per-frame", type=int, default=128)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--window-frames", type=int, default=6)
    parser.add_argument("--seed", type=int, default=260204789)
    args = parser.parse_args()
    artifact = sliding_window_artifact(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
