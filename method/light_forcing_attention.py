#!/usr/bin/env python3
"""Light Forcing CAG + HSA sparse attention primitives.

This module implements the paper's computational steps at tensor benchmark
scale: chunk-aware sparsity allocation, hierarchical frame/block retrieval,
and masked block attention for autoregressive chunks.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from data.attention_inputs import make_structured_qkv


@dataclass(frozen=True)
class LightForcingConfig:
    chunks: int = 7
    frames_per_chunk: int = 3
    tokens_per_frame: int = 256
    dim: int = 64
    block_size: int = 64
    topk_frames: int = 6
    s_target: float = 0.90
    s_base: float = 0.98
    allocation: str = "cag"
    seed: int = 260204789


def allocate_chunk_sparsities(
    *,
    chunks: int,
    frames_per_chunk: int,
    tokens_per_frame: int,
    block_size: int,
    s_target: float,
    s_base: float,
) -> list[float]:
    """Solve the CAG allocation from Eq. 5-6.

    The first chunk is dense, matching Appendix A. For later chunks, alpha_i is
    proportional to the finite-step sampling-error term 1/sqrt(t), where later
    chunks inherit more denoising history and therefore receive higher sparsity.
    """

    q_len = frames_per_chunk * tokens_per_frame
    dense_costs = []
    alphas = []
    for chunk in range(chunks):
        kv_len = (chunk + 1) * frames_per_chunk * tokens_per_frame
        dense_costs.append(q_len * kv_len)
        alphas.append(1.0 / math.sqrt(chunk + 1))

    adjustable = range(1, chunks)
    numerator = sum((s_base - s_target) * dense_costs[i] for i in adjustable)
    denominator = sum(alphas[i] * dense_costs[i] for i in adjustable)
    beta = numerator / max(denominator, 1e-12)

    sparsities = [0.0]
    for i in adjustable:
        sparsities.append(max(0.0, min(0.99, s_base - alphas[i] * beta)))
    return sparsities


def allocate_uniform_sparsities(*, chunks: int, s_target: float) -> list[float]:
    """Ablation for CAG: keep the paper's dense first chunk, then use one sparsity."""

    return [0.0] + [s_target for _ in range(1, chunks)]


def _block_mean(x: torch.Tensor, block_size: int) -> torch.Tensor:
    blocks = math.ceil(x.shape[0] / block_size)
    padded = torch.zeros(blocks * block_size, x.shape[1], dtype=x.dtype, device=x.device)
    padded[: x.shape[0]] = x
    return padded.view(blocks, block_size, x.shape[1]).mean(dim=1)


def _frame_block_means(k: torch.Tensor, tokens_per_frame: int, block_size: int) -> torch.Tensor:
    frames = k.shape[0] // tokens_per_frame
    blocks_per_frame = math.ceil(tokens_per_frame / block_size)
    out = []
    for frame in range(frames):
        start = frame * tokens_per_frame
        end = start + tokens_per_frame
        out.append(_block_mean(k[start:end], block_size))
    return torch.stack(out, dim=0).reshape(frames, blocks_per_frame, k.shape[-1])


def _expand_block_mask(mask: torch.Tensor, q_len: int, kv_len: int, block_size: int) -> torch.Tensor:
    token_mask = mask.repeat_interleave(block_size, 0).repeat_interleave(block_size, 1)
    return token_mask[:q_len, :kv_len]


def hierarchical_sparse_attention(
    q: torch.Tensor,
    k_all: torch.Tensor,
    v_all: torch.Tensor,
    *,
    chunk_index: int,
    sparsity: float,
    frames_per_chunk: int,
    tokens_per_frame: int,
    block_size: int,
    topk_frames: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute HSA for one AR chunk and return output plus block mask."""

    q_blocks = _block_mean(q, block_size)
    k_blocks_by_frame = _frame_block_means(k_all, tokens_per_frame, block_size)
    frames_seen, blocks_per_frame, dim = k_blocks_by_frame.shape
    kv_blocks = frames_seen * blocks_per_frame

    if chunk_index == 0:
        mask = torch.ones(q_blocks.shape[0], kv_blocks, dtype=torch.bool, device=q.device)
    else:
        mask = torch.zeros(q_blocks.shape[0], kv_blocks, dtype=torch.bool, device=q.device)
        current_start = chunk_index * frames_per_chunk
        current_frames = set(range(current_start, current_start + frames_per_chunk))
        past_frame_keys = k_blocks_by_frame[:current_start].mean(dim=1)
        selected_per_query = max(1, int(round((1.0 - sparsity) * kv_blocks)))

        for q_idx, q_summary in enumerate(q_blocks):
            logits = q_summary @ past_frame_keys.T
            top_count = min(topk_frames, past_frame_keys.shape[0])
            frame_ids = set(torch.topk(logits, k=top_count).indices.tolist()) if top_count else set()
            frame_ids.update(current_frames)

            candidates = []
            for frame_id in sorted(frame_ids):
                block_logits = k_blocks_by_frame[frame_id] @ q_summary
                for block_id in range(blocks_per_frame):
                    global_block = frame_id * blocks_per_frame + block_id
                    candidates.append((float(block_logits[block_id]), global_block))
            candidates.sort(reverse=True)
            for _, global_block in candidates[: min(selected_per_query, len(candidates))]:
                mask[q_idx, global_block] = True

    token_mask = _expand_block_mask(mask, q.shape[0], k_all.shape[0], block_size)
    scores = q @ k_all.T / math.sqrt(q.shape[-1])
    scores = scores.masked_fill(~token_mask, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v_all, mask


def run_light_forcing(config: LightForcingConfig) -> dict:
    q_chunks, k_chunks, v_chunks = make_structured_qkv(
        chunks=config.chunks,
        frames_per_chunk=config.frames_per_chunk,
        tokens_per_frame=config.tokens_per_frame,
        dim=config.dim,
        seed=config.seed,
    )
    if config.allocation == "cag":
        sparsities = allocate_chunk_sparsities(
            chunks=config.chunks,
            frames_per_chunk=config.frames_per_chunk,
            tokens_per_frame=config.tokens_per_frame,
            block_size=config.block_size,
            s_target=config.s_target,
            s_base=config.s_base,
        )
    elif config.allocation == "uniform":
        sparsities = allocate_uniform_sparsities(chunks=config.chunks, s_target=config.s_target)
    else:
        raise ValueError(f"unknown allocation: {config.allocation}")
    outputs = []
    masks = []
    active_blocks = 0
    total_blocks = 0
    past_k: list[torch.Tensor] = []
    past_v: list[torch.Tensor] = []
    for chunk, (q, k, v) in enumerate(zip(q_chunks, k_chunks, v_chunks)):
        past_k.append(k)
        past_v.append(v)
        k_all = torch.cat(past_k, dim=0)
        v_all = torch.cat(past_v, dim=0)
        out, mask = hierarchical_sparse_attention(
            q,
            k_all,
            v_all,
            chunk_index=chunk,
            sparsity=sparsities[chunk],
            frames_per_chunk=config.frames_per_chunk,
            tokens_per_frame=config.tokens_per_frame,
            block_size=config.block_size,
            topk_frames=config.topk_frames,
        )
        outputs.append(out)
        masks.append(mask.cpu())
        active_blocks += int(mask.sum().item())
        total_blocks += int(mask.numel())
    return {
        "outputs": outputs,
        "masks": masks,
        "stats": {
            "active_blocks": active_blocks,
            "total_blocks": total_blocks,
            "active_block_fraction": active_blocks / max(total_blocks, 1),
            "chunk_sparsities": sparsities,
            "config": asdict(config),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/light_forcing.pt"))
    parser.add_argument("--chunks", type=int, default=7)
    parser.add_argument("--frames-per-chunk", type=int, default=3)
    parser.add_argument("--tokens-per-frame", type=int, default=256)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--topk-frames", type=int, default=6)
    parser.add_argument("--s-target", type=float, default=0.90)
    parser.add_argument("--s-base", type=float, default=0.98)
    parser.add_argument("--allocation", choices=["cag", "uniform"], default="cag")
    parser.add_argument("--seed", type=int, default=260204789)
    args = parser.parse_args()

    config = LightForcingConfig(
        chunks=args.chunks,
        frames_per_chunk=args.frames_per_chunk,
        tokens_per_frame=args.tokens_per_frame,
        dim=args.dim,
        block_size=args.block_size,
        topk_frames=args.topk_frames,
        s_target=args.s_target,
        s_base=args.s_base,
        allocation=args.allocation,
        seed=args.seed,
    )
    artifact = run_light_forcing(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, args.output)
    print(json.dumps(artifact["stats"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
