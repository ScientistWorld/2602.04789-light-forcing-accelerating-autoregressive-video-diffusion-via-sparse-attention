"""Deterministic inputs for sparse autoregressive attention benchmarks."""

from __future__ import annotations

import math

import torch


def make_structured_qkv(
    *,
    chunks: int,
    frames_per_chunk: int,
    tokens_per_frame: int,
    dim: int,
    seed: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """Create deterministic AR-video-like query, key, and value chunks."""

    generator = torch.Generator().manual_seed(seed)
    total_frames = chunks * frames_per_chunk
    frame_ids = torch.arange(total_frames, dtype=torch.float32)
    token_ids = torch.arange(tokens_per_frame, dtype=torch.float32)

    frame_phase = torch.stack(
        [
            torch.sin(frame_ids / 2.0),
            torch.cos(frame_ids / 3.0),
            torch.sin(frame_ids / 5.0),
            torch.cos(frame_ids / 7.0),
        ],
        dim=-1,
    )
    token_phase = torch.stack(
        [
            torch.sin(token_ids / 3.0),
            torch.cos(token_ids / 4.0),
            torch.sin(token_ids / 9.0),
            torch.cos(token_ids / 11.0),
        ],
        dim=-1,
    )
    basis = torch.randn(8, dim, generator=generator)

    frames: list[torch.Tensor] = []
    for frame in range(total_frames):
        pattern = torch.cat(
            [frame_phase[frame].repeat(tokens_per_frame, 1), token_phase],
            dim=-1,
        )
        recurrent = 0.35 * torch.sin((token_ids[:, None] + frame) / 6.0)
        noise = 0.04 * torch.randn(tokens_per_frame, dim, generator=generator)
        frames.append(pattern @ basis + recurrent + noise)

    q_chunks: list[torch.Tensor] = []
    k_chunks: list[torch.Tensor] = []
    v_chunks: list[torch.Tensor] = []
    k_proj = torch.randn(dim, dim, generator=generator) / math.sqrt(dim)
    v_proj = torch.randn(dim, dim, generator=generator) / math.sqrt(dim)
    for chunk in range(chunks):
        start = chunk * frames_per_chunk
        end = start + frames_per_chunk
        x = torch.cat(frames[start:end], dim=0)
        q_chunks.append(x + 0.03 * torch.randn(x.shape, generator=generator))
        k_chunks.append(x @ k_proj)
        v_chunks.append(torch.tanh(x @ v_proj))
    return q_chunks, k_chunks, v_chunks
