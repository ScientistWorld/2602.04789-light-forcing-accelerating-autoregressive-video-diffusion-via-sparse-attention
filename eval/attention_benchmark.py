#!/usr/bin/env python3
"""Evaluation for sparse autoregressive attention artifacts.

The evaluator intentionally does not import from method/. It constructs the
fixed benchmark inputs and dense reference attention independently, then scores
whatever output artifact a candidate method writes.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from data.attention_inputs import make_structured_qkv


def dense_chunk_outputs(
    q_chunks: list[torch.Tensor],
    k_chunks: list[torch.Tensor],
    v_chunks: list[torch.Tensor],
) -> list[torch.Tensor]:
    outputs: list[torch.Tensor] = []
    past_k: list[torch.Tensor] = []
    past_v: list[torch.Tensor] = []
    for q, k, v in zip(q_chunks, k_chunks, v_chunks):
        past_k.append(k)
        past_v.append(v)
        keys = torch.cat(past_k, dim=0)
        values = torch.cat(past_v, dim=0)
        scores = q @ keys.T / math.sqrt(q.shape[-1])
        outputs.append(torch.softmax(scores, dim=-1) @ values)
    return outputs


def load_outputs(path: Path) -> dict[str, Any]:
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise TypeError(f"{path} must contain a dict artifact")
    if "outputs" not in obj or "stats" not in obj:
        raise KeyError(f"{path} must contain 'outputs' and 'stats'")
    return obj


def score_artifact(path: Path, args: argparse.Namespace) -> dict[str, float]:
    q_chunks, k_chunks, v_chunks = make_structured_qkv(
        chunks=args.chunks,
        frames_per_chunk=args.frames_per_chunk,
        tokens_per_frame=args.tokens_per_frame,
        dim=args.dim,
        seed=args.seed,
    )
    dense_outputs = dense_chunk_outputs(q_chunks, k_chunks, v_chunks)
    artifact = load_outputs(path)
    outputs = artifact["outputs"]
    if len(outputs) != len(dense_outputs):
        raise ValueError(f"expected {len(dense_outputs)} chunk outputs, got {len(outputs)}")

    rel_errors = []
    cosine = []
    for got, ref in zip(outputs, dense_outputs):
        got = got.detach().cpu().float()
        ref = ref.detach().cpu().float()
        rel_errors.append((got - ref).norm().item() / max(ref.norm().item(), 1e-12))
        cosine.append(F.cosine_similarity(got.flatten(), ref.flatten(), dim=0).item())

    stats = artifact["stats"]
    active = float(stats.get("active_blocks", 0.0))
    total = float(stats.get("total_blocks", 1.0))
    active_fraction = active / max(total, 1.0)
    return {
        "mean_relative_l2_error": float(sum(rel_errors) / len(rel_errors)),
        "mean_cosine_similarity": float(sum(cosine) / len(cosine)),
        "active_block_fraction": active_fraction,
        "dense_equivalent_fraction": 1.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--method-name", default="light_forcing")
    parser.add_argument("--experiment", default="scaled_attention_core")
    parser.add_argument("--scores", type=Path, default=Path("scoring/scores.json"))
    parser.add_argument("--chunks", type=int, default=7)
    parser.add_argument("--frames-per-chunk", type=int, default=3)
    parser.add_argument("--tokens-per-frame", type=int, default=128)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=260204789)
    args = parser.parse_args()

    metrics = score_artifact(args.artifact, args)
    args.scores.parent.mkdir(parents=True, exist_ok=True)
    if args.scores.exists():
        data = json.loads(args.scores.read_text())
    else:
        data = {"experiments": {}}
    data.setdefault("experiments", {}).setdefault(args.experiment, {}).setdefault("results", {})
    data["experiments"][args.experiment]["results"][args.method_name] = metrics
    args.scores.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
