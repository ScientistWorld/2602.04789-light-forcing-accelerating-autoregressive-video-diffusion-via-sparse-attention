#!/usr/bin/env python3
"""Visible train-slice evaluator for sparse autoregressive attention artifacts.

This module is method-agnostic: it reads candidate `.pt` artifacts from
`outputs/`, compares stored chunk outputs against independently reconstructed
dense attention references, and writes `scoring/scores_train.json`.
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


SLICE_NAME = "train"
EXPERIMENT_SPECS: dict[str, dict[str, int | str]] = {
    "scaled_attention_core": {
        "artifact_suffix": "",
        "chunks": 7,
        "frames_per_chunk": 3,
        "tokens_per_frame": 256,
        "dim": 64,
        "seed": 260204789,
    },
    "scaled_attention_long_context": {
        "artifact_suffix": "_long_context",
        "chunks": 9,
        "frames_per_chunk": 3,
        "tokens_per_frame": 256,
        "dim": 64,
        "seed": 260204790,
    },
}


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


def train_chunk_indices(chunks: int) -> list[int]:
    """Balanced visible slice: every other chunk, including the first dense chunk."""

    return list(range(0, chunks, 2))


def benchmark_config(experiment: str, args: argparse.Namespace) -> dict[str, int]:
    if experiment not in EXPERIMENT_SPECS:
        raise KeyError(f"unknown experiment: {experiment}")
    spec = EXPERIMENT_SPECS[experiment]
    return {
        "chunks": args.chunks if args.chunks is not None else int(spec["chunks"]),
        "frames_per_chunk": (
            args.frames_per_chunk
            if args.frames_per_chunk is not None
            else int(spec["frames_per_chunk"])
        ),
        "tokens_per_frame": (
            args.tokens_per_frame
            if args.tokens_per_frame is not None
            else int(spec["tokens_per_frame"])
        ),
        "dim": args.dim if args.dim is not None else int(spec["dim"]),
        "seed": args.seed if args.seed is not None else int(spec["seed"]),
    }


def score_artifact(path: Path, experiment: str, args: argparse.Namespace) -> dict[str, float]:
    config = benchmark_config(experiment, args)
    q_chunks, k_chunks, v_chunks = make_structured_qkv(**config)
    dense_outputs = dense_chunk_outputs(q_chunks, k_chunks, v_chunks)
    artifact = load_outputs(path)
    outputs = artifact["outputs"]
    if len(outputs) != len(dense_outputs):
        raise ValueError(f"expected {len(dense_outputs)} chunk outputs, got {len(outputs)}")

    selected = train_chunk_indices(len(dense_outputs))
    rel_errors: list[float] = []
    cosine: list[float] = []
    for idx in selected:
        got = outputs[idx].detach().cpu().float()
        ref = dense_outputs[idx].detach().cpu().float()
        rel_errors.append((got - ref).norm().item() / max(ref.norm().item(), 1e-12))
        cosine.append(F.cosine_similarity(got.flatten(), ref.flatten(), dim=0).item())

    stats = artifact["stats"]
    if isinstance(artifact.get("masks"), list) and len(artifact["masks"]) == len(outputs):
        active = sum(float(artifact["masks"][idx].sum().item()) for idx in selected)
        total = sum(float(artifact["masks"][idx].numel()) for idx in selected)
    else:
        active = float(stats.get("active_blocks", 0.0))
        total = float(stats.get("total_blocks", 1.0))

    return {
        "mean_relative_l2_error": float(sum(rel_errors) / len(rel_errors)),
        "mean_cosine_similarity": float(sum(cosine) / len(cosine)),
        "active_block_fraction": active / max(total, 1.0),
        "dense_equivalent_fraction": 1.0,
    }


def discover_artifacts(outputs_dir: Path, experiment: str) -> dict[str, Path]:
    suffix = str(EXPERIMENT_SPECS[experiment]["artifact_suffix"])
    artifacts: dict[str, Path] = {}
    if suffix:
        pattern = f"*{suffix}.pt"
        for path in sorted(outputs_dir.glob(pattern)):
            method_name = path.stem[: -len(suffix)]
            if method_name:
                artifacts[method_name] = path
    else:
        heldout_suffixes = {
            str(spec["artifact_suffix"])
            for spec in EXPERIMENT_SPECS.values()
            if str(spec["artifact_suffix"])
        }
        for path in sorted(outputs_dir.glob("*.pt")):
            if any(path.stem.endswith(heldout_suffix) for heldout_suffix in heldout_suffixes):
                continue
            artifacts[path.stem] = path
    return artifacts


def write_scores(scores_path: Path, data: dict[str, Any]) -> None:
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    scores_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def score_single(args: argparse.Namespace) -> dict[str, Any]:
    if not args.method_name or not args.artifact or not args.experiment:
        raise ValueError("--method-name, --artifact, and --experiment are required together")
    if args.scores.exists():
        data = json.loads(args.scores.read_text())
    else:
        data = {"_meta": {"slice": SLICE_NAME}, "experiments": {}}
    data.setdefault("_meta", {})["slice"] = SLICE_NAME
    data.setdefault("experiments", {}).setdefault(args.experiment, {}).setdefault("results", {})
    exp_data = data["experiments"][args.experiment]
    exp_data["slice"] = SLICE_NAME
    exp_data["chunk_indices"] = train_chunk_indices(benchmark_config(args.experiment, args)["chunks"])
    exp_data["results"][args.method_name] = score_artifact(args.artifact, args.experiment, args)
    return data


def score_all(args: argparse.Namespace) -> dict[str, Any]:
    data: dict[str, Any] = {"_meta": {"slice": SLICE_NAME}, "experiments": {}}
    for experiment in EXPERIMENT_SPECS:
        artifacts = discover_artifacts(args.outputs_dir, experiment)
        if not artifacts:
            continue
        config = benchmark_config(experiment, args)
        exp_data = {
            "slice": SLICE_NAME,
            "chunk_indices": train_chunk_indices(config["chunks"]),
            "results": {},
        }
        for method_name, artifact in artifacts.items():
            exp_data["results"][method_name] = score_artifact(artifact, experiment, args)
        data["experiments"][experiment] = exp_data
    if not data["experiments"]:
        raise FileNotFoundError(f"no .pt artifacts found in {args.outputs_dir}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--method-name")
    parser.add_argument("--experiment")
    parser.add_argument("--scores", type=Path, default=Path("scoring/scores_train.json"))
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--chunks", type=int)
    parser.add_argument("--frames-per-chunk", type=int)
    parser.add_argument("--tokens-per-frame", type=int)
    parser.add_argument("--dim", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    data = score_single(args) if args.method_name or args.artifact or args.experiment else score_all(args)
    write_scores(args.scores, data)
    print(json.dumps(data, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
