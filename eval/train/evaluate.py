#!/usr/bin/env python3
"""Visible train-slice evaluator for sparse-attention mask selectors.

Every method registered as ``outputs/methods/<name>.json`` (``{"command": [...]}``)
is started as a subprocess and drives the closed-loop autoregressive rollouts of
``eval/train/world.py`` through the mask-selector protocol. The evaluator
computes every attention output, every rollout, the tile cost and every metric
itself; it never reads numbers, outputs or statistics written by a method.

A method fails (worst value on every metric of every experiment) if it crashes,
replies with anything but a 0/1 tile mask of the right shape, or exceeds the
experiment's tile budget in any rollout.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from eval.train.selector_host import SelectorError, SelectorProcess, resolve_command
from eval.train.world import (
    BLOCK_SIDE,
    BLOCK_TOKENS,
    BLOCKS_PER_FRAME,
    FRAMES_PER_CHUNK,
    GRID,
    HEAD_DIM,
    HEADS,
    PROBE_COST,
    SIGMAS,
    TOKENS_PER_FRAME,
    BudgetExceeded,
    World,
    budget_tiles,
    rollout,
    rollout_metrics,
)

SLICE_NAME = "train"
ROOT = Path(__file__).resolve().parents[2]
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$")
MIN_REVISIT_TOKENS = 64

EXPERIMENTS: dict[str, dict[str, Any]] = {
    "ar_rollout_core": {"chunks": 7, "target_sparsity": 0.90, "instances": 4, "seed_base": 11_000},
    "ar_rollout_long_context": {"chunks": 10, "target_sparsity": 0.90, "instances": 2, "seed_base": 12_000},
    "ar_rollout_high_sparsity": {"chunks": 7, "target_sparsity": 0.95, "instances": 4, "seed_base": 13_000},
}

METRICS = (
    "mean_generation_rel_error",
    "final_chunk_rel_error",
    "revisit_rel_error",
    "mean_attention_cosine",
    "tile_cost_fraction",
)
WORST = {
    "mean_generation_rel_error": 2.0,
    "final_chunk_rel_error": 2.0,
    "revisit_rel_error": 2.0,
    "mean_attention_cosine": -1.0,
    "tile_cost_fraction": 1.0,
}
BOUNDS = {
    "mean_generation_rel_error": (0.0, 2.0),
    "final_chunk_rel_error": (0.0, 2.0),
    "revisit_rel_error": (0.0, 2.0),
    "mean_attention_cosine": (-1.0, 1.0),
    "tile_cost_fraction": (0.0, 1.0),
}


def instance_worlds(spec: dict, limit: int | None) -> list[World]:
    count = spec["instances"] if limit is None else min(limit, spec["instances"])
    worlds: list[World] = []
    seed = spec["seed_base"]
    while len(worlds) < count:
        world = World(seed=seed, chunks=spec["chunks"])
        revisits = sum(int(world.revisit_tokens(i).sum()) for i in range(world.chunks))
        if revisits >= MIN_REVISIT_TOKENS:
            worlds.append(world)
        seed += 1
    return worlds


def rollout_meta(experiment: str, index: int, spec: dict) -> dict:
    budget, dense = budget_tiles(spec["chunks"], spec["target_sparsity"])
    return {
        "experiment": experiment,
        "rollout_index": index,
        "chunks": spec["chunks"],
        "frames_per_chunk": FRAMES_PER_CHUNK,
        "grid": GRID,
        "block_side": BLOCK_SIDE,
        "tokens_per_frame": TOKENS_PER_FRAME,
        "block_tokens": BLOCK_TOKENS,
        "blocks_per_frame": BLOCKS_PER_FRAME,
        "heads": HEADS,
        "head_dim": HEAD_DIM,
        "sigmas": list(SIGMAS),
        "target_sparsity": spec["target_sparsity"],
        "budget_tiles": budget,
        "dense_tiles": dense,
        "probe_cost": PROBE_COST,
    }


def discover(methods_dir: Path) -> tuple[list[tuple[str, Path]], list[str]]:
    methods, skipped = [], []
    if not methods_dir.is_dir():
        return methods, skipped
    for path in sorted(methods_dir.glob("*.json")):
        name = path.stem
        if name.startswith(("_", ".")):
            continue
        if not NAME_RE.match(name):
            skipped.append(name)
            continue
        methods.append((name, path))
    return methods, skipped


def run_method(name: str, registration: Path, worlds: dict[str, list[World]], log_dir: Path) -> tuple[dict, str | None]:
    """Run all rollouts of one method; returns ({exp: [run, ...]}, failure reason)."""

    runs: dict[str, list[dict]] = {exp: [] for exp in worlds}
    try:
        command = resolve_command(json.loads(registration.read_text()).get("command"))
    except (OSError, json.JSONDecodeError, AttributeError, SelectorError) as exc:
        return runs, f"invalid registration: {exc}"
    try:
        proc = SelectorProcess(command, ROOT, log_dir / f"{SLICE_NAME}_{name}.log", {})
    except SelectorError as exc:
        return runs, str(exc)
    failure = None
    try:
        for exp, exp_worlds in worlds.items():
            spec = EXPERIMENTS[exp]
            for idx, world in enumerate(exp_worlds):
                meta = rollout_meta(exp, idx, spec)
                proc.begin(meta)
                budget = meta["budget_tiles"]
                run = rollout(world, _bind_state(proc, budget), budget=budget)
                proc.end()
                runs[exp].append(run)
    except (BudgetExceeded, SelectorError) as exc:
        failure = f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # any other crash while serving the method counts as a failed run
        failure = f"{type(exc).__name__}: {exc}"
    finally:
        proc.close()
    return runs, failure


def _bind_state(proc: SelectorProcess, budget: float):
    def select(call):
        return proc.select(call, HEADS, budget, call._rollout)

    return select


def aggregate(per_instance: list[dict]) -> dict[str, float]:
    out: dict[str, float] = {}
    for metric in METRICS:
        vals = [m[metric] for m in per_instance if math.isfinite(m[metric])]
        out[metric] = float(np.mean(vals)) if vals else float("nan")
    return out


def valid(metrics: dict[str, float]) -> bool:
    for metric in METRICS:
        lo, hi = BOUNDS[metric]
        v = metrics.get(metric)
        if v is None or not math.isfinite(v) or v < lo - 1e-9 or v > hi + 1e-9:
            return False
    return True


def write_scores(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods-dir", type=Path, default=Path("outputs/methods"))
    parser.add_argument("--scores", type=Path, default=Path("scoring/scores_train.json"))
    parser.add_argument("--log-dir", type=Path, default=Path("outputs/selector_logs"))
    parser.add_argument("--only", action="append", default=None, help="score only these method names")
    parser.add_argument("--experiments", action="append", default=None)
    parser.add_argument("--max-instances", type=int, default=None, help="quick development runs only")
    args = parser.parse_args()

    torch.set_num_threads(1)
    exps = args.experiments or list(EXPERIMENTS)
    for exp in exps:
        if exp not in EXPERIMENTS:
            raise SystemExit(f"unknown experiment {exp}; choose from {sorted(EXPERIMENTS)}")
    worlds = {exp: instance_worlds(EXPERIMENTS[exp], args.max_instances) for exp in exps}
    methods, skipped = discover(args.methods_dir)
    if args.only:
        methods = [(n, p) for n, p in methods if n in set(args.only)]

    data: dict[str, Any] = {
        "_meta": {"slice": SLICE_NAME, "failures": {}, "skipped_registrations": skipped},
        "experiments": {},
    }
    for exp in exps:
        budget, dense = budget_tiles(EXPERIMENTS[exp]["chunks"], EXPERIMENTS[exp]["target_sparsity"])
        data["experiments"][exp] = {
            "slice": SLICE_NAME,
            "instances": len(worlds[exp]),
            "chunks": EXPERIMENTS[exp]["chunks"],
            "target_sparsity": EXPERIMENTS[exp]["target_sparsity"],
            "budget_tile_fraction": budget / dense,
            "results": {},
            "chunk_error_profiles": {},
            "per_instance": {},
        }
    dense_runs: dict[str, list[dict]] = {}
    for name, registration in methods:
        start = time.time()
        runs, failure = run_method(name, registration, worlds, args.log_dir)
        per_exp: dict[str, dict] = {}
        if failure is None:
            for exp in exps:
                if exp not in dense_runs:
                    dense_runs[exp] = [rollout(w, None) for w in worlds[exp]]
                _, dense_tiles = budget_tiles(EXPERIMENTS[exp]["chunks"], EXPERIMENTS[exp]["target_sparsity"])
                inst = [rollout_metrics(w, r, d, dense_tiles)
                        for w, r, d in zip(worlds[exp], runs[exp], dense_runs[exp])]
                metrics = aggregate(inst)
                if not valid(metrics):
                    failure = f"impossible metric values in {exp}: {metrics}"
                    break
                per_exp[exp] = (
                    metrics,
                    np.mean([m["chunk_errors"] for m in inst], axis=0).tolist(),
                    [{k: m[k] for k in METRICS} for m in inst],
                )
        for exp in exps:
            exp_data = data["experiments"][exp]
            if failure is None:
                metrics, profile, instances = per_exp[exp]
                exp_data["results"][name] = metrics
                exp_data["chunk_error_profiles"][name] = profile
                exp_data["per_instance"][name] = instances
            else:
                exp_data["results"][name] = dict(WORST)
        if failure is not None:
            data["_meta"]["failures"][name] = failure
        print(f"[{SLICE_NAME}] {name}: {'FAILED ' + failure if failure else 'ok'} ({time.time() - start:.0f}s)",
              flush=True)
        write_scores(args.scores, data)
    write_scores(args.scores, data)
    print(json.dumps({e: d["results"] for e, d in data["experiments"].items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    sys.exit(main())
