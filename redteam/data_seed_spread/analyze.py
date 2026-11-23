#!/usr/bin/env python3
"""Spread of the test score across input draws (generator seeds).

The reference method is deterministic and the hidden test uses one fixed input
draw, so reruns have zero spread. To see how large a gap must be before it is
more than an artifact of that one draw, regenerate the benchmark inputs with
three generator seeds (the default one plus two others), run the reference, its
ablation, the better/weaker variants and the baselines on each draw, and score
them with the gym's own train/test slice scorers (explicit --seed).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, "/home/user")

import torch  # noqa: E402

from baseline.sparse_baselines import sliding_window_artifact  # noqa: E402
from eval.test import evaluate as test_eval  # noqa: E402
from eval.train import evaluate as train_eval  # noqa: E402
from redteam.common.rank import metric_spec, rank_delta  # noqa: E402
from redteam.common.rtlib import (  # noqa: E402
    EXPERIMENTS,
    cag_sparsities,
    inputs,
    score_oracle,
    selection_attention,
)
from redteam.variants.make_artifacts import build as build_variant  # noqa: E402

OFFSETS = (0, 101, 202)
EXP_NAMES = list(EXPERIMENTS)
VARIANTS = (
    "light_forcing",
    "lf_ablation_uniform",
    "lf_better_lse_scoring",
    "lf_better_sampled_rows",
    "lf_weaker_random_blocks",
    "lf_weaker_random_blocks_s1",
    "lf_weaker_random_blocks_s2",
    "lf_weaker_local_window",
    "lf_no_frame_retrieval",
    "lf_weaker_budget95",
)
WORK = Path("/home/user/tmp/redteam_work/seed_artifacts")
OUT = Path("/home/user/redteam/data_seed_spread")


def artifacts_for(experiment: str, seed: int) -> dict[str, dict]:
    spec = EXPERIMENTS[experiment]
    arts = {}
    for name in VARIANTS:
        outs, masks, stats = build_variant(name, experiment, seed=seed)
        arts[name] = {"outputs": outs, "masks": masks, "stats": stats}
    q, k, v = inputs(spec, seed)
    outs, masks = selection_attention(q, k, v, spec, cag_sparsities(spec), score_oracle, use_frames=False)
    arts["oracle_blocks"] = {"outputs": outs, "masks": masks, "stats": {}}
    sw_args = argparse.Namespace(
        chunks=spec["chunks"], frames_per_chunk=spec["frames_per_chunk"],
        tokens_per_frame=spec["tokens_per_frame"], dim=spec["dim"], seed=seed,
        block_size=64, window_frames=6,
    )
    arts["sliding_window"] = sliding_window_artifact(sw_args)
    return arts


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    spec = metric_spec()
    results: dict = {}
    for off in OFFSETS:
        results[str(off)] = {}
        for experiment, espec in EXPERIMENTS.items():
            seed = espec["seed"] + off
            ns = argparse.Namespace(chunks=None, frames_per_chunk=None, tokens_per_frame=None, dim=None, seed=seed)
            per_slice = {"test": {}, "train": {}}
            for name, art in artifacts_for(experiment, seed).items():
                path = WORK / f"{name}_{experiment}_{seed}.pt"
                torch.save(art, path)
                per_slice["test"][name] = test_eval.score_artifact(path, experiment, ns)
                per_slice["train"][name] = train_eval.score_artifact(path, experiment, ns)
                path.unlink()
            results[str(off)][experiment] = {"seed": seed, **per_slice}
            print(f"offset {off} {experiment} done", flush=True)

    # Spread of the reference's test metrics across draws.
    spread = {}
    for experiment in EXPERIMENTS:
        spread[experiment] = {}
        for m in results["0"][experiment]["test"]["light_forcing"]:
            vals = [results[str(o)][experiment]["test"]["light_forcing"][m] for o in OFFSETS]
            spread[experiment][m] = {
                "values": vals,
                "mean": statistics.fmean(vals),
                "std": statistics.pstdev(vals),
                "max_minus_min": max(vals) - min(vals),
            }

    # Rank delta of every method vs the reference on the same draw (raw coefs).
    deltas = {}
    for off in OFFSETS:
        deltas[str(off)] = {}
        for slice_name in ("test", "train"):
            ref = {e: results[str(off)][e][slice_name]["light_forcing"] for e in EXPERIMENTS}
            deltas[str(off)][slice_name] = {}
            for name in results[str(off)][EXP_NAMES[0]][slice_name]:
                cand = {e: results[str(off)][e][slice_name][name] for e in EXPERIMENTS}
                total, per_exp = rank_delta(cand, ref, spec)
                deltas[str(off)][slice_name][name] = {
                    "total": total,
                    **{e: per_exp[e]["delta"] for e in EXPERIMENTS},
                }
    # Spread of the reference's own rank total across draws, measured against
    # the default-draw reference (how much the reference "moves" by itself).
    ref0 = {e: results["0"][e]["test"]["light_forcing"] for e in EXPERIMENTS}
    ref_rank_vs_default = {
        str(off): rank_delta({e: results[str(off)][e]["test"]["light_forcing"] for e in EXPERIMENTS}, ref0, spec)[0]
        for off in OFFSETS
    }

    out = {
        "offsets": OFFSETS,
        "results": results,
        "reference_test_spread": spread,
        "rank_delta_vs_reference_same_draw": deltas,
        "reference_rank_total_vs_default_draw": ref_rank_vs_default,
    }
    (OUT / "results.json").write_text(json.dumps(out, indent=2) + "\n")

    lines = ["| method | " + " | ".join(f"test delta seed+{o}" for o in OFFSETS) + " | " +
             " | ".join(f"train delta seed+{o}" for o in OFFSETS) + " |",
             "|---|" + "---|" * (2 * len(OFFSETS))]
    for name in results["0"][EXP_NAMES[0]]["test"]:
        cells = [f"{deltas[str(o)]['test'][name]['total']:+.4f}" for o in OFFSETS]
        cells += [f"{deltas[str(o)]['train'][name]['total']:+.4f}" for o in OFFSETS]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Reference test metrics across draws (mean +- std, max-min):")
    for e in EXPERIMENTS:
        for m, s in spread[e].items():
            lines.append(f"- {e} {m}: {s['mean']:.4f} +- {s['std']:.4f} (range {s['max_minus_min']:.4f})")
    lines.append(f"- reference rank total vs default draw: {ref_rank_vs_default}")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
