#!/usr/bin/env python3
"""Honest better / weaker variants of the reference sparse attention method.

Every variant computes its outputs with blockwise sparse attention restricted to
the tiles in its reported mask, so the artifact is truthful.

  light_forcing             reference method, unchanged (method/)
  lf_reimpl_mean            re-implementation of the reference selection with the
                            paper's pooled block logits (sanity: must equal light_forcing)
  lf_better_lse_scoring     same allocation, same frame retrieval, same tile budget;
                            block score = log-sum-exp of pooled query vs each key token
                            in the block (1/64 of the dense QK^T cost)
  lf_better_sampled_rows    same allocation, frame retrieval and budget; block score =
                            softmax mass of every 8th query row (1/8 of dense QK^T cost)
  lf_weaker_random_blocks   same allocation and budget, tiles picked at random
                            (retrieval removed)
  lf_weaker_budget95        reference with target sparsity 0.95 instead of 0.90
  lf_ablation_uniform       reference with the chunk-aware allocation replaced by one
                            fixed sparsity after the dense first chunk
"""

from __future__ import annotations

import argparse
import sys

sys.path.insert(0, "/home/user")

from redteam.common.rtlib import (  # noqa: E402
    EXPERIMENTS,
    SCORERS,
    cag_sparsities,
    inputs,
    run_reference,
    save_artifact,
    selection_attention,
)

SELECT = {
    "lf_reimpl_mean": ("mean", True),
    "lf_better_lse_scoring": ("lse", True),
    "lf_better_sampled_rows": ("sampled", True),
    "lf_weaker_random_blocks": ("random", False),
}
REFERENCE = {
    "light_forcing": {},
    "lf_weaker_budget95": {"s_target": 0.95},
    "lf_ablation_uniform": {"allocation": "uniform"},
}


def build(name: str, experiment: str, seed: int | None = None):
    spec = EXPERIMENTS[experiment]
    if name in REFERENCE:
        art = run_reference(spec, seed=seed, **REFERENCE[name])
        return art["outputs"], art["masks"], art["stats"]
    scorer, use_frames = SELECT[name]
    q, k, v = inputs(spec, seed)
    outs, masks = selection_attention(
        q, k, v, spec, cag_sparsities(spec), SCORERS[scorer], use_frames=use_frames
    )
    return outs, masks, {"variant": name}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="+")
    args = parser.parse_args()
    for name in args.names:
        for experiment in EXPERIMENTS:
            outs, masks, stats = build(name, experiment)
            save_artifact(name, experiment, outs, masks=masks, stats=stats)


if __name__ == "__main__":
    main()
