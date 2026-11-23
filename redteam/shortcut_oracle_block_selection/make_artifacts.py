#!/usr/bin/env python3
"""Brute force: pick tiles by their exact dense attention mass.

Same dense first chunk and same per-chunk tile budget as the reference (its
chunk-aware allocation), so the reported active fraction equals the
reference's. The selection itself needs the full dense QK^T softmax, i.e. the
cost the sparse method exists to avoid; the evaluator does not account for
selection cost. Outputs are honestly computed from the reported masks.
"""

import sys

sys.path.insert(0, "/home/user")

from redteam.common.rtlib import (  # noqa: E402
    EXPERIMENTS,
    cag_sparsities,
    inputs,
    save_artifact,
    score_oracle,
    selection_attention,
)

NAME = "oracle_blocks"

for experiment, spec in EXPERIMENTS.items():
    q, k, v = inputs(spec)
    outs, masks = selection_attention(q, k, v, spec, cag_sparsities(spec), score_oracle, use_frames=False)
    save_artifact(NAME, experiment, outs, masks=masks)
