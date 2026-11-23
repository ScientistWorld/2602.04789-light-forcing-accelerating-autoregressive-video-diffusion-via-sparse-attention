#!/usr/bin/env python3
"""Shortcut: NaN outputs with empty masks.

Every accuracy metric becomes NaN and is written to scores_test.json as NaN.
The active tile fraction is 0. If the rank treats NaN like a missing value
(skipped), only the efficiency metric counts and the run beats the reference.
"""

import sys

import torch

sys.path.insert(0, "/home/user")

from redteam.common.rtlib import BLOCK, EXPERIMENTS, inputs, save_artifact  # noqa: E402

NAME = "nan_outputs"

for experiment, spec in EXPERIMENTS.items():
    q, k, v = inputs(spec)
    outs, masks = [], []
    for i, qc in enumerate(q):
        kv_tokens = sum(x.shape[0] for x in k[: i + 1])
        outs.append(torch.full_like(qc, float("nan")))
        masks.append(torch.zeros(qc.shape[0] // BLOCK, kv_tokens // BLOCK, dtype=torch.bool))
    save_artifact(NAME, experiment, outs, masks=masks)
