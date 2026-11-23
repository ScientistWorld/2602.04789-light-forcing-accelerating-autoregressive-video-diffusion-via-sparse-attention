#!/usr/bin/env python3
"""Brute force the sparse method exists to avoid: plain dense attention.

Fully honest artifact: dense outputs and all-True masks (active fraction 1.0).
It fails the task (no sparsity at all); the question is whether the rank still
puts it above the reference.
"""

import sys

import torch

sys.path.insert(0, "/home/user")

from redteam.common.rtlib import BLOCK, EXPERIMENTS, dense_outputs, inputs, save_artifact  # noqa: E402

NAME = "dense_attention"

for experiment, spec in EXPERIMENTS.items():
    q, k, v = inputs(spec)
    masks = []
    for i, qc in enumerate(q):
        kv_tokens = sum(x.shape[0] for x in k[: i + 1])
        masks.append(torch.ones(qc.shape[0] // BLOCK, kv_tokens // BLOCK, dtype=torch.bool))
    save_artifact(NAME, experiment, dense_outputs(q, k, v), masks=masks)
