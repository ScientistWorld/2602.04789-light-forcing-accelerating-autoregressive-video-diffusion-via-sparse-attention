#!/usr/bin/env python3
"""Shortcut attempt: degenerate outputs with zero active tiles.

  degenerate_mean_value : every query gets the plain average of all cached
                          values (no query-key scores at all)
  degenerate_zero       : all-zero outputs
  degenerate_copy_value : output = the current chunk's own values (copied input)

All report empty masks (0 active tiles).
"""

import sys

import torch

sys.path.insert(0, "/home/user")

from redteam.common.rtlib import BLOCK, EXPERIMENTS, history, inputs, save_artifact  # noqa: E402

for experiment, spec in EXPERIMENTS.items():
    q, k, v = inputs(spec)
    mean_out, zero_out, copy_out, masks = [], [], [], []
    for i, qc in enumerate(q):
        keys, values = history(k, v, i)
        mean_out.append(values.mean(dim=0, keepdim=True).expand_as(qc).clone())
        zero_out.append(torch.zeros_like(qc))
        copy_out.append(v[i].clone())
        masks.append(torch.zeros(qc.shape[0] // BLOCK, keys.shape[0] // BLOCK, dtype=torch.bool))
    save_artifact("degenerate_mean_value", experiment, mean_out, masks=masks)
    save_artifact("degenerate_zero", experiment, zero_out, masks=masks)
    save_artifact("degenerate_copy_value", experiment, copy_out, masks=masks)
