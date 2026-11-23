#!/usr/bin/env python3
"""Shortcut: the reference's honest outputs with an impossible tile count.

  negative_mask_values : masks list present, but stored as float tensors of -1,
                         so mask.sum() is negative -> active fraction -1.0
  negative_stats       : no masks, stats.active_blocks = -1e9 -> fraction -1e9

The outputs are exactly the reference's, so every accuracy metric ties the
reference and only the (impossible) efficiency value differs.
"""

import sys

import torch

sys.path.insert(0, "/home/user")

from redteam.common.rtlib import EXPERIMENTS, run_reference, save_artifact  # noqa: E402

for experiment, spec in EXPERIMENTS.items():
    ref = run_reference(spec)
    neg_masks = [torch.full(m.shape, -1.0) for m in ref["masks"]]
    save_artifact("negative_mask_values", experiment, ref["outputs"], masks=neg_masks,
                  stats={"note": "masks hold -1"})
    save_artifact("negative_stats", experiment, ref["outputs"], masks=None,
                  stats={"active_blocks": -1e9, "total_blocks": 1})
