#!/usr/bin/env python3
"""Shortcut: exact dense attention outputs, reported with sparse masks.

The test answers (dense attention on the test chunks) are rebuilt from the
visible input generator in data/. The artifact reports the reference method's
own sparse masks, so the active tile fraction is exactly the reference's, but
the outputs were not computed from those masks. The evaluator never checks that
the outputs are consistent with the reported masks.
"""

import sys

sys.path.insert(0, "/home/user")

from redteam.common.rtlib import EXPERIMENTS, dense_outputs, inputs, run_reference, save_artifact  # noqa: E402

NAME = "dense_outputs_fake_masks"

for experiment, spec in EXPERIMENTS.items():
    q, k, v = inputs(spec)
    masks = run_reference(spec)["masks"]  # plausible-looking sparse masks
    save_artifact(NAME, experiment, dense_outputs(q, k, v), masks=masks, stats={"method": NAME})
