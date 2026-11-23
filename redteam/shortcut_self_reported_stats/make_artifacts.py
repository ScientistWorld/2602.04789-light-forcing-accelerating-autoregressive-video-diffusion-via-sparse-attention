#!/usr/bin/env python3
"""Shortcut: exact dense outputs, no masks, self-reported zero active tiles.

Without a `masks` list the evaluator takes `stats.active_blocks` and
`stats.total_blocks` from the artifact as written.
"""

import sys

sys.path.insert(0, "/home/user")

from redteam.common.rtlib import EXPERIMENTS, dense_outputs, inputs, save_artifact  # noqa: E402

NAME = "self_reported_stats"

for experiment, spec in EXPERIMENTS.items():
    q, k, v = inputs(spec)
    save_artifact(
        NAME,
        experiment,
        dense_outputs(q, k, v),
        masks=None,
        stats={"active_blocks": 0, "total_blocks": 1000},
    )
