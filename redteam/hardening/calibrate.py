#!/usr/bin/env python3
"""Archived calibration script (job 3ff53c2d-419), kept as evidence.

It was written against the calibration revision of method/light_forcing_selector.py,
where the default selection was the per-frame split (now `--selection hsa_per_frame`)
and `--selection hsa_global` was the Appendix-A reading (now the default `hsa`). So in
calibration.log, `light_forcing` = per-frame reading and `lf_hsa_global` = today's reference.

Calibration of the closed-loop world (in-process selectors, train-side code).

Prints, per world preset and experiment, every method's metrics and its paired
difference to the reference (light_forcing) over instances.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/user")

from baseline.sparse_baselines import Baseline  # noqa: E402
from eval.train import world as W  # noqa: E402
from method.light_forcing_selector import (  # noqa: E402
    LightForcingSelector,
    cag_sparsities,
    parse_args,
    tiles_per_query_block,
)

OUT = Path("/home/user/tmp/calibration")
OUT.mkdir(parents=True, exist_ok=True)


class CallView:
    def __init__(self, call, meta):
        self.meta = meta
        self.chunk, self.step, self.sigma = call.chunk, call.step, call.sigma
        self.q = call.q_pooled.numpy().astype(np.float32)
        self.k = call.k_pooled.numpy().astype(np.float32)
        self.heads, self.nqb, self.dim = self.q.shape
        self.nkb = self.k.shape[1]
        self.frames_per_chunk = W.FRAMES_PER_CHUNK
        self.blocks_per_frame = W.BLOCKS_PER_FRAME
        self.current_frame_start = call.current_frame_start
        self.frames_cached = call.frames_cached
        self.budget_tiles = meta["budget_tiles"]
        self.budget_used = call._rollout.cost
        self._call = call

    def probe(self, tiles):
        return self._call.probe(tiles)


class Oracle:
    """Exact tile mass ranking (free, calibration only) at the CAG tile counts."""

    def __init__(self, allocation="cag"):
        self.allocation = allocation

    def begin_rollout(self, meta):
        s = float(meta["target_sparsity"])
        c = int(meta["chunks"])
        self.sp = cag_sparsities(c, s, 0.98) if self.allocation == "cag" else [0.0] + [s] * (c - 1)

    def select(self, view):
        n = tiles_per_query_block(view.chunk, self.sp[view.chunk], view.nkb)
        if view.chunk == 0:
            return np.ones((view.heads, view.nqb, view.nkb), bool)
        lse = view._call.tile_lse().numpy()
        top = np.argsort(-lse, axis=-1, kind="stable")[..., :n]
        m = np.zeros(lse.shape, bool)
        np.put_along_axis(m, top, True, axis=-1)
        return m


def make_methods():
    lf = lambda *a: LightForcingSelector(parse_args(list(a)))  # noqa: E731
    return {
        "light_forcing": lambda: lf(),
        "lf_hsa_global": lambda: lf("--selection", "hsa_global"),
        "lf_uniform": lambda: lf("--allocation", "uniform"),
        "cag_one_d": lambda: lf("--selection", "global"),
        "one_d_sparse": lambda: Baseline("one_d_sparse", 0),
        "local_window": lambda: Baseline("local_window", 0),
        "random_tiles": lambda: Baseline("random_tiles", 0),
        "lf_frame_max": lambda: lf("--frame-score", "max"),
        "lf_adaptive": lambda: lf("--adaptive"),
        "lf_topk12": lambda: lf("--topk-frames", "12"),
        "lf_sparser": lambda: lf("--extra-sparsity", "0.05"),
        "oracle_cag": lambda: Oracle("cag"),
    }


def run_setting(tag, preset, chunks, target, seeds, methods):
    budget, dense_tiles = W.budget_tiles(chunks, target)
    worlds = []
    for seed in seeds:
        worlds.append(W.World(seed=seed, chunks=chunks, preset=preset))
    rv = [sum(int(w.revisit_tokens(i).sum()) for i in range(w.chunks)) for w in worlds]
    print(f"\n### {tag}: chunks={chunks} target={target} budget_frac={budget/dense_tiles:.4f} "
          f"revisit_tokens={rv}", flush=True)
    t0 = time.time()
    dense = [W.rollout(w, None) for w in worlds]
    print(f"dense rollouts: {(time.time()-t0)/len(worlds):.1f}s each", flush=True)
    results = {}
    for name, factory in methods.items():
        per = []
        t0 = time.time()
        failed = None
        for idx, (w, d) in enumerate(zip(worlds, dense)):
            sel = factory()
            meta = {"experiment": tag, "rollout_index": idx, "chunks": chunks, "target_sparsity": target,
                    "budget_tiles": budget, "dense_tiles": dense_tiles, "frames_per_chunk": 3,
                    "blocks_per_frame": 16, "grid": 16, "block_side": 4, "heads": 4, "probe_cost": 0.5}
            sel.begin_rollout(meta)
            try:
                run = W.rollout(w, lambda call: sel.select(CallView(call, meta)), budget=budget)
            except W.BudgetExceeded as exc:
                failed = str(exc)
                break
            per.append(W.rollout_metrics(w, run, d, dense_tiles))
        dt = (time.time() - t0) / max(1, len(per))
        if failed:
            print(f"{name:16s} FAILED {failed}", flush=True)
            continue
        results[name] = per
        agg = {m: float(np.nanmean([p[m] for p in per])) for m in
               ("mean_generation_rel_error", "final_chunk_rel_error", "revisit_rel_error",
                "mean_attention_cosine", "tile_cost_fraction")}
        line = (f"{name:16s} gen={agg['mean_generation_rel_error']:.4f} fin={agg['final_chunk_rel_error']:.4f} "
                f"rev={agg['revisit_rel_error']:.4f} cos={agg['mean_attention_cosine']:.4f} "
                f"cost={agg['tile_cost_fraction']:.4f}")
        if "light_forcing" in results and name != "light_forcing":
            ref = results["light_forcing"]
            for m, short in (("mean_generation_rel_error", "dgen"), ("revisit_rel_error", "drev"),
                             ("mean_attention_cosine", "dcos")):
                d_ = np.array([p[m] - r[m] for p, r in zip(per, ref)])
                line += f" {short}={d_.mean():+.4f}±{d_.std():.4f}"
        line += f" [{dt:.1f}s/rollout]"
        print(line, flush=True)
        prof = np.mean([p["chunk_errors"] for p in per], axis=0)
        print(f"{'':16s} chunk errors: " + " ".join(f"{e:.3f}" for e in prof), flush=True)
    dump = {n: [{k: v for k, v in p.items()} for p in per] for n, per in results.items()}
    (OUT / f"{tag}.json").write_text(json.dumps(dump, indent=1))
    return results


def main():
    torch.set_num_threads(1)
    methods = make_methods()
    which = sys.argv[1:] or ["A_core", "A_long", "A_hs", "B_core", "C_core"]
    presets = {
        "A": dict(W.DEFAULT_PRESET),
        "B": dict(W.DEFAULT_PRESET, sharpness=1.4, revisits_per_frame=0.10),
        "C": dict(W.DEFAULT_PRESET, sharpness=0.7, speed=(0.8, 1.6)),
    }
    for item in which:
        p, kind = item.split("_")
        if kind == "core":
            run_setting(item, presets[p], 7, 0.90, [101, 102, 103], methods)
        elif kind == "long":
            run_setting(item, presets[p], 10, 0.90, [201, 202], methods)
        elif kind == "hs":
            run_setting(item, presets[p], 7, 0.95, [301, 302, 303], methods)


if __name__ == "__main__":
    main()
