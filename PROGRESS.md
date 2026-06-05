# Progress

## What Works

- Paper analysis is complete for the core method, main VBench results, ablations, and implementation hyperparameters.
- `method/light_forcing_attention.py` implements Chunk-Aware Growth sparsity allocation and Hierarchical Sparse Attention mask selection/attention.
- `eval/attention_benchmark.py` evaluates method artifacts without importing `method/`.
- `scripts/reproduce.sh` runs a sparse baseline, the Light Forcing method, and validation.
- CPU job `3f3f2610-17a` completed end to end, so the implementation reaches `method_runs`.
- CPU job `e2d100d1-446` supports the minimum `core_claim` on the scaled benchmark.

## Results

- Job `3f3f2610-17a`: Light Forcing mean relative L2 error 0.497, cosine 0.855, active block fraction 0.137.
- Same job: sliding-window baseline mean relative L2 error 0.444, cosine 0.859, active block fraction 0.464.
- Interpretation: the sparse method is much cheaper but not yet better on the primary fidelity metric, so `core_claim` is not claimed.
- Job `9445fbe2-8d9`: finer block granularity improved Light Forcing to mean relative L2 error 0.367 and cosine 0.923 at active block fraction 0.131, but the denser sliding-window baseline still had lower error.
- Job `e2d100d1-446`: at the same active block fraction of 0.131, Light Forcing beat the chunk-agnostic one-dimensional sparse baseline with lower mean relative L2 error (0.367 vs. 0.374) and higher cosine similarity (0.923 vs. 0.920).

## Remaining

- Run the added longer-context scaled benchmark setting to check whether the same-budget sparse-attention advantage holds beyond the minimum setting.
- Optionally add top-k and CAG ablation artifacts after the core run succeeds.

## Issues

- Full 1.3B video generation and VBench evaluation are not feasible as a first pass in this workspace because the paper's upstream code and model weights are not released in a directly runnable form here.

## Deviations from Paper

- Paper setup: Self Forcing and LongLive 1.3B autoregressive video diffusion models evaluated with VBench on generated videos.
- Reproduction setup: deterministic scaled AR-style attention tensors evaluated against dense attention outputs.
- Reason: this preserves the paper's actual CAG + HSA sparse attention computation while avoiding unavailable full video-model weights and expensive VBench generation in the initial reproduction.
