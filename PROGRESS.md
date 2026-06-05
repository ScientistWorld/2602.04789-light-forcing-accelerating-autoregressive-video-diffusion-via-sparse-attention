# Progress

## What Works

- Paper analysis is complete for the core method, main VBench results, ablations, and implementation hyperparameters.
- `method/light_forcing_attention.py` implements Chunk-Aware Growth sparsity allocation and Hierarchical Sparse Attention mask selection/attention.
- `eval/attention_benchmark.py` evaluates method artifacts without importing `method/`.
- `scripts/reproduce.sh` runs a sparse baseline, the Light Forcing method, and validation.
- CPU job `3f3f2610-17a` completed end to end, so the implementation reaches `method_runs`.

## Results

- Job `3f3f2610-17a`: Light Forcing mean relative L2 error 0.497, cosine 0.855, active block fraction 0.137.
- Same job: sliding-window baseline mean relative L2 error 0.444, cosine 0.859, active block fraction 0.464.
- Interpretation: the sparse method is much cheaper but not yet better on the primary fidelity metric, so `core_claim` is not claimed.

## Remaining

- Rerun after increasing scaled benchmark block granularity to better match the paper's many-block-per-frame regime.
- If the Light Forcing artifact scores better than the baseline under active-block constraints, update milestone to `core_claim`; otherwise diagnose and resubmit.
- Optionally add top-k and CAG ablation artifacts after the core run succeeds.

## Issues

- Full 1.3B video generation and VBench evaluation are not feasible as a first pass in this workspace because the paper's upstream code and model weights are not released in a directly runnable form here.

## Deviations from Paper

- Paper setup: Self Forcing and LongLive 1.3B autoregressive video diffusion models evaluated with VBench on generated videos.
- Reproduction setup: deterministic scaled AR-style attention tensors evaluated against dense attention outputs.
- Reason: this preserves the paper's actual CAG + HSA sparse attention computation while avoiding unavailable full video-model weights and expensive VBench generation in the initial reproduction.
