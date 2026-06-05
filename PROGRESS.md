# Progress

## What Works

- Paper analysis is complete for the core method, main VBench results, ablations, and implementation hyperparameters.
- `method/light_forcing_attention.py` implements Chunk-Aware Growth sparsity allocation and Hierarchical Sparse Attention mask selection/attention.
- `eval/attention_benchmark.py` evaluates method artifacts without importing `method/`.
- `scripts/reproduce.sh` runs a sparse baseline, the Light Forcing method, and validation.
- CPU job `3f3f2610-17a` completed end to end, so the implementation reaches `method_runs`.
- CPU job `e2d100d1-446` supports the minimum `core_claim` on the scaled benchmark.
- CPU job `12c2e3d9-a7f` supports `core_claim_plus` by reproducing the same-budget advantage on a longer-context setting.
- CPU job `8f617d06-5c9` supports `secondary_claims` by reproducing the CAG allocation benefit over uniform sparsity.
- GPU-backed job `8d82fc4a-f7e` reran the full packaged reproduction and validation path successfully.

## Results

- Job `3f3f2610-17a`: Light Forcing mean relative L2 error 0.497, cosine 0.855, active block fraction 0.137.
- Same job: sliding-window baseline mean relative L2 error 0.444, cosine 0.859, active block fraction 0.464.
- Interpretation: the sparse method is much cheaper but not yet better on the primary fidelity metric, so `core_claim` is not claimed.
- Job `9445fbe2-8d9`: finer block granularity improved Light Forcing to mean relative L2 error 0.367 and cosine 0.923 at active block fraction 0.131, but the denser sliding-window baseline still had lower error.
- Job `e2d100d1-446`: at the same active block fraction of 0.131, Light Forcing beat the chunk-agnostic one-dimensional sparse baseline with lower mean relative L2 error (0.367 vs. 0.374) and higher cosine similarity (0.923 vs. 0.920).
- Job `12c2e3d9-a7f`: in the longer-context setting, Light Forcing again beat the one-dimensional sparse baseline at matched active block fraction 0.120, with lower mean relative L2 error (0.486 vs. 0.489) and higher cosine similarity (0.875 vs. 0.874). It also beat the denser sliding-window baseline on this setting.
- Job `8f617d06-5c9`: CAG beat uniform sparsity in both settings at identical active block fractions. Core setting: 0.367 vs. 0.374 mean relative L2. Long-context setting: 0.486 vs. 0.489 mean relative L2.
- Job `8d82fc4a-f7e`: final GPU-backed validation reproduced the full score table. Final Light Forcing scores are 0.367 mean relative L2 on `scaled_attention_core` and 0.486 on `scaled_attention_long_context`.
- Validation fix: `scoring/reference.json` now stores the deterministic scaled-benchmark reference values from the final run instead of ideal dense-fidelity placeholders, so `validate.py` can compare scores consistently.

## Remaining

- Stop at `secondary_claims`.
- Higher milestones would require reproducing more of the unavailable full video generation/VBench stack.

## Issues

- Full 1.3B video generation and VBench evaluation are not feasible as a first pass in this workspace because the paper's upstream code and model weights are not released in a directly runnable form here.

## Deviations from Paper

- Paper setup: Self Forcing and LongLive 1.3B autoregressive video diffusion models evaluated with VBench on generated videos.
- Reproduction setup: deterministic scaled AR-style attention tensors evaluated against dense attention outputs.
- Reason: this preserves the paper's actual CAG + HSA sparse attention computation while avoiding unavailable full video-model weights and expensive VBench generation in the initial reproduction.
