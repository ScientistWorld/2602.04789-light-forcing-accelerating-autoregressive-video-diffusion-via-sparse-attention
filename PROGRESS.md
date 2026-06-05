# Progress

## What Works

- Paper analysis is complete for the core method, main VBench results, ablations, and implementation hyperparameters.
- `method/light_forcing_attention.py` implements Chunk-Aware Growth sparsity allocation and Hierarchical Sparse Attention mask selection/attention.
- `eval/attention_benchmark.py` evaluates method artifacts without importing `method/`.
- `scripts/reproduce.sh` runs a sparse baseline, the Light Forcing method, and validation.

## Results

- No compute-job results yet. The first submitted job will write `scoring/scores.json`.

## Remaining

- Inspect the smoke-job results.
- If the Light Forcing artifact scores better than the baseline under active-block constraints, update milestone to `core_claim`; otherwise diagnose and resubmit.
- Optionally add top-k and CAG ablation artifacts after the core run succeeds.

## Issues

- Full 1.3B video generation and VBench evaluation are not feasible as a first pass in this workspace because the paper's upstream code and model weights are not released in a directly runnable form here.

## Deviations from Paper

- Paper setup: Self Forcing and LongLive 1.3B autoregressive video diffusion models evaluated with VBench on generated videos.
- Reproduction setup: deterministic scaled AR-style attention tensors evaluated against dense attention outputs.
- Reason: this preserves the paper's actual CAG + HSA sparse attention computation while avoiding unavailable full video-model weights and expensive VBench generation in the initial reproduction.
