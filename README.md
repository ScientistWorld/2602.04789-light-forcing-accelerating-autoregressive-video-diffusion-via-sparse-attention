# Overview

- **Paper ID:** 2602.04789
- **Title:** Light Forcing: Accelerating Autoregressive Video Diffusion via Sparse Attention
- **Domain:** Video generation / efficient attention
- **TL;DR:** The paper makes sparse attention practical for autoregressive video diffusion by allocating sparsity across chunks and selecting historical context hierarchically.

## Short Summary

Autoregressive video diffusion can stream long videos, but attention over the growing history becomes the main runtime bottleneck. Light Forcing addresses this with chunk-aware sparsity allocation and coarse-to-fine sparse context retrieval. The paper reports VBench quality comparable to or better than dense attention with lower latency.

## Key Results

- Self Forcing 1.3B: VBench total score 84.5 with 1.30x end-to-end speedup.
- LongLive 1.3B: VBench total score 83.9 with 1.19x end-to-end speedup.
- Efficient deployment with FP8 and LightVAE reports 19.7 FPS on RTX 5090.

---

# Problem Definition

## Research Question

How can autoregressive video generators reduce the quadratic cost of growing historical attention without losing the visual consistency and motion quality that come from using long context?

## Why It Matters

Autoregressive video models generate chunks sequentially, so their key-value history grows with video length. Attention can dominate latency in later chunks, limiting interactive or real-time deployment.

## Success Criteria

A successful solution should preserve dense-attention outputs or video-quality metrics while using substantially fewer active attention blocks as the autoregressive context grows. It should also avoid early-chunk degradation and long-context forgetting.

---

# Paper's Method

## Key Contribution

Light Forcing is a sparse attention method tailored to autoregressive video diffusion. Its key idea is to assign different sparsity budgets to chunks and to retrieve historical context in a two-stage frame/block hierarchy.

## Approach

The method keeps the first chunk dense, then uses Chunk-Aware Growth to solve a chunk-specific sparsity allocation from the target sparsity and an error model that gives earlier chunks more attention budget. For each later chunk, Hierarchical Sparse Attention first mean-pools query/key blocks, retrieves top historical frames for each query block, selects the most relevant key blocks inside those frames, and computes masked blockwise attention.

## Main Claims

The primary claim is that this sparse attention design preserves or improves autoregressive video quality while reducing attention cost. Secondary claims are that each component matters, top-k frame retrieval is robust, and the method can combine with quantization and a faster VAE for real-time deployment.

---

# Evaluation

## Metrics

The reusable benchmark scores generated attention artifacts against an independently computed dense-attention reference. `mean_relative_l2_error` is lower-is-better output error, `mean_cosine_similarity` is higher-is-better output agreement, and `active_block_fraction` is the lower-is-better fraction of active blockwise attention tiles.

## Baselines and Targets

The dense reference has zero output error and uses all attention blocks. The paper's full video benchmark reports that its proposed sparse method preserves or improves VBench total score while reaching about 90% attention sparsity and 1.2-1.3x end-to-end speedup on 1.3B autoregressive video models.

## Evaluation Protocol

Methods must write an artifact containing per-chunk attention outputs and active-block statistics. `scripts/evaluate.sh` reads that artifact and writes `scoring/scores.json`; it does not call or import the submitted method.

---

# Reproduction Log

### Iteration 1: gpt-5.5
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 27m | **GPU**: 0.0h
- **Jobs**: 6 total (6 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-05 03:10] - none
- Read the paper and appendix details for Light Forcing.
- Implemented a scaled tensor benchmark for the actual CAG and HSA sparse attention computation.
- Prepared scripts and validation; submitted first CPU smoke job to run the method and evaluator end to end.

### [2026-06-05 03:20] - method_runs
- CPU job `3f3f2610-17a` completed successfully.
- Ran sliding-window baseline, Light Forcing CAG+HSA, evaluator, and `validate.py --compare` end to end.
- Light Forcing activated 13.7% of blocks, but its output error was slightly worse than the sliding-window baseline, so this is not yet `core_claim`.

### [2026-06-05 03:28] - method_runs
- CPU job `9445fbe2-8d9` completed successfully with finer block granularity.
- Light Forcing improved to 0.367 mean relative L2 error and 0.923 cosine at 13.1% active blocks.
- Added a same-budget chunk-agnostic sparse baseline for the next run to test the paper's direct sparse-attention ablation more faithfully.

### [2026-06-05 03:35] - core_claim
- CPU job `e2d100d1-446` completed successfully with the same-budget chunk-agnostic sparse baseline.
- At the same 13.1% active block fraction, Light Forcing scored lower mean relative L2 error than the one-dimensional sparse baseline: 0.367 vs. 0.374.
- Added a longer-context scaled benchmark setting for the next run to test whether this advantage persists beyond the minimum setting.

### [2026-06-05 03:43] - core_claim_plus
- CPU job `12c2e3d9-a7f` completed successfully with core and longer-context settings.
- In the longer-context setting, Light Forcing scored lower mean relative L2 error than the same-budget one-dimensional sparse baseline: 0.486 vs. 0.489 at the same 12.0% active block fraction.
- Added a uniform-sparsity ablation for the next run to test the chunk-aware allocation contribution.

### [2026-06-05 03:52] - secondary_claims
- CPU job `8f617d06-5c9` completed successfully with the uniform-sparsity allocation ablation.
- CAG beat uniform sparsity in the core setting: 0.367 vs. 0.374 mean relative L2 at the same 13.1% active block fraction.
- CAG also beat uniform sparsity in the longer-context setting: 0.486 vs. 0.489 mean relative L2 at the same 12.0% active block fraction.
- Preparing one final GPU-backed scheduler validation before stopping.

### [2026-06-05 03:59] - secondary_claims
- GPU-backed job `8d82fc4a-f7e` completed successfully.
- The packaged `scripts/reproduce.sh` path reran all baselines, Light Forcing, the uniform allocation ablation, and `validate.py --compare` without errors.
- Final scores preserve the core, longer-context, and CAG-ablation findings.

</details>


---

# Reproduction Milestones

**Current: secondary_claims**

## Stop Justification
- Completed at milestone `secondary_claims`.
