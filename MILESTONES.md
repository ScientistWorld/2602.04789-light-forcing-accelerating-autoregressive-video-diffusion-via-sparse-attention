# Reproduction Milestones

**Current: secondary_claims**

<!-- Milestone levels (update "Current" above as you progress):
  none             — just started, no meaningful progress yet
  method_runs      — the paper's method executes end-to-end without errors
  core_claim       — minimum experiment supports the paper's central claim
  core_claim_plus  — core claim reproduced on additional settings
  secondary_claims — secondary results or contributions reproduced
  majority         — more than half of reported results reproduced
  near_complete    — most results reproduced, only minor gaps remain
  full             — all reported results reproduced
-->

## Progress Log

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

## Stop Justification
- Scaled reproduction is complete through the core claim, an additional context setting, and a CAG ablation. Full video-generation/VBench reproduction remains out of scope because the paper's directly runnable 1.3B video model weights are not available in this workspace.
