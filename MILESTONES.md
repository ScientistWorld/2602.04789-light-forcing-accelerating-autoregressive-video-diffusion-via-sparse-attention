# Reproduction Milestones

**Current: method_runs**

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

## Stop Justification
- Not stopped.
