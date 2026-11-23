# Experiment Overview

All runnable experiments generate videos autoregressively, chunk by chunk, in a closed loop. Each generated chunk becomes context for the next ones. The tile budget is fixed by the evaluator. Every experiment asks the same question: at that budget, how close does the generated video stay to the one dense attention would produce from the same noise?

## ar_rollout_core

The main setting: 7 chunks (21 frames, about a 5-second clip in the paper's models) at 90% target attention sparsity. It tests:

- whether a sparse attention policy preserves generation quality at the paper's operating point;
- whether errors accumulate over later chunks;
- whether places the camera returns to are rendered consistently.

## ar_rollout_long_context

The same question with 10 chunks (30 frames). The history is longer and there are more returns to places filmed long ago. It tests whether a policy stays useful, and avoids forgetting, as the cached history grows.

## ar_rollout_high_sparsity

The core setting at 95% target sparsity, half the attended tiles of the main operating point. It tests how gracefully quality degrades when the budget is much tighter. It leaves room for policies that use a small budget better.

## paper_* (context only)

These are the paper's reported VBench tables (main comparison on two models, component ablation, top-k sensitivity). They are kept for reference. The gym cannot regenerate them, and no method is scored on them.
