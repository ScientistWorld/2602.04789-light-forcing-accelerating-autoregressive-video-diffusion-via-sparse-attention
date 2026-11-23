# Evaluation Targets

## Paper-Reported Reference Targets

`scoring/reference.json` keeps the paper's VBench tables unchanged:

- Table 1: Self-Forcing and LongLive main comparison;
- Table 2: component ablation;
- Table 3: top-k sensitivity.

They document the original claim: preserve or improve video quality at about 90% attention sparsity with a 1.2–1.3× speedup. The runnable gym does not regenerate 1.3B videos or run VBench. It scores the same question on a closed-loop toy AR attention world (see `briefing/evaluation.md`).

All runnable targets are measured by `bash scripts/evaluate_test.sh` (hidden test; train analogue `bash scripts/evaluate_train.sh`). They are written to `scoring/scores_test.json`.

## Primary: ar_rollout_core

Generation quality at the paper's operating point (target sparsity 0.9).

- **Metric**: `mean_generation_rel_error` (lower is better; primary)
- **Metric**: `final_chunk_rel_error` (lower is better)
- **Metric**: `revisit_rel_error` (lower is better)
- **Metric**: `mean_attention_cosine` (higher is better)

## Constraint: tile budget (all ar_rollout_* experiments)

Quality must be reached within the evaluator-fixed tile budget:

- the dense first chunk, plus (1 − target sparsity) of every later chunk's tiles;
- probes cost 0.5 tile each.

Exceeding the budget in any rollout, or sending an invalid mask, fails the run. A failed run receives the worst value of every metric in every experiment.

- **Metric**: `tile_cost_fraction` (reported; enforced as a hard constraint, not scored)

## Baselines: ar_rollout_core / long_context / high_sparsity

At the same budget, these baselines anchor the scale:

- chunk-agnostic 1D block-sparse attention (`one_d_sparse`);
- a static spatio-temporal local window (`local_window`);
- uniformly random tiles (`random_tiles`).

Dense attention is the zero-error reference, but it violates the budget.

## Robustness: ar_rollout_long_context

The same metrics with 10 chunks: longer history and more long-range revisits.

## Headroom: ar_rollout_high_sparsity

The same metrics at target sparsity 0.95, where the paper's method degrades most.

## Ablations (reference runs)

- `light_forcing_uniform`: the chunk-adaptive budget is removed.
- `light_forcing_cag_only`: the frame-level retrieval stage is removed.

Both are scored on all three experiments with the metrics above.
