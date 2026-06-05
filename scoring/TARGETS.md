# Evaluation Targets

## Primary: scaled_attention_core

Tests whether a sparse autoregressive attention method preserves the dense-attention output on a deterministic chunked sequence benchmark.

- **Metric**: `mean_relative_l2_error` (lower is better)
- **Metric**: `mean_cosine_similarity` (higher is better)
- **Run**: `bash scripts/evaluate.sh <method_name> <artifact.pt>`

## Constraint: scaled_attention_core

Tests whether output preservation is achieved with fewer active blockwise attention tiles rather than by falling back to dense attention.

- **Metric**: `active_block_fraction` (lower is better)
- **Metric**: `dense_equivalent_fraction` (lower is better)
- **Run**: `bash scripts/evaluate.sh <method_name> <artifact.pt>`

## Baseline: scaled_attention_core

Compares against a high-cost local-context baseline and a chunk-agnostic sparse-attention baseline so improvements cannot come only from using more tiles.

- **Metric**: `mean_relative_l2_error` (lower is better)
- **Metric**: `active_block_fraction` (lower is better)
- **Run**: `bash scripts/reproduce.sh`
