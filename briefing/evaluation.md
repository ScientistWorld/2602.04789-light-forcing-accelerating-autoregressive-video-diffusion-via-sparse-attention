# Evaluation

## Metrics

The reusable benchmark scores generated attention artifacts against an independently computed dense-attention reference. `mean_relative_l2_error` is lower-is-better output error, `mean_cosine_similarity` is higher-is-better output agreement, and `active_block_fraction` is the lower-is-better fraction of active blockwise attention tiles.

## Baselines and Targets

The dense reference has zero output error and uses all attention blocks. The paper's full video benchmark reports that its proposed sparse method preserves or improves VBench total score while reaching about 90% attention sparsity and 1.2-1.3x end-to-end speedup on 1.3B autoregressive video models.

## Evaluation Protocol

Methods must write an artifact containing per-chunk attention outputs and active-block statistics. `scripts/evaluate.sh` reads that artifact and writes `scoring/scores.json`; it does not call or import the submitted method.
