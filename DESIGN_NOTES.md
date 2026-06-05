# Design Notes

## Checked

- Read `scripts/evaluate_train.sh` and `scripts/evaluate_test.sh` side by side. The train script calls `eval.train.evaluate` and writes `scoring/scores_train.json`; the test script calls `eval.test.evaluate` and writes `scoring/scores_test.json`.
- Checked the train/test evaluators for method attachment. Both read submitted `.pt` artifacts from `outputs/` or an explicit artifact path, then compare stored outputs against an independently reconstructed dense attention reference.
- Checked for train/test cross-imports with grep. The slice evaluators duplicate their local logic and do not import from each other.
- Checked that `scoring/scores_train.json` and `scoring/scores_test.json` are absent.
- Ran `bash -n` on both train/test shell entry points only. No experiments or evaluators were run.

## Changed

- Added three anti-shortcut metrics to both `eval/train/evaluate.py` and `eval/test/evaluate.py`:
  - `max_relative_l2_error`
  - `min_cosine_similarity`
  - `last_chunk_relative_l2_error`
- These metrics make the split less gameable than mean-only scoring. A method that preserves only the visible/easy chunks can no longer hide a collapsed chunk behind the average, and the latest evaluated chunk directly probes autoregressive error accumulation.
- Updated `scoring/reference.json` so every metric definition has `higher_is_better`, `coefficient`, and `description`.
- Corrected lower-is-better runnable error metric coefficients to negative values.
- Set paper-table `latency_s` and `speedup` coefficients to `0.0` because they are wall-clock hardware-dependent. The runnable efficiency constraint is now carried by `active_block_fraction`.
- Set `dense_equivalent_fraction` coefficient to `0.0` because the current evaluator emits it as a constant context value; it should not influence ranking.
- Shifted experiment weights toward the runnable train/test attention benchmarks while keeping the paper-reported experiments as low-weight context.

## Runtime Concerns For Next Iteration

- The train/test split is still a deterministic chunk-index split over one structured input generator. The new worst/last-chunk metrics should improve shortcut detection, but a future runner should watch for saturation or near-identical gaps across all methods.
- `dense_equivalent_fraction` is constant in the current evaluators. It is intentionally zero-weighted now; remove it in a future cleanup if no downstream tooling requires it.
- The default held-out test mode depends on `scoring/scores_train.json` to enumerate method/experiment entries. That is good for matching train/test coverage, but the runner should confirm the train score file exists before invoking the no-argument test script.
- The paper markdown path from the prompt was absent, but the same paper markdown exists at `paper/2602.04789.md`; metric decisions were checked against that local copy.
