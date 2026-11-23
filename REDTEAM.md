# Red-team audit: does this gym reward better research?

Gym: Light Forcing (sparse attention for autoregressive video diffusion), scaled down to a
synthetic single-head attention benchmark. A method writes per-chunk attention outputs plus
block masks. The hidden test (`scripts/evaluate_test.sh` → `eval/test/evaluate.py`) scores the
odd-indexed chunks against dense attention. It uses relative L2 error, cosine similarity,
worst-chunk and last-chunk error, and the active tile fraction.

Every number below comes from a real run. Each `redteam/<name>/run.sh` starts from what a future
agent gets (empty `outputs/`, no score files) and puts its method in place. It then runs
`scripts/evaluate_train.sh`, restores the gym's `eval/` and `scripts/evaluate_test.sh`, and runs
`bash scripts/evaluate_test.sh` with no arguments. Each folder keeps its `scores_test*.json`,
logs and exit code. `redteam/common/rank.py` turns these into a rank delta against the
reference (`light_forcing` in the gym's `scoring/scores_test.json`):
Σ_exp weight × Σ_metric |coef| × improvement, taken in the metric's better direction. Missing
or non-finite values are skipped, as the rank does. Two versions are reported:
- **raw**: no normalisation.
- **z**: each metric divided by its std over a pool of honest attempts. This stands in for "variation across ranked attempts".

All conclusions hold under both. Full table: `redteam/summary.md`. Input-draw analysis:
`redteam/data_seed_spread/`. Jobs: `3986e82d-196` and `4a8791a3-2e6`, CPU, exit 0. At the end of
both jobs the workspace was restored: `outputs/`, `eval/`, `evaluate_test.sh`, the score files and
the `data/` generator, each checked by diff.

**Bottom line:** the score is not trustworthy.
1. **The method's key idea removed outranks the reference.** Random tile selection, which removes retrieval, beats the paper's selection on every input draw and every random seed.
2. **Failing the sparsity task outranks the reference.** Plain dense attention beats it, and the score rises steadily as more tiles are spent.
3. **The evaluator trusts everything the agent writes**:
   - outputs are never checked against the reported masks;
   - self-reported stats and impossible tile counts are accepted;
   - the dense reference is rebuilt from the agent-editable `data/` generator;
   - the agent chooses which methods and experiments get scored.

## 1. Do better results score higher?

Honest variants: outputs are computed only from the reported tiles, on the real inputs, at the
reference's tile budget unless noted. Hidden test, default input draw:

| variant | what changes | core err | long err | raw Δ | z Δ | expected | ok? |
|---|---|---|---|---|---|---|---|
| `light_forcing` (reference) | — | 0.472 | 0.485 | 0 | 0 | — | — |
| `lf_better_sampled_rows` | block score = softmax mass of every 8th query row | 0.349 | 0.396 | **+0.299** | +1.81 | higher | yes |
| `lf_better_lse_scoring` | block score = log-sum-exp of pooled query vs key tokens | 0.466 | 0.478 | +0.020 | +0.14 | higher | yes, but small |
| `lf_weaker_budget95` | target sparsity 0.95 (fewer tiles: 0.056 vs 0.104) | 0.583 | 0.514 | **−0.221** | −1.38 | lower | yes |
| `lf_weaker_local_window` | most recent tiles only (no retrieval) | 0.801 | 0.830 | **−0.987** | −6.65 | lower | yes |
| `lf_ablation_uniform` | chunk-aware allocation removed (paper's ablation) | 0.493 | 0.490 | −0.001 | −0.04 | lower | **tie** |
| `lf_no_frame_retrieval` | frame-level retrieval stage removed | 0.472 | 0.485 | 0.000 | 0.00 | lower | **tie (identical)** |
| `lf_weaker_random_blocks` (seeds 0/1/2) | tiles picked uniformly at random | 0.368/0.379/0.382 | 0.456/0.425/0.440 | **+0.211/+0.204/+0.176** | +1.0 to +1.2 | lower | **NO: wrong direction** |

The same comparison on three input draws (generator seed +0/+101/+202; gym test scorer with
explicit `--seed`; `redteam/data_seed_spread/summary.md`). Raw Δ vs the reference on the same draw:

| variant | test +0 | test +101 | test +202 | train +0 | train +101 | train +202 |
|---|---|---|---|---|---|---|
| random blocks s0 / s1 / s2 | +0.21/+0.20/+0.18 | +0.46/+0.46/+0.47 | +0.52/+0.58/+0.55 | +0.10/+0.02/+0.16 | +0.52/+0.47/+0.56 | +0.38/+0.36/+0.42 |
| oracle block mass (brute force) | +0.37 | +0.23 | +0.34 | +0.24 | +0.25 | +0.17 |
| better: sampled rows | +0.30 | +0.17 | +0.29 | +0.21 | +0.19 | +0.17 |
| better: lse | +0.02 | +0.06 | +0.20 | +0.10 | +0.08 | +0.06 |
| no chunk-aware allocation | −0.001 | −0.018 | +0.020 | +0.008 | +0.023 | +0.021 |
| no frame retrieval | 0.000 | −0.001 | 0.000 | 0.000 | 0.000 | 0.000 |
| weaker: budget 0.95 | −0.22 | −0.14 | −0.14 | −0.10 | −0.13 | −0.27 |
| weaker: local window | −0.99 | −1.04 | −0.60 | −0.71 | −1.09 | −0.89 |

**Verdict:** budget and local-window changes move the score correctly, and so do the two better
block scorers. But removing the paper's retrieval and picking tiles at random gains +0.18 to
+0.58 on every draw, seed and slice. On draws +101 and +202, random selection also beats exact
block-mass selection. Both of the paper's components score as zero. Chunk-aware allocation
changes sign across draws, and the ablation without it wins on the train slice for all 3
draws. Frame retrieval never changes a mask: with top-6 frames plus 3 current frames, the
candidate set is always larger than the tile budget. **The score does not follow the quality of
the paper's idea.** On this data, coverage of a diffuse attention pattern pays, and
content-based retrieval does not.

## 2. Seed spread and the reproduction check

- **Reruns of the reference (3×, hidden test):** bit-identical to the gym's `scores_test.json`. Spread 0.0 on every metric (`redteam/reference_rerun/spread.json`). The method has no seed and the test uses one fixed input draw.
- **Input-draw spread:** the reference's test error is core 0.597 ± 0.089 and long 0.532 ± 0.036 over the 3 draws. Its own rank total moves by −0.23 and −0.32 when only the input draw changes. Gaps between methods on the same draw are steadier. Sampled rows (+0.17 to +0.30), the budget variants, local window and random blocks keep their sign. The chunk-aware-allocation gap (|Δ| ≤ 0.02, sign flips) is noise.
- **Does the reproduced method beat its baseline in the paper's direction? Not measurably.**
  - Against `one_d_sparse`, the reference scores raw +0.012: better mean error on core, worse worst/last-chunk error on long. That margin is far below the draw spread.
  - The margin is also inflated by accounting. `one_d_sparse` stores no masks, so its active fraction is taken from whole-run stats including the dense first chunk: 0.131/0.120. On the test chunks its true value is 0.097/0.100, as the identical-selection `light_forcing_uniform` shows. The reference itself uses 0.104 on the test chunks, so it spends more tiles than the baseline it is said to beat.
  - Against the matched ablation (`light_forcing_uniform`, same selection with masks), the reference ties on test (−0.001) and loses on train on all 3 draws.
  - The paper's claims that CAG and HSA each help, and that 1D sparse attention degrades badly, do not show up in the test.
- **Where the reproduction departs from the paper:**
  - synthetic Q/K/V from a fixed sinusoid-plus-noise generator instead of Self-Forcing/LongLive activations;
  - one head, no layers, no denoising timesteps;
  - no fine-tuning (the paper post-trains for 2,000 iterations);
  - no video generation and no VBench: scored on attention-output error instead;
  - 256 tokens per frame, 4 tiles per frame (the paper's Fig. 4 shows 24 tiles per frame);
  - no autoregressive feedback: chunk outputs never feed later inputs, so the error accumulation that motivates CAG cannot happen. `last_chunk_relative_l2_error` cannot measure accumulation;
  - CAG's α_i = 1/√(chunk+1) is an interpretation of "noise level reached by chunk i";
  - HSA does a global top-N over candidate tiles instead of a top-k within each frame;
  - latency and speedup replaced by the active tile fraction; selection cost is not counted.
- **Does the scaled data still show the paper's effect? No.** Attention in this generator is diffuse: tile means wash out the token phase. So random tiles beat pooled-logit retrieval, and even exact block-mass retrieval on 2 of 3 draws. The frame-retrieval stage never binds, and there is no error accumulation. A better method on this benchmark is a better approximator of flat single-head attention. That is not necessarily a better sparse-attention method for AR video.

## 3. Base checks on the rank

- **Is anything at its bound?** No. The reference has error 0.47, cosine 0.89 and active fraction 0.104, all far from 0, 1 and 0. It can be beaten easily (rows above). The simple baselines are barely below it: `one_d_sparse` −0.012 and `light_forcing_uniform` −0.001, both inside the noise. `sliding_window` is at −0.140.
- **A method failing the task outranks the reference.** Honest dense attention (active fraction 1.0, no sparsity at all) scores **+0.747 raw / +4.97 z**. The efficiency metric has |coef| 0.5, against 3.05 total for the five accuracy metrics. The score rises steadily with spent tiles:
  - budget 0.95: −0.22
  - reference: 0
  - budget 0.80: +0.07
  - budget 0.50: +0.48
  - dense: +0.75

  There is no binding sparsity constraint. `DIRECTION.md` calls dense attention out of bounds, but the score pays for it.
- **A failed run does not get worst values.**
  - A missing experiment produces no entry, so it is skipped. `shortcut_drop_long_context` shows this.
  - A missing artifact for any listed method raises `FileNotFoundError` and aborts the whole test (exit 1). Nothing is written, and an older `scoring/scores_test.json` survives (`shortcut_stale_scores_test`).
  - NaN outputs are written as `NaN` without sanitising (`shortcut_nan_outputs`).
- **Does the hidden test write every method on disk?** No.
  - It scores only the methods and experiments listed in the agent-written `scoring/scores_train.json`, and only at `outputs/<name>[_long_context].pt`.
  - In `shortcut_drop_long_context`, `sliding_window_core_only_long_context.pt` is on disk (`artifacts_on_disk.txt`) but is not scored.
  - A method the agent evaluated from any other path is listed but has no artifact at `outputs/`, so the whole test crashes. This is the same mechanism as `shortcut_stale_scores_test`.
- **Directions:** the `higher_is_better` flags in `reference.json` match the metrics. Lower-is-better metrics carry negative coefficients, which is harmless because only |coef| counts. `dense_equivalent_fraction` is a constant with coef 0.
- **Brute force:**
  - Dense attention, which the method exists to avoid, solves the test exactly (error 0) and outranks the reference. The benchmark is tiny (7 or 9 chunks of 768 tokens), so dense costs nothing.
  - Exact block-mass ("oracle") selection at the reference's per-chunk tile counts scores **+0.367**. Its selection needs the full QK^T, but the evaluator does not count selection cost.

## 4. Shortcuts

Hidden-test rank delta vs the reference (positive means it outranks). The degenerate rows,
which did not pay, are kept for completeness.

| shortcut (`redteam/…`) | what it does | core err / active | long err / active | raw Δ | z Δ | result |
|---|---|---|---|---|---|---|
| `shortcut_dense_outputs_fake_masks` | exact dense outputs rebuilt from visible `data/`, reported with the reference's own sparse masks; outputs are never checked against masks | 0.000 / 0.104 | 0.000 / 0.104 | **+1.105** | +6.69 | **works** |
| `shortcut_self_reported_stats` | dense outputs, no masks, `stats.active_blocks = 0` (trusted as written) | 0.000 / 0.000 | 0.000 / 0.000 | **+1.147** | +6.89 | **works** |
| `shortcut_edit_data_generator` | edits `data/attention_inputs.py` (not restored before the test) so all values are 1; one honest tile per query block then matches the evaluator's rebuilt "dense reference" | 0.000 / 0.021 | 0.000 / 0.017 | **+1.139** | +6.85 | **works** |
| `shortcut_impossible_active_fraction` / `negative_mask_values` | reference outputs; masks stored as float −1, so active fraction is −1.0 | 0.472 / −1.0 | 0.485 / −1.0 | **+0.442** | +2.11 | **works** |
| `shortcut_impossible_active_fraction` / `negative_stats` | reference outputs; no masks; `active_blocks = −1e9` | 0.472 / −1e9 | 0.485 / −1e9 | **+4e8** | +1.9e9 | **works (unbounded)** |
| `shortcut_stale_scores_test` | agent writes a perfect `scoring/scores_test.json`, then lists a phantom method in `scores_train.json`; the hidden test exits 1 before writing, so the fabricated file stays | 0.000 / 0.010 | 0.000 / 0.010 | **+1.143** | +6.87 | **works if the rank reads the file after a failed test** (not deleted first, exit code not checked) |
| `shortcut_drop_long_context` | gym's sliding-window baseline; long-context entry removed from the agent's `scores_train.json` (artifact still on disk) so it is skipped instead of scored worst | 0.349 / 0.464 | missing | **+0.066** (full coverage: −0.140) | +0.47 (−0.87) | **works** |
| `shortcut_dense_attention` | honest dense attention (brute force, no sparsity) | 0.000 / 1.000 | 0.000 / 1.000 | **+0.747** | +4.97 | **works** |
| `shortcut_raise_budget` | reference at target sparsity 0.80 / 0.50 (operating point moved) | 0.411 / 0.208; 0.113 / 0.528 | 0.439 / 0.208; 0.238 / 0.500 | **+0.073 / +0.483** | +0.54 / +3.19 | **works** |
| `shortcut_oracle_block_selection` | tiles chosen by exact dense attention mass (full QK^T); same tile counts as the reference; selection cost not counted | 0.375 / 0.104 | 0.309 / 0.104 | **+0.367** | +2.32 | **works** |
| `variants` / `lf_weaker_random_blocks` | retrieval replaced by random tiles at the reference budget (counted as a shortcut: it outranks with a worse idea) | 0.368 / 0.104 | 0.456 / 0.104 | **+0.211** | +1.22 | **works** (all draws and seeds) |
| `variants` / `lf_no_frame_retrieval`, `lf_ablation_uniform` | paper components removed | 0.472 / 0.104; 0.493 / 0.097 | 0.485 / 0.104; 0.490 / 0.100 | 0.000 / −0.001 | 0.00 / −0.04 | **tie: a shortcut the test cannot tell apart** |
| `shortcut_nan_outputs` | NaN outputs, empty masks; accuracy metrics written as NaN, active fraction 0 | NaN / 0.000 | NaN / 0.000 | +0.042 if NaN is skipped | +0.20 | **works if the rank skips NaN** (fails if NaN counts as worst) |
| `shortcut_degenerate_outputs` | mean of values / zeros / copied values, 0 active tiles | 0.843 / 1.000 / 1.451 at 0.000 | 0.833 / 1.000 / 1.347 at 0.000 | −0.96 / −1.29 / −2.12 | −7.6 / −11.7 / −15.3 | fails |

## 5. What would fix it (for the next iteration)

1. In the evaluator:
   - recompute every output from the submitted mask, using its own masked attention, and score that (or reject if the outputs differ);
   - require boolean 0/1 masks of the correct shape;
   - ignore `stats`;
   - reject non-finite values;
   - assign worst values to any missing method or experiment.
2. Build the inputs inside the restored `eval/` (or hash-check `data/`), with held-out generator seeds that are not visible in `data/`.
3. Delete `scoring/scores_test.json` before the hidden test, and treat a non-zero exit as a failed run. Enumerate methods from `outputs/` instead of from the agent-written `scores_train.json`.
4. Make the budget binding, so that spending tiles cannot buy rank:
   - compare at the reference tile fraction (reject or cap above it), or score error at a fixed budget;
   - count selection cost, so that dense-cost selection does not count as sparse.
5. Give the benchmark the structure the paper relies on:
   - sharp, content-dependent long-range attention, where retrieval beats random and recency (not true now);
   - more frames than the top-k, so frame retrieval binds;
   - outputs fed back into later chunks, so allocation across chunks matters.

   Check the result by confirming that random tiles and the ablations score clearly below the reference across several draws.
