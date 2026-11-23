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

How can autoregressive (AR) video generators reduce the quadratic cost of attention over a growing history without losing what long context provides? Long context gives visual consistency, motion quality, and consistency with places and objects seen long ago.

## Why It Matters

AR video diffusion models generate a video chunk by chunk. Each chunk is denoised in a few steps while attending to the key-value cache of every previously generated chunk. The cache grows with video length, so attention dominates latency in later chunks. Every error a cheaper attention makes is written into the cache, and later chunks inherit it.

## The Task in This Benchmark

You design a **block-sparse attention policy**. For every attention call of an AR rollout, it decides which (query block, key block) tiles are computed. The constraints:

- A fixed tile budget per generated video. It corresponds to a target attention sparsity that the evaluator sets.
- Only block-level summaries of the queries and keys are visible when making the decision. Exact tile scores can be bought at a price, as part of the same budget.

The rollouts are closed-loop. Each generated chunk enters the cache, so errors propagate. A camera moves through a scene and sometimes returns to places filmed long ago.

## Success Criteria

At the same tile budget, a better policy generates videos closer to the dense-attention video. In particular, it shows:

- less error accumulation in later chunks;
- better consistency when revisiting places seen long ago;
- graceful behaviour at higher sparsity and with longer history.

---

# Paper's Method

## Key Contribution

Light Forcing is a sparse attention method tailored to autoregressive video diffusion. It assigns different sparsity budgets to chunks (Chunk-Aware Growth, CAG). It retrieves historical context with a two-stage frame-then-block hierarchy (Hierarchical Sparse Attention, HSA).

## Approach

- **CAG (Sec. 4.1, Eq. 5–6).** The first chunk is dense. Chunk i gets sparsity s_i = s_base − α_i·β, with:
  - α_i ∝ 1/√i: the noise level reached after i chunks of few-step denoising;
  - s_base = 0.98;
  - β solved so that the attention FLOPs of chunks 2..n equal those of a uniform target sparsity (0.9 in the paper).

  Earlier chunks therefore get more budget. Later chunks inherit their priors.
- **HSA (Sec. 4.2, Eq. 7–13).** Queries are mean-pooled per block; keys per block and per frame.
  1. Each query block retrieves its top-k past frames by the inner product with frame-pooled keys (k = 6), and always keeps the current chunk's frames.
  2. Inside those frames it keeps the key blocks with the largest pooled logits.
  3. The number of blocks per query block is (1 − s_i)·(#key blocks) (App. A).
  4. Attention is computed only on the selected tiles.

## Implementation in This Gym

`method/light_forcing_selector.py` implements CAG and HSA on the selector protocol, using the paper's settings: s_base 0.98, top-6 frames, mean pooling. The target sparsity is the evaluator's operating point.

The block count is fixed per query block and chosen within the retrieved frames (App. A). A literal per-frame equal split of Eq. 11 is available as `--selection hsa_per_frame`.

`scripts/method.sh` registers three selectors:

- `light_forcing`;
- `light_forcing_uniform` (CAG removed);
- `light_forcing_cag_only` (HSA's frame stage removed, i.e. 1D block selection with CAG).

## Main Claims

- **Primary:** this sparse attention design preserves AR video quality while cutting attention cost, better than prior sparse attention methods.
- **Secondary:**
  - each component helps;
  - top-k frame retrieval is robust to k;
  - the method combines with FP8 quantization and a faster VAE for real-time generation.

---

# Evaluation

## What the Paper Measured

The paper generates 5-second videos with 1.3B AR video models (Self Forcing, LongLive) and scores them with VBench: quality, semantic and total scores, plus component dimensions. It also reports latency and speedup at about 90% attention sparsity. These numbers are kept for context in `scoring/reference.json` (`paper_*` experiments). This gym cannot regenerate them.

## What This Gym Measures

The gym uses a closed-loop toy AR video-diffusion attention world. Each video consists of:

- 3 frames per chunk, 16×16 tokens per frame, 4×4-token tiles (16 key blocks per frame);
- 4 denoising steps per chunk;
- one 4-head attention layer over the KV cache of all clean past chunks plus the current noisy chunk.

The camera pans, turns and cuts back to places filmed earlier. The evaluator generates each video twice from the same noise: once with dense attention and once with the method's tile masks. It computes every attention output itself.

Metrics, per experiment, averaged over instances:

| metric | direction | meaning |
|---|---|---|
| `mean_generation_rel_error` (primary) | lower | relative L2 distance between the sparse-attention video and the dense-attention video, averaged over chunks (range 0–2) |
| `final_chunk_rel_error` | lower | the same distance on the last chunk (accumulated error) |
| `revisit_rel_error` | lower | the same distance on tokens showing a place filmed before, but not in the 6 frames before the chunk (long-range consistency) |
| `mean_attention_cosine` | higher | cosine between sparse and dense attention outputs of each call, given identical inputs |
| `tile_cost_fraction` | context only | tiles used (+0.5 per probed tile) / dense tiles; the budget is enforced, not scored |

Experiments (operating points fixed by `eval/`):

- `ar_rollout_core`: 7 chunks (21 frames); budget = dense first chunk + 10% of every later chunk's tiles.
- `ar_rollout_long_context`: 10 chunks (30 frames), same budget rule.
- `ar_rollout_high_sparsity`: 7 chunks, budget = dense first chunk + 5% of later tiles.

## Protocol

A method is a **mask selector**: a command that speaks the protocol in `data/selector_protocol.py`. Register it with `bash scripts/register_method.sh <name> <command...>`, which writes `outputs/methods/<name>.json`.

1. For each rollout, the selector receives the experiment's metadata, including the target sparsity and the tile budget.
2. For each attention call it receives:
   - the chunk index, denoising step and noise level;
   - block means of the current chunk's queries and of all keys, per head;
   - the budget spent so far.
3. It answers with a 0/1 mask of shape `[heads, query blocks, key blocks]`. Before answering, it may request exact log-sum-exp values for chosen tiles. Each probed tile costs 0.5 tile.
4. The evaluator computes the masked attention, the rollout and the metrics.

A run **fails** if, in any rollout, the selector:

- crashes or sends a malformed reply (wrong shape or dtype, values other than 0/1, extra arrays);
- exceeds the tile budget.

A failed method receives the worst value of every metric in every experiment (errors 2.0, cosine −1).

- `bash scripts/evaluate_train.sh` scores every registered method on the visible train instances and writes `scoring/scores_train.json`. It accepts `--only <name>` and `--max-instances N` for quick development runs.
- `bash scripts/evaluate.sh` does the same and writes `scoring/scores.json`.
- `bash scripts/evaluate_test.sh` (no arguments) scores every registered method on held-out instances and writes `scoring/scores_test.json`. This is the ranked result.

## Reference Points

On the hidden test, generation error:

| policy | core | long | high sparsity |
|---|---|---|---|
| dense attention | 0 (not allowed: exceeds the budget) | | |
| uniformly random tiles | 0.39 | 0.40 | 0.50 |
| static local window | ~1.0 | | |

The other reference runs, including the paper's method, are listed in `scoring/reference.json`.

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

### Iteration 2: gpt-5.5
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 4m | **GPU**: 0.0h
- **Jobs**: 6 total (6 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-05 03:36] - secondary_claims
- Implemented and validated Light Forcing CAG + HSA on deterministic scaled autoregressive attention tensors.
- Reproduced the core sparse-attention quality/cost claim on default and longer-context settings.
- Reproduced the CAG secondary claim by showing chunk-aware allocation improves over fixed post-first-chunk sparsity at matched active block fraction.

</details>

### Iteration 3: gpt-5.5
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 6m | **GPU**: 0.0h
- **Jobs**: 6 total (6 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-05 03:46] - secondary_claims
- Audited the continuation workspace after the milestone reset to `core_claim_plus`.
- Confirmed the method implements the paper's CAG sparsity allocation and HSA frame/block mask selection, not a surrogate model.
- Re-ran `python validate.py --compare`; scoring structure, eval/method separation, and reproduced scaled scores all pass.
- Restored the milestone to `secondary_claims`.

### [2026-06-05 03:42] - secondary_claims
- Verified the packaged scaled reproduction with `python validate.py --compare`.
- Confirmed existing completed jobs cover method execution, core claim, longer-context core claim plus, and the CAG allocation ablation.
- Aligned the machine-readable milestone with the documented achieved `secondary_claims` state.

</details>

### Iteration 4: gpt-5.5 (designer_stage_1)
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 5m | **GPU**: 0.0h
- **Jobs**: 6 total (6 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-05 03:46] - secondary_claims
- Audited the continuation workspace after the milestone reset to `core_claim_plus`.
- Confirmed the method implements the paper's CAG sparsity allocation and HSA frame/block mask selection, not a surrogate model.
- Re-ran `python validate.py --compare`; scoring structure, eval/method separation, and reproduced scaled scores all pass.
- Restored the milestone to `secondary_claims`.

### [2026-06-05 03:42] - secondary_claims
- Verified the packaged scaled reproduction with `python validate.py --compare`.
- Confirmed existing completed jobs cover method execution, core claim, longer-context core claim plus, and the CAG allocation ablation.
- Aligned the machine-readable milestone with the documented achieved `secondary_claims` state.

</details>

### Iteration 5: gpt-5.5 (designer_stage_2)
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 5m | **GPU**: 0.0h
- **Jobs**: 6 total (6 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-05 03:46] - secondary_claims
- Audited the continuation workspace after the milestone reset to `core_claim_plus`.
- Confirmed the method implements the paper's CAG sparsity allocation and HSA frame/block mask selection, not a surrogate model.
- Re-ran `python validate.py --compare`; scoring structure, eval/method separation, and reproduced scaled scores all pass.
- Restored the milestone to `secondary_claims`.

### [2026-06-05 03:42] - secondary_claims
- Verified the packaged scaled reproduction with `python validate.py --compare`.
- Confirmed existing completed jobs cover method execution, core claim, longer-context core claim plus, and the CAG allocation ablation.
- Aligned the machine-readable milestone with the documented achieved `secondary_claims` state.

</details>

### Iteration 6: gpt-5.5 (designer_stage_3)
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 2m | **GPU**: 0.0h
- **Jobs**: 7 total (7 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-05 03:46] - secondary_claims
- Audited the continuation workspace after the milestone reset to `core_claim_plus`.
- Confirmed the method implements the paper's CAG sparsity allocation and HSA frame/block mask selection, not a surrogate model.
- Re-ran `python validate.py --compare`; scoring structure, eval/method separation, and reproduced scaled scores all pass.
- Restored the milestone to `secondary_claims`.

### [2026-06-05 03:42] - secondary_claims
- Verified the packaged scaled reproduction with `python validate.py --compare`.
- Confirmed existing completed jobs cover method execution, core claim, longer-context core claim plus, and the CAG allocation ablation.
- Aligned the machine-readable milestone with the documented achieved `secondary_claims` state.

</details>

### Iteration 7: gpt-5.5 (packager)
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 1m | **GPU**: 0.0h
- **Jobs**: 7 total (7 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-05 03:46] - secondary_claims
- Audited the continuation workspace after the milestone reset to `core_claim_plus`.
- Confirmed the method implements the paper's CAG sparsity allocation and HSA frame/block mask selection, not a surrogate model.
- Re-ran `python validate.py --compare`; scoring structure, eval/method separation, and reproduced scaled scores all pass.
- Restored the milestone to `secondary_claims`.

### [2026-06-05 03:42] - secondary_claims
- Verified the packaged scaled reproduction with `python validate.py --compare`.
- Confirmed existing completed jobs cover method execution, core claim, longer-context core claim plus, and the CAG allocation ablation.
- Aligned the machine-readable milestone with the documented achieved `secondary_claims` state.

</details>

### Iteration 8: gpt-5.5 (verifier)
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 6m | **GPU**: 0.0h
- **Jobs**: 8 total (8 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-05 03:46] - secondary_claims
- Audited the continuation workspace after the milestone reset to `core_claim_plus`.
- Confirmed the method implements the paper's CAG sparsity allocation and HSA frame/block mask selection, not a surrogate model.
- Re-ran `python validate.py --compare`; scoring structure, eval/method separation, and reproduced scaled scores all pass.
- Restored the milestone to `secondary_claims`.

### [2026-06-05 03:42] - secondary_claims
- Verified the packaged scaled reproduction with `python validate.py --compare`.
- Confirmed existing completed jobs cover method execution, core claim, longer-context core claim plus, and the CAG allocation ablation.
- Aligned the machine-readable milestone with the documented achieved `secondary_claims` state.

</details>

### Iteration 9: gpt-5.5 (verifier)
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 7m | **GPU**: 0.0h
- **Jobs**: 8 total (8 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-05 04:18] - secondary_claims
- Completed final verification of folder structure, method-agnostic documentation, reusable scoring, train/test split artifacts, and download portability.
- Removed generated Python cache directories from source folders.
- Kept the milestone at `secondary_claims`; the workspace is ready as a scaled attention-computation gym with the documented limitation that it does not run the full video/VBench stack.

### [2026-06-05 04:15] - secondary_claims
- Re-audited the existing reproduction after a fresh paper read and sidecar extraction.
- Confirmed the method code implements Chunk-Aware Growth plus Hierarchical Sparse Attention with dense first chunk, blockwise/framewise mean-pooling, top-k historical frame retrieval, and block-level sparse attention.
- Confirmed `python validate.py --compare` passes and prior container job `7998f48d-bf2` verified the packaged reproduction path.
- Kept the milestone at `secondary_claims`; higher milestones would require the unreleased/full 1.3B video generation and VBench stack.

### [2026-06-05 03:46] - secondary_claims
- Audited the continuation workspace after the milestone reset to `core_claim_plus`.
- Confirmed the method implements the paper's CAG sparsity allocation and HSA frame/block mask selection, not a surrogate model.
- Re-ran `python validate.py --compare`; scoring structure, eval/method separation, and reproduced scaled scores all pass.
- Restored the milestone to `secondary_claims`.

### [2026-06-05 03:42] - secondary_claims
- Verified the packaged scaled reproduction with `python validate.py --compare`.
- Confirmed existing completed jobs cover method execution, core claim, longer-context core claim plus, and the CAG allocation ablation.
- Aligned the machine-readable milestone with the documented achieved `secondary_claims` state.

</details>

### Iteration 10: gpt-6.1-sol (redteam)
- **Milestone**: `secondary_claims` | **Status**: error
- **Working time**: 15m | **GPU**: 0.0h
- **Jobs**: 8 total (8 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-05 04:18] - secondary_claims
- Completed final verification of folder structure, method-agnostic documentation, reusable scoring, train/test split artifacts, and download portability.
- Removed generated Python cache directories from source folders.
- Kept the milestone at `secondary_claims`; the workspace is ready as a scaled attention-computation gym with the documented limitation that it does not run the full video/VBench stack.

### [2026-06-05 04:15] - secondary_claims
- Re-audited the existing reproduction after a fresh paper read and sidecar extraction.
- Confirmed the method code implements Chunk-Aware Growth plus Hierarchical Sparse Attention with dense first chunk, blockwise/framewise mean-pooling, top-k historical frame retrieval, and block-level sparse attention.
- Confirmed `python validate.py --compare` passes and prior container job `7998f48d-bf2` verified the packaged reproduction path.
- Kept the milestone at `secondary_claims`; higher milestones would require the unreleased/full 1.3B video generation and VBench stack.

### [2026-06-05 03:46] - secondary_claims
- Audited the continuation workspace after the milestone reset to `core_claim_plus`.
- Confirmed the method implements the paper's CAG sparsity allocation and HSA frame/block mask selection, not a surrogate model.
- Re-ran `python validate.py --compare`; scoring structure, eval/method separation, and reproduced scaled scores all pass.
- Restored the milestone to `secondary_claims`.

### [2026-06-05 03:42] - secondary_claims
- Verified the packaged scaled reproduction with `python validate.py --compare`.
- Confirmed existing completed jobs cover method execution, core claim, longer-context core claim plus, and the CAG allocation ablation.
- Aligned the machine-readable milestone with the documented achieved `secondary_claims` state.

</details>

### Iteration 10: claude-opus-5-5 (redteam)
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 15m | **GPU**: 0.0h
- **Jobs**: 10 total (10 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-05 04:18] - secondary_claims
- Completed final verification of folder structure, method-agnostic documentation, reusable scoring, train/test split artifacts, and download portability.
- Removed generated Python cache directories from source folders.
- Kept the milestone at `secondary_claims`; the workspace is ready as a scaled attention-computation gym with the documented limitation that it does not run the full video/VBench stack.

### [2026-06-05 04:15] - secondary_claims
- Re-audited the existing reproduction after a fresh paper read and sidecar extraction.
- Confirmed the method code implements Chunk-Aware Growth plus Hierarchical Sparse Attention with dense first chunk, blockwise/framewise mean-pooling, top-k historical frame retrieval, and block-level sparse attention.
- Confirmed `python validate.py --compare` passes and prior container job `7998f48d-bf2` verified the packaged reproduction path.
- Kept the milestone at `secondary_claims`; higher milestones would require the unreleased/full 1.3B video generation and VBench stack.

### [2026-06-05 03:46] - secondary_claims
- Audited the continuation workspace after the milestone reset to `core_claim_plus`.
- Confirmed the method implements the paper's CAG sparsity allocation and HSA frame/block mask selection, not a surrogate model.
- Re-ran `python validate.py --compare`; scoring structure, eval/method separation, and reproduced scaled scores all pass.
- Restored the milestone to `secondary_claims`.

### [2026-06-05 03:42] - secondary_claims
- Verified the packaged scaled reproduction with `python validate.py --compare`.
- Confirmed existing completed jobs cover method execution, core claim, longer-context core claim plus, and the CAG allocation ablation.
- Aligned the machine-readable milestone with the documented achieved `secondary_claims` state.

</details>

### Iteration 11: claude-opus-5-5 (hardener)
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 49m | **GPU**: 0.0h
- **Jobs**: 12 total (12 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-11-23] - secondary_claims (hardening)
- Rebuilt the gym as a closed-loop AR sparse-attention benchmark with sandboxed mask selectors, an enforced tile budget, and hidden test seeds.
- Reproduced:
  - the core claim (Light Forcing beats 1D sparse, local window and random tiles at matched budget on 20/20 paired instances);
  - CAG's benefit.
- Not reproduced: HSA's frame-retrieval benefit over CAG + 1D selection.
- See HARDENING.md.

### [2026-06-05 04:18] - secondary_claims
- Completed final verification of folder structure, method-agnostic documentation, reusable scoring, train/test split artifacts, and download portability.
- Removed generated Python cache directories from source folders.
- Kept the milestone at `secondary_claims`; the workspace is ready as a scaled attention-computation gym with the documented limitation that it does not run the full video/VBench stack.

### [2026-06-05 04:15] - secondary_claims
- Re-audited the existing reproduction after a fresh paper read and sidecar extraction.
- Confirmed the method code implements Chunk-Aware Growth plus Hierarchical Sparse Attention with dense first chunk, blockwise/framewise mean-pooling, top-k historical frame retrieval, and block-level sparse attention.
- Confirmed `python validate.py --compare` passes and prior container job `7998f48d-bf2` verified the packaged reproduction path.
- Kept the milestone at `secondary_claims`; higher milestones would require the unreleased/full 1.3B video generation and VBench stack.

### [2026-06-05 03:46] - secondary_claims
- Audited the continuation workspace after the milestone reset to `core_claim_plus`.
- Confirmed the method implements the paper's CAG sparsity allocation and HSA frame/block mask selection, not a surrogate model.
- Re-ran `python validate.py --compare`; scoring structure, eval/method separation, and reproduced scaled scores all pass.
- Restored the milestone to `secondary_claims`.

### [2026-06-05 03:42] - secondary_claims
- Verified the packaged scaled reproduction with `python validate.py --compare`.
- Confirmed existing completed jobs cover method execution, core claim, longer-context core claim plus, and the CAG allocation ablation.
- Aligned the machine-readable milestone with the documented achieved `secondary_claims` state.

</details>


---

# Reproduction Milestones

**Current: secondary_claims**

## Stop Justification
- Completed at milestone `secondary_claims`.
