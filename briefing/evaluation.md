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
