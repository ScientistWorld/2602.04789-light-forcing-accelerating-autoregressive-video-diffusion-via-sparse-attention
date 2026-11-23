# Hardening report (iteration 11)

**Bottom line.**
- **Benchmark replaced.** The red team showed the old gym rewarded the wrong things, and that needed a new benchmark, not patches. The old one scored stored attention outputs on diffuse synthetic Q/K/V, where random tiles beat retrieval. The new one is a closed-loop autoregressive (AR) generation benchmark.
- **What a method does now.** A method only chooses which attention tiles to compute. It runs as a sandboxed subprocess and sees nothing but block-pooled queries and keys.
- **What the evaluator owns.** The rollout, every attention output, the tile accounting and every metric. The tile budget is the paper's operating point, and the evaluator enforces it as a hard constraint.
- **Results.**
  - Every shortcut the red team ran is now closed.
  - A genuinely better variant beats the reference on every paired instance.
  - Weaker variants and the baselines score lower.
  - The reference reruns bit-identically.
- **One thing does not follow the paper.** The paper's frame-retrieval stage (part of HSA) gives no gain here. Its CAG-only ablation ties the reference on generation error and outranks it through better long-range consistency (§7).

All numbers below come from real runs:
- job `3ff53c2d-419`: calibration;
- job `a5c5d692-ab2`: reference runs, then variants and shortcuts through `scripts/evaluate_train.sh` and `scripts/evaluate_test.sh`.

Evidence files:
- `redteam/hardening/summary.md`: full tables;
- `redteam/hardening/rank_deltas.json`;
- `redteam/hardening/phase{1,2}_scores_{train,test}.json`;
- `redteam/hardening/sc_read_hidden_test.log`;
- `redteam/hardening/stale_check.txt`.

The old red-team material in `redteam/*` and `REDTEAM.md` is kept as the "before" state.

## 1. What changed

### Benchmark (`eval/train/world.py`; the hidden copy is `eval/test/world.py`)

Each instance generates a video chunk by chunk, the way Self Forcing does:
- 3 frames per chunk, 16×16 tokens per frame;
- 4×4-token tiles, so 16 key blocks per frame;
- a 4-step denoising loop per chunk (σ = 1, .75, .5, .25);
- the denoiser is one 4-head attention layer over the KV cache of all clean past chunks plus the current noisy chunk;
- each finished chunk is appended to the cache, so attention errors propagate to later chunks.

A camera pans, turns and cuts back to places it filmed long ago. Per-instance random query/key features combine:
- fine and coarse world-position codes;
- a temporal code;
- a first-chunk sink code;
- a noise-level code;
- token content;
- a random rotation per head.

Heads therefore show local/diagonal, long-range retrieval, sink and content patterns. Attention is sharp: random tiles now lose on every instance.

The scored quantity is the paper's goal: the video generated with sparse attention should match the dense-attention video from the same noise. The metrics are:
- mean and last-chunk relative error;
- error on revisited places (long-range consistency);
- per-call attention-output cosine.

Experiments:
- `ar_rollout_core`: 7 chunks / 21 frames, the 5 s Self-Forcing setting, at the paper's s_target = 0.9;
- `ar_rollout_long_context`: 10 chunks;
- `ar_rollout_high_sparsity`: s_target = 0.95, added for headroom.

### Interface (`data/selector_protocol.py`, `eval/*/selector_host.py`)

- **Registration.** A method is a command registered as `outputs/methods/<name>.json`.
- **What the selector sees.** The evaluator starts it as a subprocess. For every attention call it sends block-mean queries and keys and budget information.
- **What the selector may return.** Only a 0/1 tile mask of exactly the right shape and dtype. It may also request "probes", which return the exact log-sum-exp of chosen tiles and cost 0.5 tile each.
- **Everything else the evaluator rejects.** Any other array, dtype, shape or value fails the run.
- **No method outputs, statistics or numbers are read.**

### Budget

- The budget is fixed by `eval/`: a dense first chunk plus (1 − s_target) of every later chunk's tiles.
- Methods may redistribute it freely: across chunks, steps, heads and rows.
- Exceeding it in any rollout fails the method.

### Failures

A crash, timeout, invalid reply, exceeded budget or impossible metric value fails the method. It then gets the worst value in **every** metric of **every** experiment:
- 2.0 for the errors, which is their mathematical upper bound because outputs are RMS-normalised;
- −1 for cosine;
- 1.0 for cost.

Nothing is left missing.

### Hidden test (`scripts/evaluate_test.sh`, no arguments)

- Deletes any existing `scoring/scores_test.json` first.
- Enumerates every registration on disk, and runs every experiment for every method.
- Uses seeds that exist only in `eval/test/` (8 + 4 + 8 instances).
- While a selector runs:
  - `eval/test/` is chmod 000;
  - the evaluator is `PR_SET_DUMPABLE 0`, so `/proc/<pid>/mem`, `environ` and `maps` are unreadable;
  - dense reference rollouts are computed only after the selector has been shut down.
- `eval/test/` shares no code with `eval/train/` and imports nothing from `data/` or `method/`.

### Reference method (`method/light_forcing_selector.py`)

- **CAG (Eq. 5–6):**
  - dense first chunk;
  - α_i = 1/√i;
  - s_base = 0.98;
  - β solved so the FLOPs of chunks 2..n equal those of s_target.
- **HSA (Eq. 7–13):**
  - block-mean queries and keys;
  - frame keys = mean of block keys;
  - top-6 past frames per query block, plus the current chunk's frames;
  - top blocks by pooled logit inside those frames.
- **Tile count.** Fixed per query block at (1 − s_i)·#key blocks and chosen within the retrieved frames. This is the reading of Appendix A: "the number of selected blocks … is fixed accordingly … We then perform block selection within the frames … chosen by the frame-wise mask selection".
- **Alternative reading.** The literal Eq. 11, an equal split of that count over the selected frames, is kept as `--selection hsa_per_frame`. It is much worse: core generation error 0.304 vs 0.213, 0/8 instances better (§3). I chose the Appendix-A reading because it is the paper's own implementation detail, not because of its score. Both are reported.

### Removed

These are gone, which removes every path the red team used:
- `data/attention_inputs.py`;
- `eval/attention_benchmark.py`;
- `method/light_forcing_attention.py`;
- `.pt` artifact scoring.

## 2. Reference score before / after

| | before (old gym, test) | after (new gym, hidden test) |
|---|---|---|
| core | rel. L2 0.472, cos 0.890, active 0.104 | generation err **0.2126**, final 0.2763, revisit 0.3059, attention cos 0.9926, tile cost 0.1302 (budget 0.1321) |
| long context | rel. L2 0.485, cos 0.889, active 0.104 | generation err **0.1862**, final 0.2286, revisit 0.3064, cos 0.9951, cost 0.1148 |
| high sparsity | (did not exist) | generation err **0.3485**, final 0.4314, revisit 0.4449, cos 0.9826, cost 0.0818 |

The metrics differ, because the benchmark changed. Reruns of the reference in the same job are bit-identical, with max |difference| = 0 for all six official methods. The method has no randomness; the instances are fixed seeds; the evaluator runs single-threaded in float64.

## 3. Do better results score higher?

Rank Δ vs the reference on the hidden test:
- raw = Σ_exp weight · Σ_metric |coef| · improvement;
- z = the same, with each metric divided by its std over the honest methods.

The paired column is (candidate − reference) generation error over the same instances, as core / long / high-sparsity, followed by the number of instances where the candidate is better.

| method | what changes | before (old gym) raw Δ | after raw Δ | after z Δ | paired Δ gen. error | expected | ok? |
|---|---|---|---|---|---|---|---|
| `lf_alpha1` | stronger chunk-aware growth (α_i = 1/i) | — | **+0.041** | +0.21 | −0.032 / −0.038 / −0.026; **8/8, 4/4, 8/8** better (t ≈ −16, −8, −12) | higher | **yes** |
| `lf_frame_max` | frame score = best block in the frame instead of the mean frame key | — | **+0.048** | +0.24 | −0.005 / −0.010 / −0.005; 7/8, 4/4, 6/8; revisit 0.244 vs 0.306 | higher | **yes** |
| `lf_topk12` | 12 retrieved frames (paper Table 3) | — | +0.025 | +0.13 | ≈0 (t ≈ ±0.4); revisit better | ≈ (paper: robust) | yes |
| `light_forcing_uniform` | CAG removed (paper ablation) | −0.001 | **−0.051** | −0.27 | +0.036 / +0.055 / +0.033; 0/20 better | lower | **yes** (was a tie) |
| `one_d_sparse` | 1D sparse, chunk-agnostic (paper's "+1D sparse") | −0.012 | **−0.021** | −0.12 | +0.038 / +0.053 / +0.032; 0/20 better | lower | **yes** |
| `lf_step_front` | tiles moved to early denoising steps | — | −0.040 | −0.20 | +0.027 / +0.023 / +0.026 | (unknown) | — |
| `lf_explore15` | 15 % of tiles random | — | −0.015 | −0.08 | +0.012 / +0.011 / +0.006 | lower | yes |
| `light_forcing_per_frame` | literal Eq. 11 equal split | — | −0.158 | −0.78 | +0.09 everywhere | — | — |
| `lf_sparser` | spends fewer tiles than allowed (s + 0.03) | −0.221 (budget 0.95) | **−0.160** | −0.88 | +0.067 / +0.067 / +0.247 | lower | **yes** |
| `random_tiles` (seeds 0/1/2) | retrieval removed | **+0.21 / +0.20 / +0.18** | **−0.219 / −0.228 / −0.218** | −1.65 to −1.71 | +0.17 / +0.22 / +0.15; **0/20 better, every seed** | lower | **yes (was the wrong direction)** |
| `local_window` | static spatio-temporal window (STA / sliding style) | −0.987 | −1.360 | −7.7 | +0.77 / +0.83 / +0.69 | lower | yes |
| `light_forcing_cag_only` | frame stage of HSA removed (paper's "+CAG") | 0.000 | **+0.033** | +0.17 | ≈0 (t ≈ +0.45 / −0.41 / −0.39); revisit 0.255 vs 0.306 | lower | **no**, see §7 |

**The paper's claims, as measured here:**
- **Core claim** (Light Forcing beats chunk-agnostic sparse attention and sliding/static windows at a matched budget): reproduced on all 3 experiments and all 20 paired instances, against `one_d_sparse`, `local_window` and `random_tiles`.
- **CAG helps:** reproduced, 0/20 instances better without it.
- **HSA's frame stage helps over CAG + 1D:** not reproduced. It ties on generation error and is worse on revisits.
- **top-k = 6 vs 12 is robust:** reproduced on generation error (paper Table 3).

## 4. Shortcuts: fix, score before and after

"Before" is the red team's raw Δ on the old gym. "After" is each shortcut replayed through the new hidden test (`redteam/hardening/shortcuts.py`, registered under `outputs/methods`, `bash scripts/evaluate_test.sh`). Failed methods carry worst values everywhere: raw Δ −3.56, z −35.

| shortcut | fix | before | after |
|---|---|---|---|
| dense outputs reported with sparse masks | the evaluator computes every attention output from the mask itself; methods cannot submit outputs. `sc_extra_outputs`, which sends extra `outputs`/`stats` arrays, is rejected | +1.105 | **fails** (−3.56) |
| self-reported stats (`active_blocks = 0`) | stats are never read; cost is counted from the masks and probes the evaluator received | +1.147 | not expressible; dense masks fail as below |
| editing the `data/` generator | the test world lives only in `eval/test/`; `data/` holds only the client-side protocol helper; the generator file was deleted | +1.139 | not expressible |
| impossible active fraction (−1 masks, −1e9 stats) | masks must be uint8/bool with 0/1 values and the exact shape (`sc_negative`, `sc_float_mask`) | +0.442 / +4e8 | **fails** |
| stale `scores_test.json` after a crashed test | `evaluate_test.sh` deletes the file first; per-method failures are caught and scored worst, so the test never aborts. A fabricated pre-existing file (`sc_stale_phantom`) and a selector overwriting the file mid-run (`sc_write_scores`) both left no trace | +1.143 | **no effect** (`stale_check.txt`) |
| dropping the long-context entry | experiments come from `eval/`, methods from every registration on disk; a selector that refuses one experiment (`sc_crash_long`) fails everywhere | +0.066 | **fails** |
| dense attention (brute force) | hard tile budget (`sc_dense`) | +0.747 | **fails**: BudgetExceeded |
| raising the budget (s = 0.8 / 0.5) | hard tile budget (`sc_raise_budget`, s = 0.7) | +0.073 / +0.483 | **fails** |
| oracle selection from the full QK^T | selectors never see raw Q/K. Exact tile scores are available only as probes at 0.5 tile each; `sc_probe_all` (probe everything, keep exact top tiles) exceeds the budget | +0.367 | **fails** |
| random tiles (key idea removed) | sharp, content-dependent attention; closed loop | +0.18 to +0.21 | **−0.22** (0/60 paired instances better) |
| NaN outputs | outputs are computed by the evaluator; non-finite or out-of-bound metrics fail the run | +0.042 | not expressible |
| phantom / broken registration | scored worst, does not crash the test (`sc_phantom`) | crashed the whole test | **fails** (worst values) |
| *new:* read hidden seeds / evaluator memory | `eval/test/` chmod 000 while selectors run; evaluator non-dumpable. `sc_read_hidden` got `PermissionError` on `eval/test/*.py`, the directory listing, and `/proc/<ppid>/{mem,environ,maps}` | — | blocked; ties the reference (it falls back to it) |
| *new:* front-loading the whole budget | allowed (legal redistribution), but `sc_frontload` spends it densely on early chunks and then sends empty masks | — | −1.53 |

## 5. Shortcuts left, and why

- **Git history and `.git` objects.** These contain `eval/test/` with the hidden seeds. A selector that decompresses `.git` objects at test time could recover them, mirror the rollout and select with full knowledge.
  - I did not defend against this, because hiding the repository history is the job of the infrastructure that ships the workspace without `eval/test`.
  - The post-hoc grader should flag any selector that reads `.git`, `eval/test` or `/proc`.
  - Note: even full-knowledge exact-mass tile selection did not beat the reference in calibration (core 0.263 vs 0.228). Only a closed-loop search over masks would.
- **Root.** A selector running as root, or with ptrace rights over the parent (`ptrace_scope` 0 combined with a dumpable parent), could read memory. The evaluator sets itself non-dumpable, which blocks this for same-uid processes. Root is out of scope.
- **Leftover processes.** A daemon started by a selector that outlives its process group is not defended against, as instructed. Each selector is killed by process group after its rollouts.
- **Hang guard.** A per-message timeout of 1800 s exists only as a hang guard. An honest selector answers in milliseconds; the whole hidden test takes about 1.5 min per method on one core (plus about 1 min once for the dense rollouts). So stored scores do not depend on load.
- **Probes.** These are an honest, priced capability, not a loophole: they let a method buy exact tile scores. Probing everything exceeds the budget.

## 6. Headroom check

- **No scored metric is at its bound for the reference:**
  - generation error 0.21 / 0.19 / 0.35, against bounds 0–2;
  - attention cosine 0.993, which is not 1;
  - tile cost is unscored (coefficient 0) and enforced as a constraint.
- **Variants that outscore the reference by more than the noise:**
  - `lf_alpha1`: 20/20 paired instances;
  - `lf_frame_max`: 17/20 on generation error, and much better on revisits.
- **Brute force cannot pay:** dense, raised-budget and probe-everything all fail.
- **The added experiment is `ar_rollout_high_sparsity`,** the 0.95 operating point, where the paper's method degrades (error 0.35 vs 0.21) and `lf_alpha1` still improves on it.
- **The simple baselines score clearly below the reference on a scale with room above it:**

  | baseline | raw Δ vs reference |
  |---|---|
  | `one_d_sparse` | −0.021 |
  | `random_tiles` | −0.22 |
  | `local_window` | −1.36 |

## 7. What I could not fix

**The paper's HSA frame stage does not help in this benchmark.**
- `light_forcing_cag_only` (CAG + 1D top blocks over the whole cache, i.e. the paper's "+CAG" row) ties the reference on generation error. The paired t is ≈ ±0.4.
- It is better on `revisit_rel_error`: 0.255 vs 0.306 (core) and 0.185 vs 0.306 (long).
- So it outranks the reference by +0.033 raw (+0.17 z), all of it from the revisit metric.
- The reason: mean-pooled frame keys rank recently seen frames above frames that showed the same place long ago, so the top-6 frame filter sometimes drops exactly the frames a revisit needs. Max-pooled frame scores (`lf_frame_max`) fix this and outrank both.

I did not:
- change the reference, because the paper specifies mean-pooled frame keys and top-6;
- drop or down-weight the revisit metric, because that would only serve the reference;
- retune the world to favour the frame stage. Calibration showed its effect flips sign between world presets (a sharper preset makes it hurt, a softer one makes random tiles win), so any retuning would be choosing the world to match the paper.

The world preset is the one designed before any results (preset A). All three presets, including the two probes, are in `redteam/hardening/calibration.log`; the script is `redteam/hardening/calibrate.py`. In that log, `light_forcing` is the per-frame reading and `lf_hsa_global` is today's reference.

Consequences:
- A future agent can gain about +0.03 by deleting or repairing the frame stage. That gain is real on this benchmark: better long-range consistency at equal error.
- The paper's HSA secondary claim is **not** reproduced here. Its claim against sliding/static windows is.

**Other departures from the paper, each forced by scale:**
- a toy attention world instead of Self-Forcing/LongLive activations;
- one attention layer, with no learned weights and no fine-tuning (the paper post-trains for 2,000 iterations);
- generation fidelity to the dense rollout instead of VBench;
- tile budget instead of latency;
- 16 blocks per frame instead of 24.

The paper's VBench tables are kept unchanged in `scoring/reference.json` (`paper_*` experiments). Their weights are unchanged at 0.05 each; the gym cannot score them.

**Train vs test.** Train has 4 / 2 / 4 instances, test 8 / 4 / 8. Method orderings agree. Absolute long-context errors differ between the slices (e.g. reference 0.272 train vs 0.186 test), because train has only two long instances. The train slice is for development; the test slice decides.

## 8. Weights and coefficients

| experiment | weight | why |
|---|---|---|
| `ar_rollout_core` | 0.40 | the paper's main setting (Table 1, Self Forcing, 5 s) |
| `ar_rollout_long_context` | 0.25 | LongLive / longer generation: the paper's motivation for HSA |
| `ar_rollout_high_sparsity` | 0.15 | headroom: where the method falls short |
| `paper_*` | 0.05 each | unchanged |

| metric | coefficient |
|---|---|
| mean generation error (primary; VBench total analogue) | 1.0 |
| last-chunk error (accumulation / over-saturation, which CAG targets) | 0.5 |
| revisit error (history forgetting, which HSA targets) | 0.5 |
| attention cosine (selection quality per call) | 0.5 |
| tile cost (context only; the budget is a hard constraint) | 0 |

All signs follow `higher_is_better`. Lower-is-better metrics carry negative coefficients; only the magnitude counts.

## 9. How to rerun

```bash
bash scripts/reproduce.sh               # official reference runs, ~20 min on 1 CPU
bash redteam/hardening/replay.sh        # variants + shortcuts through train and the hidden test, then analysis
```
