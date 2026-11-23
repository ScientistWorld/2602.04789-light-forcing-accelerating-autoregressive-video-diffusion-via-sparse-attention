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
