# Research Direction

## Research Problem

Efficient sparse attention for autoregressive video diffusion. Each new chunk is denoised while attending to a growing KV cache of previously generated chunks. Attention errors are written into that cache and inherited by every later chunk.

## Core Contribution

Generate videos closer to the dense-attention video at the same attention-tile budget. In particular:

- less error accumulation over later chunks;
- better long-range consistency when the scene returns to places seen long ago;
- graceful behaviour at higher sparsity.

## Approach Scope

In-scope work may change:

- how the budget is allocated across chunks, denoising steps, heads and query blocks;
- how candidate context (frames or blocks) is retrieved and scored from the pooled summaries;
- how probes are used to refine selections;
- how selections are reused across steps or chunks.

The method must remain a block-sparse attention policy driven through the selector protocol.

## Out of Bounds

- Reading the hidden instances or evaluator state.
- Modifying the evaluator, world, budget or metrics.
- Exceeding the budget.
- Replacing the problem with model compression, distillation, quantization or caching.
- Dense attention presented as sparse attention. It cannot fit the budget.
