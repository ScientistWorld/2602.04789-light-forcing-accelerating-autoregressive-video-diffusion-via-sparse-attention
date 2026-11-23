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
