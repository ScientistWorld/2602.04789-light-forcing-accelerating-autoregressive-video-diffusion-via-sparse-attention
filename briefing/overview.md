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
