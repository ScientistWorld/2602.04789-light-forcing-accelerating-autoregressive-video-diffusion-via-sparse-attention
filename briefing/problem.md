# Problem Definition

## Research Question

How can autoregressive video generators reduce the quadratic cost of growing historical attention without losing the visual consistency and motion quality that come from using long context?

## Why It Matters

Autoregressive video models generate chunks sequentially, so their key-value history grows with video length. Attention can dominate latency in later chunks, limiting interactive or real-time deployment.

## Success Criteria

A successful solution should preserve dense-attention outputs or video-quality metrics while using substantially fewer active attention blocks as the autoregressive context grows. It should also avoid early-chunk degradation and long-context forgetting.
