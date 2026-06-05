# Experiment Overview

## scaled_attention_core

Tests the central efficiency-quality trade-off for autoregressive attention on a deterministic chunked sequence benchmark. The experiment asks whether a method can approximate dense long-context attention while activating only a small fraction of blockwise attention tiles, and compares against both a high-cost local-context baseline and a chunk-agnostic sparse baseline.
It also includes a fixed-sparsity variant to test whether adapting the sparsity budget across chunks matters.

## scaled_attention_long_context

Repeats the same efficiency-quality test with more autoregressive chunks. This setting checks whether a submitted sparse attention method remains useful as the amount of cached history grows.
It includes the same fixed-sparsity variant as the core setting.
