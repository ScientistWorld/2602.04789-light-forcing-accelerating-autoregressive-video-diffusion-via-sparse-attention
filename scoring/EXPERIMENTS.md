# Experiment Overview

## scaled_attention_core

Tests the central efficiency-quality trade-off for autoregressive attention on a deterministic chunked sequence benchmark. The experiment asks whether a method can approximate dense long-context attention while activating only a small fraction of blockwise attention tiles. It compares submitted artifacts against local-context and chunk-agnostic sparse baselines.

## scaled_attention_long_context

Repeats the same efficiency-quality test with more autoregressive chunks. This setting checks whether a submitted sparse attention method remains useful as the amount of cached history grows.
