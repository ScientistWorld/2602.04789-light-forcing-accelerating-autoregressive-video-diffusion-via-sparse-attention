# Experiment Overview

## scaled_attention_core

Tests the central efficiency-quality trade-off for autoregressive attention on a deterministic chunked sequence benchmark. The experiment asks whether a method can approximate dense long-context attention while activating only a small fraction of blockwise attention tiles, and compares against both a high-cost local-context baseline and a chunk-agnostic sparse baseline.
