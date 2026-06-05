# Research Direction

## Research Problem

This benchmark targets efficient sparse attention for autoregressive video-style generation, where each new chunk attends to a growing history and dense attention becomes increasingly expensive.

## Core Contribution

Work should improve sparse long-context attention for chunked autoregressive generation while preserving dense-attention behavior.

## Approach Scope

In-scope work may change sparsity allocation, frame or block retrieval, mask construction, block scoring, or sparse attention accounting. The method must remain a sparse attention method for chunked autoregressive context.

## Out of Bounds

Replacing the problem with model compression, distillation, quantization-only acceleration, caching-only acceleration, or a non-attention sequence model is out of scope. Reporting dense attention as an improved sparse method is also out of scope.
