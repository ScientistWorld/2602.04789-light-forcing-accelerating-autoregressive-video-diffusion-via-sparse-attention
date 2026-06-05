# Research Direction

## Research Problem

This benchmark targets efficient sparse attention for autoregressive video-style generation, where each new chunk attends to a growing history and dense attention becomes increasingly expensive.

## Core Contribution

Work should improve sparse long-context attention for chunked autoregressive generation while preserving dense-attention behavior.

## Approach Scope

In-scope work may change how candidate methods assign attention budgets, select historical context, construct masks, score candidate tiles, or account for sparse attention cost. The method must remain a sparse attention method for chunked autoregressive context.

## Out of Bounds

Replacing the problem with model compression, distillation, quantization-only acceleration, caching-only acceleration, or a non-attention sequence model is out of scope. Reporting dense attention as an improved sparse method is also out of scope.
