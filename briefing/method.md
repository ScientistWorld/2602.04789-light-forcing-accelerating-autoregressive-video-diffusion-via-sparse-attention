# Paper's Method

## Key Contribution

Light Forcing is a sparse attention method tailored to autoregressive video diffusion. Its key idea is to assign different sparsity budgets to chunks and to retrieve historical context in a two-stage frame/block hierarchy.

## Approach

The method keeps the first chunk dense, then uses Chunk-Aware Growth to solve a chunk-specific sparsity allocation from the target sparsity and an error model that gives earlier chunks more attention budget. For each later chunk, Hierarchical Sparse Attention first mean-pools query/key blocks, retrieves top historical frames for each query block, selects the most relevant key blocks inside those frames, and computes masked blockwise attention.

## Main Claims

The primary claim is that this sparse attention design preserves or improves autoregressive video quality while reducing attention cost. Secondary claims are that each component matters, top-k frame retrieval is robust, and the method can combine with quantization and a faster VAE for real-time deployment.
