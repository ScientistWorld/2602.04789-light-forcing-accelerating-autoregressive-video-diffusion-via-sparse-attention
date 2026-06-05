# Reproduction Milestones

**Current: secondary_claims**

## Progress Log

### [2026-06-05 04:15] - secondary_claims
- Re-audited the existing reproduction after a fresh paper read and sidecar extraction.
- Confirmed the method code implements Chunk-Aware Growth plus Hierarchical Sparse Attention with dense first chunk, blockwise/framewise mean-pooling, top-k historical frame retrieval, and block-level sparse attention.
- Confirmed `python validate.py --compare` passes and prior container job `7998f48d-bf2` verified the packaged reproduction path.
- Kept the milestone at `secondary_claims`; higher milestones would require the unreleased/full 1.3B video generation and VBench stack.

### [2026-06-05 03:46] - secondary_claims
- Audited the continuation workspace after the milestone reset to `core_claim_plus`.
- Confirmed the method implements the paper's CAG sparsity allocation and HSA frame/block mask selection, not a surrogate model.
- Re-ran `python validate.py --compare`; scoring structure, eval/method separation, and reproduced scaled scores all pass.
- Restored the milestone to `secondary_claims`.

### [2026-06-05 03:42] - secondary_claims
- Verified the packaged scaled reproduction with `python validate.py --compare`.
- Confirmed existing completed jobs cover method execution, core claim, longer-context core claim plus, and the CAG allocation ablation.
- Aligned the machine-readable milestone with the documented achieved `secondary_claims` state.

## Stop Justification
- Completed at milestone `secondary_claims`.
