# Reproduction Milestones

**Current: secondary_claims**

## Progress Log

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
