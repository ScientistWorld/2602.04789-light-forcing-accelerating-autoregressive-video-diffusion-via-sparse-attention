#!/bin/bash
# Shortcut: same reference method, moved to a cheaper-to-score operating point.
# Only the target sparsity changes (0.90 -> 0.80 / 0.50); outputs and masks are
# honest. This tests whether buying accuracy with more tiles outranks the
# reference, i.e. whether the efficiency constraint holds.
set -eo pipefail
source /home/user/redteam/common/harness.sh
rt_begin shortcut_raise_budget
trap rt_end EXIT

LONG=(--chunks 9 --frames-per-chunk 3 --tokens-per-frame 256 --dim 64 --seed 260204790)
for s in 80 50; do
  python -m method.light_forcing_attention --s-target "0.$s" --output "outputs/lf_budget${s}.pt" > /dev/null
  python -m method.light_forcing_attention --s-target "0.$s" --output "outputs/lf_budget${s}_long_context.pt" "${LONG[@]}" > /dev/null
done
bash scripts/evaluate_train.sh > "$RT_DIR/evaluate_train.log" 2>&1
rt_hidden_test
