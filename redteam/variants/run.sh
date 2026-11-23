#!/bin/bash
# Better and weaker honest variants of the reference, scored by the hidden test.
set -eo pipefail
source /home/user/redteam/common/harness.sh
rt_begin variants
trap rt_end EXIT

python redteam/variants/make_artifacts.py \
  light_forcing lf_reimpl_mean \
  lf_better_lse_scoring lf_better_sampled_rows \
  lf_weaker_random_blocks lf_weaker_budget95 lf_ablation_uniform

bash scripts/evaluate_train.sh > "$RT_DIR/evaluate_train.log" 2>&1
rt_hidden_test
