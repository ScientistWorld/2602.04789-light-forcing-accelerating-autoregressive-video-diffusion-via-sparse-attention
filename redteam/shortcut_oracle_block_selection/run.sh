#!/bin/bash
set -eo pipefail
source /home/user/redteam/common/harness.sh
rt_begin shortcut_oracle_block_selection
trap rt_end EXIT
python redteam/shortcut_oracle_block_selection/make_artifacts.py
bash scripts/evaluate_train.sh > "$RT_DIR/evaluate_train.log" 2>&1
rt_hidden_test
