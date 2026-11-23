#!/bin/bash
set -eo pipefail
source /home/user/redteam/common/harness.sh
rt_begin shortcut_self_reported_stats
trap rt_end EXIT
python redteam/shortcut_self_reported_stats/make_artifacts.py
bash scripts/evaluate_train.sh > "$RT_DIR/evaluate_train.log" 2>&1
rt_hidden_test
