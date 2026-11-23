#!/bin/bash
# Reproduce the gym's reference results end to end (CPU, ~1 h on one core).
#
# Registers the paper's method, its two component ablations and the three
# baselines, scores them on the visible train slice (scoring/scores_train.json,
# copied to scoring/scores.json) and on the hidden test slice
# (scoring/scores_test.json), then validates the workspace.
set -e
cd /home/user
rm -rf outputs/methods
bash scripts/method.sh --no-eval
for b in one_d_sparse local_window random_tiles; do
  bash scripts/baseline.sh "$b" --no-eval
done
bash scripts/evaluate_train.sh
cp scoring/scores_train.json scoring/scores.json
bash scripts/evaluate_test.sh
python validate.py --compare
