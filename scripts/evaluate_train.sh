#!/bin/bash
# Visible train-slice evaluation.
#
# Runs every method registered under outputs/methods/<name>.json through the
# closed-loop rollouts of eval/train/ and writes scoring/scores_train.json.
# Optional development filters are passed through, e.g.
#   bash scripts/evaluate_train.sh --only my_method --max-instances 1
set -e
cd /home/user
mkdir -p scoring
python -m eval.train.evaluate --methods-dir outputs/methods --scores scoring/scores_train.json "$@"
