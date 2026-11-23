#!/bin/bash
# Standard evaluation entry point: scores every registered method
# (outputs/methods/<name>.json) on the visible train slice and writes
# scoring/scores.json. It never imports method/ code: methods run as
# subprocesses that only choose attention tile masks; the evaluator computes
# every output and metric itself. Development filters are passed through:
#   bash scripts/evaluate.sh --only my_method --max-instances 1
set -e
cd /home/user
mkdir -p scoring
python -m eval.train.evaluate --methods-dir outputs/methods --scores scoring/scores.json "$@"
