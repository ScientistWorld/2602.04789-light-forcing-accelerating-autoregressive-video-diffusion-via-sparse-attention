#!/bin/bash
# Evaluation script — the standard way to evaluate work in this environment.
#
# This script must be SELF-CONTAINED using only code in eval/.
# Do NOT import from method/ — evaluate.sh must work even when the
# paper's method code is hidden from a scientist agent.
#
# Design it so that anyone following the same output format can run
# this script to evaluate their results.
#
# OUTPUT CONTRACT: This script MUST write scoring/scores.json containing
# a JSON dict nested by experiment, e.g.:
#   {"imagenet_classification": {"top1_accuracy": 82.1, "top5_accuracy": 94.8}}
# Experiment and metric names must match the keys in scoring/reference.json.
# This file is read by the system for automated comparison.
#
# Example usage:
#   ./evaluate.sh <method_and_run_name> <artifact.pt> <experiment> [benchmark args]
#
# Called from run.sh after training, or independently.

set -e

cd /home/user

mkdir -p scoring

METHOD_NAME="${1:-light_forcing}"
ARTIFACT="${2:-outputs/${METHOD_NAME}.pt}"
EXPERIMENT="${3:-scaled_attention_core}"
if [ "$#" -ge 3 ]; then
  shift 3
elif [ "$#" -ge 2 ]; then
  shift 2
elif [ "$#" -ge 1 ]; then
  shift 1
fi

python -m eval.attention_benchmark \
  --method-name "$METHOD_NAME" \
  --artifact "$ARTIFACT" \
  --experiment "$EXPERIMENT" \
  --scores scoring/scores.json \
  "$@"
