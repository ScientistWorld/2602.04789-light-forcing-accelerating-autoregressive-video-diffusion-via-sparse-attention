#!/bin/bash
# Held-out test-slice evaluation entry point.
#
# No-argument mode reads scoring/scores_train.json, scores the same
# experiment/method entries on eval/test/, and writes scoring/scores_test.json.
# Explicit mode mirrors scripts/evaluate.sh:
#   ./evaluate_test.sh <method_and_run_name> <artifact.pt> <experiment> [benchmark args]

set -e

cd /home/user

mkdir -p scoring

if [ "$#" -eq 0 ]; then
  python -m eval.test.evaluate \
    --train-scores scoring/scores_train.json \
    --scores scoring/scores_test.json
else
  if [ "$#" -lt 3 ]; then
    echo "usage: $0 <method_and_run_name> <artifact.pt> <experiment> [benchmark args]" >&2
    echo "or:    $0" >&2
    exit 2
  fi
  METHOD_NAME="$1"
  ARTIFACT="$2"
  EXPERIMENT="$3"
  shift 3

  python -m eval.test.evaluate \
    --method-name "$METHOD_NAME" \
    --artifact "$ARTIFACT" \
    --experiment "$EXPERIMENT" \
    --scores scoring/scores_test.json \
    "$@"
fi
