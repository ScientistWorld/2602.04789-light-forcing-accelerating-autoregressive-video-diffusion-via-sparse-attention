#!/bin/bash
# Visible train-slice evaluation entry point.
#
# No-argument mode enumerates method artifacts under outputs/ and writes
# scoring/scores_train.json. Explicit mode mirrors scripts/evaluate.sh:
#   ./evaluate_train.sh <method_and_run_name> <artifact.pt> <experiment> [benchmark args]

set -e

cd /home/user

mkdir -p scoring

if [ "$#" -eq 0 ]; then
  python -m eval.train.evaluate \
    --scores scoring/scores_train.json
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

  python -m eval.train.evaluate \
    --method-name "$METHOD_NAME" \
    --artifact "$ARTIFACT" \
    --experiment "$EXPERIMENT" \
    --scores scoring/scores_train.json \
    "$@"
fi
