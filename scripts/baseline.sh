#!/bin/bash
# Run a baseline method from the paper.
#
# Baselines are simple reference methods the paper compares against. 
# Implement this only if applicable, i.e. the paper has baselines.
# You do not need to implement all baselines to begin — 
# just enough to validate that the evaluation pipeline works 
# and provide a reference point. You can decide to expand more baselines later.
# Each baseline must be actually run and scored through evaluate.sh
# so the numbers are verified, not self-reported.
# You can decide what this baseline script should do yourself.
#
# Usage:
#   bash scripts/baseline.sh <name>

set -e

cd /home/user

NAME="${1:-sliding_window}"
ARTIFACT="outputs/${NAME}.pt"

mkdir -p outputs

case "$NAME" in
  sliding_window)
    python -m baseline.sparse_baselines --output "$ARTIFACT"
    bash scripts/evaluate.sh sliding_window "$ARTIFACT"
    ;;
  *)
    echo "unknown baseline: $NAME" >&2
    exit 2
    ;;
esac
