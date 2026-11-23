#!/bin/bash
# Register a baseline mask selector (and score it on the train slice).
#
# Usage: bash scripts/baseline.sh <one_d_sparse|local_window|random_tiles> [--no-eval]
set -e
cd /home/user
NAME="${1:?usage: $0 <one_d_sparse|local_window|random_tiles> [--no-eval]}"
case "$NAME" in
  one_d_sparse|local_window|random_tiles)
    bash scripts/register_method.sh "$NAME" python -m baseline.sparse_baselines --baseline "$NAME"
    ;;
  *)
    echo "unknown baseline: $NAME" >&2
    exit 2
    ;;
esac
if [ "${2:-}" != "--no-eval" ]; then
  bash scripts/evaluate.sh --only "$NAME"
fi
