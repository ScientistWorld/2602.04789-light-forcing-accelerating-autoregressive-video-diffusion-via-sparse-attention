#!/bin/bash
# Register the paper's method (Light Forcing: Chunk-Aware Growth + Hierarchical
# Sparse Attention) and its two component ablations, then score them on the
# train slice.
#
# Usage: bash scripts/method.sh [--no-eval]
set -e
cd /home/user
bash scripts/register_method.sh light_forcing python -m method.light_forcing_selector
bash scripts/register_method.sh light_forcing_uniform python -m method.light_forcing_selector --allocation uniform
bash scripts/register_method.sh light_forcing_cag_only python -m method.light_forcing_selector --selection global
if [ "${1:-}" != "--no-eval" ]; then
  bash scripts/evaluate.sh --only light_forcing --only light_forcing_uniform --only light_forcing_cag_only
fi
