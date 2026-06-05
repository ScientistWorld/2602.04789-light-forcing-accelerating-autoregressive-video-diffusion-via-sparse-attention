#!/bin/bash
# Reproduce the paper's results end-to-end.
#
# This script runs inside the compute container and should call
# scripts/baseline.sh and scripts/method.sh to run baselines and
# the paper's method, each of which calls scripts/evaluate.sh.
#
# Assumes:
#   - Container is built (environment/container.sif)
#   - Data and models are downloaded (via scripts/download.sh)
#   - Working directory is /home/user
#
# This is the "gold standard" reproduction — 
# You do not need to run this, but keep this in good shape
# so other agents can easily run a full loop of reproduction and get the paper's numbers.

set -e

cd /home/user

rm -f scoring/scores.json
bash scripts/baseline.sh sliding_window
bash scripts/baseline.sh one_d_sparse
bash scripts/method.sh
python -m method.light_forcing_attention \
  --allocation uniform \
  --output outputs/light_forcing_uniform.pt
bash scripts/evaluate.sh \
  light_forcing_uniform \
  outputs/light_forcing_uniform.pt \
  scaled_attention_core

LONG_CONTEXT_ARGS=(
  --chunks 9
  --frames-per-chunk 3
  --tokens-per-frame 256
  --dim 64
  --seed 260204790
)

python -m baseline.sparse_baselines \
  --baseline sliding_window \
  --output outputs/sliding_window_long_context.pt \
  "${LONG_CONTEXT_ARGS[@]}"
bash scripts/evaluate.sh \
  sliding_window \
  outputs/sliding_window_long_context.pt \
  scaled_attention_long_context \
  "${LONG_CONTEXT_ARGS[@]}"

python -m baseline.sparse_baselines \
  --baseline one_d_sparse \
  --output outputs/one_d_sparse_long_context.pt \
  "${LONG_CONTEXT_ARGS[@]}"
bash scripts/evaluate.sh \
  one_d_sparse \
  outputs/one_d_sparse_long_context.pt \
  scaled_attention_long_context \
  "${LONG_CONTEXT_ARGS[@]}"

python -m method.light_forcing_attention \
  --output outputs/light_forcing_long_context.pt \
  "${LONG_CONTEXT_ARGS[@]}"
bash scripts/evaluate.sh \
  light_forcing \
  outputs/light_forcing_long_context.pt \
  scaled_attention_long_context \
  "${LONG_CONTEXT_ARGS[@]}"

python -m method.light_forcing_attention \
  --allocation uniform \
  --output outputs/light_forcing_uniform_long_context.pt \
  "${LONG_CONTEXT_ARGS[@]}"
bash scripts/evaluate.sh \
  light_forcing_uniform \
  outputs/light_forcing_uniform_long_context.pt \
  scaled_attention_long_context \
  "${LONG_CONTEXT_ARGS[@]}"

python validate.py --compare
