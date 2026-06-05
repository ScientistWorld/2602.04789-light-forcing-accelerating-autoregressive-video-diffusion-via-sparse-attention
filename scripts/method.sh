#!/bin/bash
# Run the paper's proposed method.
#
# This script trains/runs the paper's method using code in method/.
#
# Usage:
#   bash scripts/method.sh

set -e

cd /home/user

mkdir -p outputs

python -m method.light_forcing_attention --output outputs/light_forcing.pt
bash scripts/evaluate.sh light_forcing outputs/light_forcing.pt
