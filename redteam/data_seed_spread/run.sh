#!/bin/bash
# Input-draw spread of the test score (analysis; uses the gym's own slice
# scorers with an explicit generator seed, not the no-argument hidden test).
set -eo pipefail
cd /home/user
export PYTHONPATH=/home/user${PYTHONPATH:+:$PYTHONPATH}
export PYTHONDONTWRITEBYTECODE=1
python redteam/data_seed_spread/analyze.py
