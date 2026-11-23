#!/bin/bash
# Job script: reproduces the gym's reference results end to end on CPU
# (registers the paper's method, its two component ablations and three
# baselines; scores them on the train slice and the hidden test; validates).
# The hardening evidence (variants + shortcut selectors) is replayed by
# redteam/hardening/replay.sh after this script.
set -e
cd /home/user
bash scripts/reproduce.sh
