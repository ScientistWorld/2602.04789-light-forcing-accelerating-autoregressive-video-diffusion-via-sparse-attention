#!/bin/bash
# TEMPORARY (red-team audit job): restored with git after the job.
set -uo pipefail
cd /home/user
bash redteam/run_all.sh
