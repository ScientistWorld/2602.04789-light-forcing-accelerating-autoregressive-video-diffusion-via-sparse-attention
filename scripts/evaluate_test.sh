#!/bin/bash
# Hidden test-slice evaluation. Takes no arguments.
#
# Runs every method registered under outputs/methods/<name>.json through the
# held-out rollouts of eval/test/ and writes scoring/scores_test.json. Any
# previous scoring/scores_test.json is deleted first, so a failed or partial
# run can never leave an older file behind.
set -e
cd /home/user
mkdir -p scoring
chmod -R u+rwX eval/test 2>/dev/null || true
rm -f scoring/scores_test.json scoring/scores_test.json.tmp
python -B -m eval.test.evaluate --methods-dir outputs/methods --scores scoring/scores_test.json
