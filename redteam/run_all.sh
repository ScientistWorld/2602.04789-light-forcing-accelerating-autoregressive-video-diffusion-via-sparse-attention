#!/bin/bash
# Run every red-team folder in sequence (one CPU job), then the rank summary.
set -uo pipefail
cd /home/user
WORK=/home/user/tmp/redteam_work
mkdir -p "$WORK"

# Pristine copies of what the gym restores before the hidden test.
rm -rf "$WORK/pristine"
mkdir -p "$WORK/pristine"
cp -r eval "$WORK/pristine/eval"
cp scripts/evaluate_test.sh "$WORK/pristine/evaluate_test.sh"
# Safety copy of the gym's own outputs/ (each run moves it aside and back).
rm -rf "$WORK/outputs_safety"
cp -r outputs "$WORK/outputs_safety"

python -c "import torch, sys; print('python', sys.version.split()[0], 'torch', torch.__version__, 'threads', torch.get_num_threads())"

FOLDERS=(
  reference_rerun
  variants
  shortcut_dense_outputs_fake_masks
  shortcut_self_reported_stats
  shortcut_impossible_active_fraction
  shortcut_nan_outputs
  shortcut_degenerate_outputs
  shortcut_edit_data_generator
  shortcut_drop_long_context
  shortcut_stale_scores_test
  shortcut_raise_budget
  shortcut_dense_attention
  shortcut_oracle_block_selection
  data_seed_spread
)
if [ "$#" -gt 0 ]; then FOLDERS=("$@"); fi
for d in "${FOLDERS[@]}"; do
  echo "=== $d start $(date +%T)"
  bash "redteam/$d/run.sh" > "redteam/$d/run.log" 2>&1
  echo "=== $d rc=$? end $(date +%T)"
  tail -n 4 "redteam/$d/run.log"
done

python redteam/common/rank.py > redteam/rank.log 2>&1 || echo "rank.py failed"
cat redteam/rank.log

# Integrity checks: the workspace must be back to its starting state.
echo "=== integrity"
diff -rq outputs "$WORK/outputs_safety" && echo "outputs/ restored"
diff -rq --exclude=__pycache__ eval "$WORK/pristine/eval" && echo "eval/ unchanged"
cmp scripts/evaluate_test.sh "$WORK/pristine/evaluate_test.sh" && echo "evaluate_test.sh unchanged"
cmp scoring/scores_test.json redteam/common/gym_scores_test.json && echo "scoring/scores_test.json unchanged"
grep -c "torch.tanh(x @ v_proj)" data/attention_inputs.py && echo "data/ generator restored"
ls "$WORK"
