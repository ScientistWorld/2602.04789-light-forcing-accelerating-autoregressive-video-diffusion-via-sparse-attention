#!/bin/bash
# Hardening evidence: replays honest better/weaker variants and shortcut
# selectors through the train evaluator and the hidden test (no arguments), then
# restores the official registrations and score files. Run after
# scripts/reproduce.sh has produced the official scores, e.g. as a job:
#   bash scripts/reproduce.sh && bash redteam/hardening/replay.sh
set -e
cd /home/user
H=redteam/hardening
mkdir -p "$H"
cp scoring/scores_train.json "$H/phase1_scores_train.json"
cp scoring/scores_test.json "$H/phase1_scores_test.json"
rm -f "$H/phase2.done"

# ---------------------------------------------------------------- phase 2
if [ ! -f "$H/phase2.done" ]; then
  R="bash scripts/register_method.sh"
  LF="python -m method.light_forcing_selector"
  SC="python redteam/hardening/shortcuts.py"
  # honest variants
  $R light_forcing_per_frame $LF --selection hsa_per_frame
  $R lf_frame_max $LF --frame-score max
  $R lf_topk12 $LF --topk-frames 12
  $R lf_alpha1 $LF --alpha-power 1.0
  $R lf_step_front $LF --step-weights 1.6,1.0,0.8,0.6
  $R lf_explore15 $LF --explore 0.15
  $R lf_sparser $LF --extra-sparsity 0.03
  $R random_tiles_s1 python -m baseline.sparse_baselines --baseline random_tiles --seed 1
  $R random_tiles_s2 python -m baseline.sparse_baselines --baseline random_tiles --seed 2
  # shortcuts
  for mode in dense probe_all negative float_mask extra_outputs crash_long frontload read_hidden write_scores; do
    $R "sc_$mode" $SC "$mode"
  done
  $R sc_raise_budget $LF --extra-sparsity -0.2
  $R sc_phantom python -m does.not.exist
  bash scripts/evaluate_train.sh 2>&1 | grep -v '^  ' | tee "$H/phase2_train.log" || true
  cp scoring/scores_train.json "$H/phase2_scores_train.json"
  # stale-file shortcut: a fabricated scores_test.json must not survive the hidden test
  python - <<'PY'
import json
fake = {"experiments": {e: {"results": {"sc_stale_phantom": {"mean_generation_rel_error": 0.0}}}
        for e in ("ar_rollout_core", "ar_rollout_long_context", "ar_rollout_high_sparsity")}}
open("scoring/scores_test.json", "w").write(json.dumps(fake))
PY
  bash scripts/evaluate_test.sh 2>&1 | grep -v '^  ' | tee "$H/phase2_test.log" || true
  cp scoring/scores_test.json "$H/phase2_scores_test.json"
  python - <<'PY'
import json
d = json.load(open("redteam/hardening/phase2_scores_test.json"))
names = {n for e in d["experiments"].values() for n in e["results"]}
msg = ("fabricated entry sc_stale_phantom ABSENT after the hidden test (file was replaced)"
       if "sc_stale_phantom" not in names else "fabricated entry sc_stale_phantom SURVIVED")
fake = d["experiments"]["ar_rollout_core"]["results"].get("sc_write_scores", {})
msg += f"; sc_write_scores core entry = {fake}"
open("redteam/hardening/stale_check.txt", "w").write(msg + "\n")
print(msg)
PY
  cp outputs/selector_logs/test_sc_read_hidden.log "$H/sc_read_hidden_test.log" 2>/dev/null || true
  # restore the official registrations and scores
  for f in outputs/methods/*.json; do
    case "$(basename "$f" .json)" in
      light_forcing|light_forcing_uniform|light_forcing_cag_only|one_d_sparse|local_window|random_tiles) ;;
      *) rm -f "$f" ;;
    esac
  done
  cp "$H/phase1_scores_train.json" scoring/scores_train.json
  cp "$H/phase1_scores_train.json" scoring/scores.json
  cp "$H/phase1_scores_test.json" scoring/scores_test.json
  touch "$H/phase2.done"
fi

python redteam/hardening/analyze.py || true
python validate.py --compare || true
