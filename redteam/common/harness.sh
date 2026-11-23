#!/bin/bash
# Shared harness for every red-team run.
#
# rt_begin <name>   : snapshot the workspace state a run can touch, then start
#                     from what a future agent starts with (empty outputs/, no
#                     scores_train.json / scores_test.json).
# rt_hidden_test [tag]
#                   : do what the gym does when the agent signals done: put the
#                     gym's own eval/ and scripts/evaluate_test.sh back, run
#                     `bash scripts/evaluate_test.sh` with no arguments, and
#                     keep the resulting scores (and exit code) in the run folder.
# rt_end            : restore the snapshot (idempotent; also used as EXIT trap).

RT_ROOT=/home/user
RT_WORK="$RT_ROOT/tmp/redteam_work"

rt_begin() {
  RT_NAME="$1"
  RT_DIR="$RT_ROOT/redteam/$RT_NAME"
  RT_BK="$RT_WORK/backup_$RT_NAME"
  RT_ENDED=0
  cd "$RT_ROOT"
  mkdir -p "$RT_WORK"
  rm -rf "$RT_BK"
  mkdir -p "$RT_BK"

  # Pristine copies of what the gym restores before the hidden test.
  if [ -d "$RT_WORK/pristine/eval" ]; then
    RT_PRISTINE="$RT_WORK/pristine"
  else
    mkdir -p "$RT_BK/pristine"
    cp -r eval "$RT_BK/pristine/eval"
    cp scripts/evaluate_test.sh "$RT_BK/pristine/evaluate_test.sh"
    RT_PRISTINE="$RT_BK/pristine"
  fi

  # Snapshot everything a run may modify.
  if [ -d outputs ]; then mv outputs "$RT_BK/outputs"; fi
  mkdir -p outputs
  for f in scores_train.json scores_test.json scores.json; do
    if [ -f "scoring/$f" ]; then cp "scoring/$f" "$RT_BK/$f"; fi
  done
  cp -r data "$RT_BK/data"
  cp -r eval "$RT_BK/eval"
  cp scripts/evaluate_test.sh "$RT_BK/evaluate_test.sh"

  # A future agent starts without train/test score files.
  rm -f scoring/scores_train.json scoring/scores_test.json

  export PYTHONPATH="$RT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
  export PYTHONDONTWRITEBYTECODE=1
  mkdir -p "$RT_DIR"
  if [ "${2:-}" != "keep" ]; then
    rm -f "$RT_DIR"/scores_test*.json "$RT_DIR"/scores_train*.json \
          "$RT_DIR"/hidden_test*.log "$RT_DIR"/hidden_test*.exitcode
  fi
}

rt_hidden_test() {
  local tag="${1:-}"
  local rc=0
  cd "$RT_ROOT"
  # The gym puts its own eval/ and scripts/evaluate_test.sh back; every other
  # file stays as the agent left it.
  rm -rf eval
  cp -r "$RT_PRISTINE/eval" eval
  cp "$RT_PRISTINE/evaluate_test.sh" scripts/evaluate_test.sh
  bash scripts/evaluate_test.sh > "$RT_DIR/hidden_test${tag}.log" 2>&1 || rc=$?
  echo "$rc" > "$RT_DIR/hidden_test${tag}.exitcode"
  echo "[rt] hidden test${tag} exit code: $rc"
  if [ -f scoring/scores_test.json ]; then
    cp scoring/scores_test.json "$RT_DIR/scores_test${tag}.json"
  else
    echo '{"experiments": {}}' > "$RT_DIR/scores_test${tag}.json"
  fi
  if [ -f scoring/scores_train.json ]; then
    cp scoring/scores_train.json "$RT_DIR/scores_train${tag}.json"
  fi
}

rt_end() {
  if [ "${RT_ENDED:-1}" = "1" ]; then return 0; fi
  RT_ENDED=1
  cd "$RT_ROOT"
  rm -rf outputs
  if [ -d "$RT_BK/outputs" ]; then mv "$RT_BK/outputs" outputs; fi
  rm -f scoring/scores_train.json scoring/scores_test.json scoring/scores.json
  for f in scores_train.json scores_test.json scores.json; do
    if [ -f "$RT_BK/$f" ]; then cp "$RT_BK/$f" "scoring/$f"; fi
  done
  rm -rf data eval
  cp -r "$RT_BK/data" data
  cp -r "$RT_BK/eval" eval
  cp "$RT_BK/evaluate_test.sh" scripts/evaluate_test.sh
  find data eval -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
  rm -rf "$RT_BK"
  echo "[rt] restored workspace after $RT_NAME"
}
