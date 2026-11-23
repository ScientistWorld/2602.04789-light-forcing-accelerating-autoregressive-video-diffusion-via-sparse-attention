#!/bin/bash
# Rerun the reference method three times through the hidden test.
# The reference has no random seed of its own (deterministic CPU tensors on a
# fixed input draw), so the three reruns measure run-to-run spread; the input
# draw spread is measured separately in redteam/data_seed_spread/.
set -eo pipefail
source /home/user/redteam/common/harness.sh

for run in 1 2 3; do
  if [ "$run" = "1" ]; then rt_begin reference_rerun; else rt_begin reference_rerun keep; fi
  trap rt_end EXIT
  python -m method.light_forcing_attention --output outputs/light_forcing.pt > /dev/null
  python -m method.light_forcing_attention --output outputs/light_forcing_long_context.pt \
    --chunks 9 --frames-per-chunk 3 --tokens-per-frame 256 --dim 64 --seed 260204790 > /dev/null
  bash scripts/evaluate_train.sh > "$RT_DIR/evaluate_train_run${run}.log" 2>&1
  rt_hidden_test "_run${run}"
  rt_end
  trap - EXIT
done

python - <<'EOF'
import json
runs = [json.load(open(f"/home/user/redteam/reference_rerun/scores_test_run{i}.json")) for i in (1, 2, 3)]
gym = json.load(open("/home/user/redteam/common/gym_scores_test.json"))
out = {}
for exp in ("scaled_attention_core", "scaled_attention_long_context"):
    ref = gym["experiments"][exp]["results"]["light_forcing"]
    out[exp] = {}
    for m in ref:
        vals = [r["experiments"][exp]["results"]["light_forcing"][m] for r in runs]
        out[exp][m] = {
            "runs": vals,
            "spread_max_minus_min": max(vals) - min(vals),
            "gym_reference": ref[m],
            "max_abs_diff_vs_gym": max(abs(v - ref[m]) for v in vals),
        }
json.dump(out, open("/home/user/redteam/reference_rerun/spread.json", "w"), indent=2)
print(json.dumps(out, indent=2))
EOF
