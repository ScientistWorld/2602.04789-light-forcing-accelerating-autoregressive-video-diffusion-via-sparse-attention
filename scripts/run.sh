#!/bin/bash
# Calibration job for the hardened closed-loop benchmark (temporary; restored to
# the end-to-end reproduction after calibration).
set -e
cd /home/user
python -c "import torch, numpy, os; print('torch', torch.__version__, 'numpy', numpy.__version__, 'cpus', os.cpu_count())"
mkdir -p tmp/calibration
echo "=== protocol smoke test (train evaluator, 1 core instance) ==="
python -m eval.train.evaluate --methods-dir tmp/protocol_smoke/methods \
  --scores tmp/protocol_smoke/scores.json --log-dir tmp/protocol_smoke/logs \
  --experiments ar_rollout_core --max-instances 1 2>&1 | tail -40
python - <<'PY'
import json
d = json.load(open("/home/user/tmp/protocol_smoke/scores.json"))
print(json.dumps(d["_meta"], indent=1))
for n, r in d["experiments"]["ar_rollout_core"]["results"].items():
    print(n, {k: round(v, 4) for k, v in r.items()})
PY
echo "=== calibration ==="
python tmp/calibrate.py A_core A_long A_hs B_core C_core 2>&1 | tee tmp/calibration/calibration.log
