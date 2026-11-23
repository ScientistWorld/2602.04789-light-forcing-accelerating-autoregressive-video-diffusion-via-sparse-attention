#!/bin/bash
# Register a mask selector so that scripts/evaluate_train.sh and
# scripts/evaluate_test.sh run it. A method is a command that serves the
# selector protocol (see data/selector_protocol.py) on stdin/stdout.
#
# Usage: bash scripts/register_method.sh <name> <command...>
#   e.g. bash scripts/register_method.sh my_method python -m my_package.my_selector --flag
set -e
cd /home/user
if [ "$#" -lt 2 ]; then
  echo "usage: $0 <name> <command...>" >&2
  exit 2
fi
mkdir -p outputs/methods
python - "$@" <<'PY'
import json
import re
import sys

name, command = sys.argv[1], sys.argv[2:]
if not re.match(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$", name):
    raise SystemExit(f"invalid method name: {name!r}")
with open(f"outputs/methods/{name}.json", "w") as fh:
    fh.write(json.dumps({"command": command}, indent=1) + "\n")
print(f"registered {name}: {command}")
PY
