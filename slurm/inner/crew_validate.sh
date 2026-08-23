#!/bin/bash
set -uo pipefail
cd "$UPSTREAM"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
"$PY" "$REPO/scripts/bench/crew_validate.py" 2>&1 | tee "$REPO/results/crew_validate.log"
# The verdict is read from the file, not the exit code: Isaac Sim's shutdown
# hard-exits the interpreter and swallows anything raised after it.
grep -q "ALL PASS" "${GATE_OUT}" || { echo "GATE FAILED"; exit 1; }
echo "GATE PASSED"
