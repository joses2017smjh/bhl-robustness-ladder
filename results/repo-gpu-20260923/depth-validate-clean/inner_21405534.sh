#!/bin/bash
set -uo pipefail
mode=$1; wt=$2; shift 2
export PYTHONPATH="$wt/src"
cd "$wt/external/Berkeley-Humanoid-Lite" || exit 3
case "$mode" in
    check)
        "$PY" - "$wt" <<'PY'
import importlib.util, sys
wt = sys.argv[1]
for name in ("bhl_robust", "berkeley_humanoid_lite"):
    try:
        spec = importlib.util.find_spec(name)
        origin = getattr(spec, "origin", None) or ""
    except Exception as exc:  # noqa: BLE001 -- the editable finder may not give an origin
        spec, origin = None, f"<find_spec raised {exc!r}>"
    # berkeley_humanoid_lite is informational only: its editable finder maps it
    # to the main checkout's submodule (same content the symlink points at).
    print(f"IMPORT_ORIGIN | {name} = {origin or '(editable finder, no origin)'}", flush=True)
    if name == "bhl_robust" and not origin.startswith(wt + "/src/"):
        print(f"IMPORT_ORIGIN | FAIL: bhl_robust does not resolve from {wt}/src", flush=True)
        sys.exit(2)
print("IMPORT_ORIGIN | ok", flush=True)
PY
        ;;
    run)
        limit=$1; shift
        timeout -k 30 "$limit" "$PY" "$wt/scripts/bench/depth_validate.py" "$@"
        ;;
    *) echo "inner: unknown mode $mode" >&2; exit 3 ;;
esac
