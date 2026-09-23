#!/bin/bash
#SBATCH --job-name=ice-exposure
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# Repo-wide GPU campaign (2026-09-23): do the TRAINED placed-ice policies ever
# put a foot on a patch? 99u (ice_placement_probe) showed the patches are
# reachable from spawn; this runs the blind-s0 and depth-s0 model_5999.pt
# checkpoints, acting, in their own training tasks (resolved from each run's
# params/env.yaml) for one full 20 s episode on 64 envs, and counts foot-on-ice
# per physics step. v51 -- the stack 92_ice_ppo trained B3 on.
#
# Job PASS = both probes produced a valid measurement (task resolved and
# matching, trained params reproduced, every episode closed, hook saw every
# physics step, policy acting). EXPOSED / NOT_EXPOSED is the scientific result
# and is reported on the same line; it does not decide PASS.
set -euo pipefail
export BHL_STACK=v51
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
source "$REPO/slurm/inner/v60_boot_gate.sh"
V60_BOOT_LOCK="$WORKSPACE/.v51-boot.lock" v60_boot_gate

PROBE_ENVS=${PROBE_ENVS:-64}
CKPT_NAME=model_5999.pt
OUT=$REPO/results/repo-gpu-20260923/ice_exposure
LOGS=$UPSTREAM/logs/rsl_rl/biped
mkdir -p "$OUT"
echo "=== node: $(hostname) | ice exposure probe | stack $BHL_STACK | envs $PROBE_ENVS | $(date) ==="

# Inner script (inside the container): baked here so no new variable has to
# cross --cleanenv. REPO and PY are forwarded by bhl_exec itself. 780 s per
# invocation x at most 4 (two arms, each with a geometric-only retry) = 52 min,
# so even the all-hang path reaches the PASS/FAIL block inside the 1 h limit.
INNER=$OUT/inner_${SLURM_JOB_ID:-local}.sh
cat > "$INNER" <<EOF
#!/bin/bash
set -uo pipefail
cd "\$REPO"
export PYTHONPATH="\$REPO/src:\${PYTHONPATH:-}"
run=\$1; out=\$2; shift 2
timeout 780 "\$PY" scripts/bench/ice_exposure_probe.py --run-dir "\$run" --checkpoint $CKPT_NAME \\
    --num-envs $PROBE_ENVS --out "\$out" "\$@"
echo "inner exit \$?"
EOF

json_ok() {  # status == ok in a probe JSON
    "$PY" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("status")=="ok" else 1)' "$1" 2>/dev/null
}

for arm in blind depth; do
    run=$(ls -d "$LOGS"/2026-09-16_*_ppo-ice-placed-${arm}-s0 2>/dev/null | tail -1 || true)
    if [ -z "$run" ] || [ ! -f "$run/$CKPT_NAME" ]; then
        echo "missing run or $CKPT_NAME for ppo-ice-placed-${arm}-s0 under $LOGS"
        continue
    fi
    json=$OUT/${arm}_s0.json
    rm -f "$json" "$OUT/${arm}_s0.log"
    echo "--- $arm: $run"
    bhl_exec "$INNER" "$run" "$json" 2>&1 | tee "$OUT/${arm}_s0.log" || true
    if ! json_ok "$json"; then
        # The filtered per-foot patch contact sensors are the only addition to
        # the trained scene. If the env did not build with them, measure with
        # the geometric footprint + sole-height rule instead.
        echo "--- $arm: no valid result with patch contact sensors; retrying geometric-only"
        [ -f "$json" ] && mv "$json" "$OUT/${arm}_s0.contact_attempt.json"
        bhl_exec "$INNER" "$run" "$json" --no-ice-contact-sensor 2>&1 | tee -a "$OUT/${arm}_s0.log" || true
    fi
done

# ------------------------------------------------------------- PASS / FAIL
set +e
"$PY" - "$OUT" "$PROBE_ENVS" <<'PY'
import json, sys
from pathlib import Path

out, n_expected = Path(sys.argv[1]), int(sys.argv[2])
expect = {"blind": "Velocity-BHL-Biped-Ice-v0", "depth": "Velocity-BHL-Biped-Ice-Depth-v0"}
problems, verdicts, rows = [], {}, {}
for arm, task in expect.items():
    p = out / f"{arm}_s0.json"
    if not p.is_file():
        problems.append(f"{arm}: no {p.name}")
        continue
    try:
        d = json.loads(p.read_text())
    except Exception as exc:  # noqa: BLE001
        problems.append(f"{arm}: unreadable JSON ({exc!r})")
        continue
    if d.get("status") != "ok":
        problems.append(f"{arm}: status {d.get('status')} {d.get('error', '')}")
        continue
    checks = {
        "task": d.get("task") == task,
        "checkpoint": str(d.get("checkpoint", "")).endswith("/model_5999.pt"),
        "n_episodes": d.get("n_episodes") == n_expected,
        "all_closed": d.get("episodes_closed") == d.get("n_episodes"),
        "params_match": d.get("params_match") is True,
        "hook_ok": d.get("hook_ok") is True,
        "policy_acting": (d.get("policy_acting") or {}).get("mean_abs_action", 0.0) > 1e-3,
        "placement": d.get("patch_placement_max_err_m", 1.0) < 1e-2,
        "verdict": d.get("verdict") in ("EXPOSED", "NOT_EXPOSED"),
        "fractions": all(isinstance(d.get(k), (int, float)) and 0.0 <= d[k] <= 1.0 for k in
                         ("episodes_with_any_ice_contact_fraction", "mean_on_ice_step_fraction")),
    }
    failed = [k for k, ok in checks.items() if not ok]
    if failed:
        problems.append(f"{arm}: failed checks {failed}"
                        + (f" mismatches {d.get('params_mismatches')}" if "params_match" in failed else ""))
    verdicts[arm] = d.get("verdict")
    rows[arm] = d
    m = d.get("methods", {})
    print(f"{arm:5s} {d['task']}  method={d['method']}  n={d['n_episodes']}  "
          f"any_ice={d['episodes_with_any_ice_contact_fraction']:.3f}  "
          f"mean_on_ice={d['mean_on_ice_step_fraction']:.4f}  verdict={d['verdict']}  "
          f"levels={sorted(d.get('by_terrain_level', {}))}  "
          f"sole_off={d.get('sole_offset_m'):.3f}  agree={d.get('contact_vs_geometric_step_agreement')}")
    for name, v in m.items():
        print(f"        {name:9s} any={v['episodes_with_any_ice_contact_fraction']:.3f} "
              f"mean={v['mean_on_ice_step_fraction']:.4f}")
    pa = d.get("policy_acting", {})
    print(f"        acting: |a|={pa.get('mean_abs_action', 0):.3f} vx={pa.get('mean_forward_speed_mps', 0):.3f} "
          f"cmd={pa.get('mean_command_speed_mps', 0):.3f} walked_med={pa.get('median_distance_walked_m', 0):.2f} m  "
          f"spawn->patch med={d.get('median_spawn_dist_to_nearest_patch_m', 0):.2f} m")

summary = {"pass": not problems, "problems": problems, "verdicts": verdicts,
           "files": {a: str(out / f"{a}_s0.json") for a in expect}}
(out / "summary.json").write_text(json.dumps(summary, indent=2))
vs = " ".join(f"{a}={verdicts.get(a, 'NA')}" for a in expect)
if problems:
    print("ICE-EXPOSURE FAIL " + vs + " | " + "; ".join(problems))
    sys.exit(1)
print("ICE-EXPOSURE PASS " + vs)
PY
rc=$?
echo "=== done: $(date) ==="
exit $rc
