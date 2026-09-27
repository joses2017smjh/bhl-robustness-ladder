#!/bin/bash
#SBATCH --job-name=sf04-bias-probe
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# SF-04 bias row (docs/SENSOR_FUSION.md SF-04; SLURM_JOBS.md "bias-only"): the
# bias-only fine-tune was trained under a per-episode gyro / gravity bias but
# the probe forced the bias to zero, so its tolerance of bias was never
# measured. scripts/bench/maze_recovery_probe.py now has the settings keys
# gyro_bias / gravity_bias; this job measures both policies with them.
#
#   sbatch slurm/repo20260923/gpu_sf04_bias_probe.sbatch      (no arguments)
#
# Policies (both LEARNED PPO MLP actors, pinned by path, nothing resolved):
#   BothBias  2026-09-24_01-37-12_sf04-ft-bothbias-s0/model_7996.pt on
#             Velocity-BHL-MazeRecovery-Full-BothBias-v0 (the checkpoint of the
#             published results/repo-gpu-20260923/sf04-finetune/BothBias-s0.json;
#             2000 it fine-tuned from the Both teacher under per-episode bias,
#             per component N(0, 0.02^2) on gyro and gravity, job 21404947)
#   control   2026-09-19_16-25-50_wknd-full-both-s0/model_5997.pt on
#             Velocity-BHL-MazeRecovery-Full-Both-v0 (the untouched teacher, the
#             same control as every other SF-04 row)
# Navigation is the ORACLE waypoint teacher's velocity command on a fixed,
# known route (the probe's navigation_source); the policies are the learned
# walking layer. The bias is SCRIPTED by the probe, not simulated physics.
#
# Settings, one Kit boot per policy, all with --keep-corruption (each task's own
# training observation noise on, as in every SF-04 row) --num-envs 32
# --steps 600 --seed 100; the SF-04 per-episode randomization is forced off by
# the probe (BothBias state bias = 0), the bias comes only from the setting:
#   bias0     gyro_bias 0.00, gravity_bias 0.00
#   bias0.02  gyro_bias 0.02, gravity_bias 0.02
#   bias0.05  gyro_bias 0.05, gravity_bias 0.05
#   bias0.10  gyro_bias 0.10, gravity_bias 0.10
# gyro_bias in rad/s, gravity_bias in unit-vector components. Each env gets a
# constant vector of exactly that NORM, direction uniform on the sphere from a
# CPU generator seeded 100 + 11 (gyro and gravity independent), identical for
# both policies and at every level; constant for the setting's whole rollout,
# i.e. over the scored first episode. Scale note: training drew each component
# from N(0, 0.02^2), a typical norm of ~0.032 (0.02*sqrt(3)*0.92); bias 0.02 is
# BELOW that, 0.05 ~1.6x it, 0.10 ~3x it.
#
# Score: first-episode button success out of 32 (round(first_episode_success_rate * 32)).
#
# PREDECLARED RULE (2026-09-26, before any bias number exists):
#   if control bias0.05 >= 28/32:
#       "bias not a threat at this level" -- no PASS/NEGATIVE is scored
#   else PASS iff BothBias bias0.05 >= control bias0.05 + 8
#            AND BothBias bias0 >= 28/32
#        NEGATIVE otherwise
#   INCOMPLETE if a JSON is missing/unreadable, a setting recorded an error or
#   is absent, a setting finished fewer than 32 first episodes, or an identity
#   check fails (task id, checkpoint path, num_envs 32, observation_corruption
#   true, bias magnitudes as requested).
# Rows bias0.02 / bias0.10 are reported, not gated. The control's bias0 is
# printed beside its published baseline (30/32) as a sanity line, not a gate.
# Final line 'SF04-BIAS RESULT: PASS|NEGATIVE|BIAS-NOT-A-THREAT|INCOMPLETE',
# computed from the JSONs, never from exit codes. Exit 0 for the first three
# (all are results), 1 for INCOMPLETE.
#
# Output: results/repo-gpu-20260926/sf04-bias-probe/{BothBias-s0,control-both-s0}.{json,log},
# settings.json and verdict.json. If that directory already holds a probe JSON
# it is left untouched and this job writes to sf04-bias-probe-rerun-<jobid>/.
set -uo pipefail
[ -n "${SLURM_JOB_ID:-}" ] || { echo "SF04-BIAS RESULT: INCOMPLETE (not inside a Slurm allocation; submit with sbatch)"; exit 2; }

export BHL_STACK=v60
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}" OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
unset BHL_POLICY BHL_RND BHL_SYMMETRY   # both policies are MLP actors: never forward the train.py overlays
source "$REPO/slurm/inner/v60_boot_gate.sh"
BOOT_LOCK="${V60_BOOT_LOCK:-/nfs/hpc/share/$USER/Humanoid_Lite/.v60-boot.lock}"
PYH=$WORKSPACE/venv/bin/python; [ -x "$PYH" ] || PYH=$(command -v python3)
LOG_ROOT=$UPSTREAM/logs/rsl_rl/biped
BIAS_CKPT=$LOG_ROOT/2026-09-24_01-37-12_sf04-ft-bothbias-s0/model_7996.pt
CTRL_CKPT=$LOG_ROOT/2026-09-19_16-25-50_wknd-full-both-s0/model_5997.pt
WALL_S=${WALL_S:-10800}   # must match #SBATCH --time
PROBE="$REPO/scripts/bench/maze_recovery_probe.py"
OUT=$REPO/results/repo-gpu-20260926/sf04-bias-probe
if ls "$OUT"/*-s0.json >/dev/null 2>&1; then
    OUT="$OUT-rerun-$SLURM_JOB_ID"; echo "=== earlier results present; writing to $OUT ==="
fi
mkdir -p "$OUT"
SETTINGS='[{"name":"bias0","gyro_bias":0.0,"gravity_bias":0.0},{"name":"bias0.02","gyro_bias":0.02,"gravity_bias":0.02},{"name":"bias0.05","gyro_bias":0.05,"gravity_bias":0.05},{"name":"bias0.10","gyro_bias":0.10,"gravity_bias":0.10}]'
# Through a file: apptainer --env splits values on commas, so the JSON never crosses as a variable.
printf '%s\n' "$SETTINGS" > "$OUT/settings.json"

echo "=== node: $(hostname) | SF-04 bias probe | stack $BHL_STACK | job $SLURM_JOB_ID | $(date) ==="
cd "$REPO" && echo "repo HEAD $(git rev-parse --short HEAD 2>/dev/null) branch $(git branch --show-current 2>/dev/null)"

# Preflight (no Kit): the probe has the bias keys, both checkpoints exist, the settings parse.
grep -q '"gyro_bias", "gravity_bias"' "$PROBE" && grep -q -- "--keep-corruption" "$PROBE" \
    || { echo "SF04-BIAS RESULT: INCOMPLETE (probe lacks the gyro_bias/gravity_bias keys or --keep-corruption)"; exit 1; }
for f in "$BIAS_CKPT" "$CTRL_CKPT"; do
    [ -f "$f" ] || { echo "SF04-BIAS RESULT: INCOMPLETE (checkpoint missing: $f)"; exit 1; }
done
"$PYH" -c 'import json,sys; s=json.load(open(sys.argv[1])); assert [x["name"] for x in s]==["bias0","bias0.02","bias0.05","bias0.10"]' "$OUT/settings.json" \
    || { echo "SF04-BIAS RESULT: INCOMPLETE (settings.json does not parse)"; exit 1; }

# One probe per bhl_exec (one Kit boot each), behind v60_boot_gate. The inner
# script is inline because bhl_exec ends in `bash "$@"`; its inputs are env
# vars forwarded through BHL_FORWARD_VARS (paths and numbers only).
export BHL_FORWARD_VARS="$BHL_FORWARD_VARS PROBE_TASK PROBE_CKPT PROBE_JSON PROBE_LOG PROBE_TMO PROBE_KEEP PROBE_SETTINGS_FILE"
INNER_PROBE='set -uo pipefail
cd "$REPO"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
extra=()
[ "${PROBE_KEEP:-0}" = 1 ] && extra+=(--keep-corruption)
[ -n "${PROBE_SETTINGS_FILE:-}" ] && extra+=(--settings "$(cat "$PROBE_SETTINGS_FILE")")
echo "=== probe $PROBE_TASK <- $PROBE_CKPT | timeout ${PROBE_TMO}s | $(date) ==="
timeout -k 60 "$PROBE_TMO" "$PY" "$REPO/scripts/bench/maze_recovery_probe.py" --headless --enable_cameras \
    --task "$PROBE_TASK" --num-envs 32 --steps 600 --seed 100 --checkpoint "$PROBE_CKPT" \
    --output "$PROBE_JSON" ${extra[@]+"${extra[@]}"} > "$PROBE_LOG" 2>&1
rc=$?
grep -E "^\{\"name\"|SETTING-ERROR|ROLLOUT-ERROR|PROBE-ERROR|^(ValueError|RuntimeError|TypeError|KeyError|AttributeError)" "$PROBE_LOG" | cut -c1-240 || true
echo "=== probe exit $rc (informational) | json $([ -s "$PROBE_JSON" ] && echo written || echo MISSING) ==="'

boot() { v60_boot_gate; GATE_PID=${!:-}; }
release() { kill "${GATE_PID:-}" 2>/dev/null || true; _v60_release "$BOOT_LOCK"; }
run_probe() {   # label task checkpoint
    local label=$1 left
    left=$(( WALL_S - SECONDS - 300 ))
    if [ "$left" -lt 420 ]; then echo "=== $label: only ${left}s of wall left; skipped ==="; return; fi
    boot
    left=$(( WALL_S - SECONDS - 300 )); [ "$left" -gt 3600 ] && left=3600; [ "$left" -lt 60 ] && left=60
    rm -f "$OUT/$label.json"   # the verdict may only come from this job's rollout
    PROBE_TASK=$2 PROBE_CKPT=$3 PROBE_JSON="$OUT/$label.json" PROBE_LOG="$OUT/$label.log" PROBE_TMO=$left \
        PROBE_KEEP=1 PROBE_SETTINGS_FILE="$OUT/settings.json" bhl_exec -c "$INNER_PROBE"
    echo "=== $label done | t=${SECONDS}s ==="; release
}

run_probe BothBias-s0      Velocity-BHL-MazeRecovery-Full-BothBias-v0 "$BIAS_CKPT"
run_probe control-both-s0  Velocity-BHL-MazeRecovery-Full-Both-v0     "$CTRL_CKPT"

# Table and verdict from the JSONs, never from exit codes.
"$PYH" - "$OUT" "$BIAS_CKPT" "$CTRL_CKPT" <<'PY'
import json, os, sys
out, bias_ckpt, ctrl_ckpt = sys.argv[1:]
names = ["bias0", "bias0.02", "bias0.05", "bias0.10"]
levels = dict(zip(names, (0.0, 0.02, 0.05, 0.10)))
problems = []

def load(label, fname, task, ckpt):
    path = os.path.join(out, fname)
    if not os.path.exists(path):
        problems.append(f"{label}: missing {fname}"); return None
    try:
        d = json.load(open(path))
    except Exception as exc:
        problems.append(f"{label}: unreadable {fname}: {exc}"); return None
    if d.get("num_envs") != 32:
        problems.append(f"{label}: num_envs {d.get('num_envs')} != 32")
    if d.get("task") != task:
        problems.append(f"{label}: task {d.get('task')!r} != {task}")
    if d.get("checkpoint") != ckpt:
        problems.append(f"{label}: checkpoint {d.get('checkpoint')!r} != {ckpt}")
    if d.get("observation_corruption") is not True:
        problems.append(f"{label}: observation_corruption is not True (--keep-corruption missing)")
    rows, first = {}, {}
    for s in d.get("settings") or []:
        nm = s.get("name")
        if "error" in s:
            problems.append(f"{label}: setting {nm} recorded an error"); continue
        if nm in levels:
            want = levels[nm]
            if s.get("gyro_bias") != want or s.get("gravity_bias") != want:
                problems.append(f"{label}: {nm} applied gyro/gravity bias {s.get('gyro_bias')}/{s.get('gravity_bias')} != {want}")
            for k in ("gyro_bias_norm_mean", "gravity_bias_norm_mean"):
                if abs(float(s.get(k, -1.0)) - want) > 1e-5:
                    problems.append(f"{label}: {nm} {k} {s.get(k)} != {want}")
        rows[nm] = round(float(s["first_episode_success_rate"]) * 32)
        first[nm] = s.get("first_episodes_completed")
        if nm in levels and first[nm] != 32:
            problems.append(f"{label}: {nm} finished {first[nm]}/32 first episodes in 600 steps (count would be short)")
    for nm in names:
        if nm not in rows:
            problems.append(f"{label}: setting {nm} absent"); rows[nm] = None
    return {"counts": rows, "first_completed": first, "checkpoint": d.get("checkpoint")}

ft = load("BothBias", "BothBias-s0.json", "Velocity-BHL-MazeRecovery-Full-BothBias-v0", bias_ckpt)
ctrl = load("control", "control-both-s0.json", "Velocity-BHL-MazeRecovery-Full-Both-v0", ctrl_ckpt)

def cell(v):
    return "--" if v is None else f"{v}/32"
print("\nfirst-episode button success, 32 envs, seed 100, training obs noise on; bias scripted by the probe;")
print("walking policy LEARNED (PPO MLP), route command from the ORACLE waypoint teacher")
print(f"{'policy':26}" + "".join(f"{nm:>10}" for nm in names))
for label, d in (("BothBias fine-tune", ft), ("control (Both teacher)", ctrl)):
    if d is None:
        print(f"{label:26}" + "".join(f"{'--':>10}" for _ in names) + "  (no JSON)"); continue
    print(f"{label:26}" + "".join(f"{cell(d['counts'][nm]):>10}" for nm in names))
    print(f"{'  first episodes done':26}" + "".join(f"{str(d['first_completed'].get(nm, '--')):>10}" for nm in names))
if ctrl and ctrl["counts"]["bias0"] is not None:
    print(f"sanity (not a gate): control bias0 {ctrl['counts']['bias0']}/32 vs published control baseline 30/32")
for p in problems:
    print("PROBLEM:", p)

verdict, checks = "INCOMPLETE", []
if ft and ctrl and not problems:
    f, c = ft["counts"], ctrl["counts"]
    if c["bias0.05"] >= 28:
        checks.append((f"control bias0.05 {c['bias0.05']}/32 >= 28/32 -> bias not a threat at this level", True))
        verdict = "BIAS-NOT-A-THREAT"
    else:
        checks.append((f"BothBias bias0.05 {f['bias0.05']} vs control {c['bias0.05']} (>= +8)", f["bias0.05"] - c["bias0.05"] >= 8))
        checks.append((f"BothBias bias0 {f['bias0']} >= 28/32", f["bias0"] >= 28))
        verdict = "PASS" if all(ok for _, ok in checks) else "NEGATIVE"
    for text, ok in checks:
        print(f"check: {text}: {'ok' if ok else 'FAILED'}")
json.dump({"verdict": verdict, "checks": [{"text": t, "ok": ok} for t, ok in checks], "problems": problems,
           "BothBias": ft, "control": ctrl,
           "rule": "if control bias0.05 >= 28/32: bias not a threat at this level; else PASS iff BothBias bias0.05 >= "
                   "control bias0.05 + 8 AND BothBias bias0 >= 28/32 (predeclared 2026-09-26)",
           "labels": {"policy": "learned PPO MLP walking policy", "navigation": "oracle waypoint teacher command",
                      "bias": "scripted, policy-side constant vector of the given norm (see settings imu_bias_model)"}},
          open(os.path.join(out, "verdict.json"), "w"), indent=2)
print(f"\nSF04-BIAS RESULT: {verdict}")
sys.exit(0 if verdict != "INCOMPLETE" else 1)
PY
rc=$?
echo "=== done: t=${SECONDS}s | $(date) ==="
exit $rc
