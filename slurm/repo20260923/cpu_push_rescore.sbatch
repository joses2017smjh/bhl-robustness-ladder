#!/bin/bash
#SBATCH --job-name=push-rescore
#SBATCH --account=eecs
#SBATCH --partition=share
#SBATCH --constraint=haswell&el8
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# Repo-wide campaign (2026-09-23): matched push protocol for the new push seeds,
# then pooled fall rates with one-sided Fisher tests against the controls.
#
# Submit AFTER the training+export jobs, i.e. with
#   --dependency=afterany:<train_push_adaptive_s2 job>:<train_arms_push_s23 array>
# afterany, not afterok, on purpose: with afterok a single failed array task
# (say arms-push-s3's EXPORT check) would hold this job at
# DependencyNeverSatisfied and no verdict would ever print. With afterany the
# job runs once every training task has ended, scores whatever was exported
# (score() skips a run that has no exported/deploy.yaml), and the summary
# reports INCOMPLETE naming the missing CSVs. It is resubmit-safe: complete
# 30-row CSVs are skipped, so after a failed seed is retrained and exported,
# resubmitting this job (no dependency needed) fills in only what is missing.
#
# Protocol (identical to cpu_eval.sbatch's push mode, which produced every
# existing CSV in results/push-matched-20260923/): run_eval, 0.5 m/s shoves
# every 3 s (harness push_interval_s), 12 s episodes, seeds 0-4 x the harness
# command set (30 episodes per policy), terrain 0, MuJoCo 3.3.5 in the plain
# venv on a haswell&el8 share node.
#
# What is scored here:
#   biped    push-adaptive-s2                           (--variant biped)
#   22-DoF   arms-push-s0 s1 s2 s3, arms-dr1.0-s0 s1    (--variant humanoid,
#            the humanoid deploy.yaml, as results/arms did)
# The 22-DoF control (arms-dr1.0) and the existing arms-push seeds are rescored
# on purpose: results/arms/*__p0.5_d0.csv were 10 s episodes (survival_s = 10.0
# in every surviving row), and a Fisher test of a 12 s arm against a 10 s
# baseline would not be matched. "Same run_eval flags as results/arms" is read
# as --variant humanoid + the humanoid deploy.yaml; the episode length follows
# the matched protocol. Six humanoid x ~2 min + one biped fits well inside 1.5 h
# (the existing biped CSVs landed 1-2 min apart).
# dr-default-s0/s1/s2 are NOT rescored: their matched CSVs already exist in
# this directory and are read as the control.
#
# Outputs: results/push-matched-20260923/<label>__p0.5.csv (one per policy,
# skipped if a complete 30-row CSV is already there) and
# results/push-matched-20260923/push_rescore_summary.json.
#
# PREDECLARED READING (written before any new number exists):
#   The push_pair caption ("push-adaptive stays upright where DR-only falls")
#   is SUPPORTED iff the pooled push-adaptive fall rate over THREE training
#   seeds (s0+s1+s2, n = 90) is below dr-default's (s0+s1+s2, n = 90) at
#   one-sided Fisher p < 0.05 (scipy fisher_exact, alternative='less' on
#   [[adaptive falls, adaptive ok], [control falls, control ok]]).
#   Otherwise NOT SUPPORTED (a direction-only result stays direction-only).
#   INCOMPLETE if any of the seven new CSVs is missing or short of 30 rows.
#   Secondary, reported either way: arms-push (4 seeds, n = 120) vs arms-dr1.0
#   (2 seeds, n = 60), both one-sided directions, because the prior n = 60
#   finding was that the push-trained 22-DoF arm fell MORE (9/60 vs 6/60).
# The line 'PUSH-RESCORE RESULT: ...' is computed from the CSVs only.
set -euo pipefail
REPO=/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder
PY=/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python
U=$REPO/external/Berkeley-Humanoid-Lite
cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO/scripts" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
CACHE=${TMPDIR:-/tmp}/bhl-cpu-eval-${SLURM_JOB_ID:-local}; mkdir -p "$CACHE"
"$PY" -c "import mujoco, onnxruntime, scipy; print('preflight OK', mujoco.__version__)"
OUT=results/push-matched-20260923
mkdir -p "$OUT"

score() {  # <biped|humanoid log root> <run name> <variant>
    local exp=$1 pat=$2 variant=$3 d label csv
    d=$(ls -d "$U"/logs/rsl_rl/"$exp"/*_"$pat" 2>/dev/null | sort | tail -1 || true)
    [ -n "$d" ] || { echo "skip (no run): $pat"; return 0; }
    [ -f "$d/exported/deploy.yaml" ] || { echo "skip (not exported): $d"; return 0; }
    label=$(basename "$d" | sed 's/^[0-9-]*_[0-9-]*_//')
    csv=$OUT/${label}__p0.5.csv
    if [ -f "$csv" ] && [ "$(tail -n +2 "$csv" | wc -l)" -eq 30 ]; then echo "have $csv"; return 0; fi
    echo "=== $label  variant=$variant  $(basename "$d")  $(date) ==="
    "$PY" -m bhl_robust.eval.run_eval --deploy-cfg "$d/exported/deploy.yaml" --upstream "$U" --cache-dir "$CACHE" \
        --out "$csv" --label "$label" --variant "$variant" \
        --episode-s 12 --seeds 5 --push-speed 0.5 --terrain-difficulty 0 \
        || echo "EVAL FAILED: $label"
}

score biped    push-adaptive-s2 biped
for r in arms-dr1.0-s0 arms-dr1.0-s1 arms-push-s0 arms-push-s1 arms-push-s2 arms-push-s3; do
    score humanoid "$r" humanoid
done

echo "=== summary $(date) ==="
"$PY" - "$OUT" "$REPO/results/arms" <<'PY'
import csv, json, sys
from pathlib import Path
from scipy.stats import fisher_exact

out = Path(sys.argv[1]); arms_prior = Path(sys.argv[2])

def read(path):
    """(falls, n) of one run_eval CSV; None if absent."""
    if not path.is_file():
        return None
    with path.open() as f:
        rows = list(csv.DictReader(f))
    return sum(r["fell"].strip().lower() == "true" for r in rows), len(rows)

def pool(labels, suffix="__p0.5.csv", root=out):
    per, falls, n, missing = {}, 0, 0, []
    for lab in labels:
        fn = read(root / f"{lab}{suffix}")
        if fn is None or fn[1] != 30:
            missing.append(lab)
        if fn is not None:
            per[lab] = {"falls": fn[0], "n": fn[1], "rate": round(fn[0] / fn[1], 4) if fn[1] else None}
            falls += fn[0]; n += fn[1]
    return {"per_seed": per, "falls": falls, "n": n, "rate": round(falls / n, 4) if n else None, "missing_or_short": missing}

def fisher(a, b, alt):
    if not a["n"] or not b["n"]:
        return None
    return float(fisher_exact([[a["falls"], a["n"] - a["falls"]], [b["falls"], b["n"] - b["falls"]]], alternative=alt)[1])

ctrl   = pool([f"dr-default-s{i}" for i in range(3)])
adapt  = pool([f"push-adaptive-s{i}" for i in range(3)])
arms_c = pool([f"arms-dr1.0-s{i}" for i in range(2)])
arms_p = pool([f"arms-push-s{i}" for i in range(4)])
prior  = {"note": "results/arms CSVs, 10 s episodes, NOT matched; for reference only",
          "arms-dr1.0": pool([f"arms-dr1.0-s{i}" for i in range(2)], "__p0.5_d0.csv", arms_prior),
          "arms-push":  pool([f"arms-push-s{i}" for i in range(2)], "__p0.5_d0.csv", arms_prior)}

p_adapt_less  = fisher(adapt, ctrl, "less")
p_arms_less   = fisher(arms_p, arms_c, "less")
p_arms_more   = fisher(arms_p, arms_c, "greater")

def show(name, g):
    seeds = ", ".join(f"{k.rsplit('-s', 1)[1]}: {v['falls']}/{v['n']}" for k, v in g["per_seed"].items())
    print(f"  {name:<14} {g['falls']:>3}/{g['n']:<3} = {g['rate']}   (seeds {seeds})")

print("Matched push protocol, 0.5 m/s every 3 s, 12 s episodes, 30 episodes per seed")
print("12-DoF biped:"); show("dr-default", ctrl); show("push-adaptive", adapt)
print(f"  one-sided Fisher (push-adaptive < dr-default): p = {p_adapt_less}")
print("22-DoF arms:"); show("arms-dr1.0", arms_c); show("arms-push", arms_p)
print(f"  one-sided Fisher (arms-push < arms-dr1.0): p = {p_arms_less};  (arms-push > arms-dr1.0): p = {p_arms_more}")

required = ["push-adaptive-s2", "arms-dr1.0-s0", "arms-dr1.0-s1"] + [f"arms-push-s{i}" for i in range(4)]
missing = [l for l in required if read(out / f"{l}__p0.5.csv") is None or read(out / f"{l}__p0.5.csv")[1] != 30]
missing += [l for l in ctrl["missing_or_short"] + adapt["missing_or_short"] if l not in missing]

if missing:
    verdict = "INCOMPLETE"
    detail = f"missing or short CSVs: {', '.join(missing)}"
elif p_adapt_less is not None and adapt["rate"] < ctrl["rate"] and p_adapt_less < 0.05:
    verdict = "SUPPORTED"
    detail = (f"push-adaptive {adapt['falls']}/{adapt['n']} ({adapt['rate']}) below dr-default "
              f"{ctrl['falls']}/{ctrl['n']} ({ctrl['rate']}), one-sided Fisher p = {p_adapt_less:.4f} < 0.05")
else:
    verdict = "NOT SUPPORTED"
    detail = (f"push-adaptive {adapt['falls']}/{adapt['n']} ({adapt['rate']}) vs dr-default "
              f"{ctrl['falls']}/{ctrl['n']} ({ctrl['rate']}), one-sided Fisher p = {p_adapt_less:.4f} (needs < 0.05)")

if arms_p["n"] and arms_c["n"]:
    if p_arms_less is not None and p_arms_less < 0.05:
        arms_read = "arms-push falls LESS than arms-dr1.0 (p < 0.05)"
    elif p_arms_more is not None and p_arms_more < 0.05:
        arms_read = "arms-push falls MORE than arms-dr1.0 (p < 0.05)"
    else:
        arms_read = "arms-push and arms-dr1.0 not distinguishable at p < 0.05"
    arms_read += f" [{arms_p['falls']}/{arms_p['n']} vs {arms_c['falls']}/{arms_c['n']}]"
else:
    arms_read = "arms comparison not available"

summary = {
    "protocol": "run_eval, 0.5 m/s shoves every 3 s, 12 s episodes, seeds 0-4 x harness command set, terrain 0; "
                "biped policies --variant biped, 22-DoF policies --variant humanoid; seed = policy training seed",
    "predeclared": "push_pair caption SUPPORTED iff pooled push-adaptive (3 seeds, n=90) < dr-default (3 seeds, n=90) "
                   "at one-sided Fisher p < 0.05; INCOMPLETE if any new CSV missing or short",
    "biped": {"dr-default": ctrl, "push-adaptive": adapt, "fisher_one_sided_adaptive_less": p_adapt_less},
    "arms": {"arms-dr1.0": arms_c, "arms-push": arms_p,
             "fisher_one_sided_push_less": p_arms_less, "fisher_one_sided_push_more": p_arms_more,
             "reading": arms_read},
    "prior_unmatched_10s": prior,
    "verdict": verdict, "detail": detail,
}
(out / "push_rescore_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(f"wrote {out / 'push_rescore_summary.json'}")
print(f"PUSH-RESCORE RESULT: {verdict} -- {detail}; arms: {arms_read}")
PY
echo REPO_CPU_EVAL_COMPLETE
