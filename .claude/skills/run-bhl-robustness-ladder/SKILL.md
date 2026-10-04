---
name: run-bhl-robustness-ladder
description: Run, drive, test or screenshot the bhl-robustness-ladder humanoid sim on the OSU HPC cluster. Drives a frozen learned gait policy (ONNX) in MuJoCo through stand/walk/turn commands on a Slurm compute node, reports yaw/displacement/falls, renders a screenshot sheet, checks a rollout reproduces a scored run, and runs pytest or any repo script on a compute node. Use when asked to run, start, test, render, screenshot or try out the robot, a policy, or this repo.
---

# Run bhl-robustness-ladder

Robustness experiments for the Berkeley Humanoid Lite: policies are trained in Isaac Lab (Slurm GPU jobs) and
evaluated in MuJoCo. There is no app window. The agent path is **`run.sh` + `driver.py`** in this skill directory.
They put a frozen policy on a compute node, drive it through a command plan, print what it did and optionally write
a screenshot sheet. All paths below are relative to the repo root
(`/nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder`). `run.sh` resolves its own paths, so any cwd works.

## Prerequisites (already on this cluster, nothing to install)

- venv outside the repo: `/nfs/hpc/share/$USER/Humanoid_Lite/venv` (Python 3.11, MuJoCo 3.3.5). Isaac Lab 3.0 jobs use
  `venv-isaac60` via `BHL_STACK=v60`; the launchers set that themselves.
- `slurm/_env.sh` defines `slurm_clean`, which strips the desktop allocation's `SLURM_*` variables. Without it,
  `srun`/`sbatch` from an Open OnDemand desktop try to run as a step of the desktop job.
- Policies: `external/Berkeley-Humanoid-Lite/logs/rsl_rl/{humanoid,biped}/<timestamp>_<run name>/exported/deploy.yaml`.

## Run (agent path): drive a policy

```bash
S=.claude/skills/run-bhl-robustness-ladder
# GPU node (EGL), with a screenshot sheet; reproduces the scored turn-test run turn+_s0 of clock-s2 exactly
$S/run.sh gpu rollout --policy arms-turngait-clock-s2 --plan stand:3,turn:0.6:6 --seed 0 \
    --expect-yaw 196.8 --png /nfs/hpc/share/$USER/Humanoid_Lite/logs/run-skill-20261004/clocks2-turn-s0.png \
    --json /nfs/hpc/share/$USER/Humanoid_Lite/logs/run-skill-20261004/clocks2-turn-s0.json
# CPU node, no screenshot (faster to get): the shipped arms gait walking 6 s
$S/run.sh cpu rollout --policy arms-dr1.0-s0 --plan stand:1,walk:0.35:6 --seed 0
# what can be driven (prints "<run name>  (<run dir>)"; --policy takes the run name)
$S/run.sh cpu policies
$S/run.sh cpu policies --variant biped
```

- **Plan** segments, comma-separated, seconds last: `stand:T`, `walk:VX:T`, `turn:WZ:T`, `cmd:VX:VY:WZ:T`
  (m/s, rad/s). The command at time t is the first segment that has not ended (the turn test's own loop).
- **Output**: one JSON line per segment (`yaw_in_segment_deg`, `yaw_since_reset_deg`, `displacement_*_m`,
  `fell_at_s`), then `DRIVER_PNG <path>` and `DRIVER_OK` / `DRIVER_FELL` / `DRIVER_MISMATCH`. Exit code 0 = ok,
  1 = fell or `--expect-yaw` missed, 2 = bad arguments.
- **Screenshot**: `--png` writes a sheet of frames at reset and at each segment's end, captioned with time and yaw.
  The camera faces the robot's start heading from 30° off-axis. Open the PNG with the Read tool and look at it.
- **Variant**: `--variant humanoid` (default, 22 DoF with arms) or `biped` (legs only); it must match the policy.
- **Same name, several runs** (repeated smokes, e.g. `arms-turngait-clock-s0-smoke`): `--policy` takes the newest.
  Use `--deploy <run dir>/exported/deploy.yaml` for a specific one.
- One rollout takes about 30 s including the queue when the cluster is quiet. Plain `srun` blocks until done.
- Labels: the driver runs a LEARNED gait (frozen ONNX policy) in MuJoCo. It is agent tooling, never a scored gate.

## Run: tests and repo scripts on a compute node

```bash
S=.claude/skills/run-bhl-robustness-ladder
$S/run.sh test tests/test_turn_clip.py            # prints the log path, pytest's summary and PYTEST_EXIT=<code>
$S/run.sh py scripts/bench/turn_test.py --help    # any repo script, PYTHONPATH=src, cwd = repo root
```

`run.sh test` writes pytest's output to `/nfs/hpc/share/$USER/Humanoid_Lite/logs/bhl-test-<time>.txt` (or
`$BHL_LOG`) and exits with pytest's own status. The full suite (`run.sh test tests`, about 1,600 tests) takes a
few minutes; set `BHL_TIME=00:45:00` if it needs longer.

## Run: the project's Slurm launchers (training, benches, smokes)

Experiments run through the launchers in `slurm/repo20260923/`. Each has a header with its rule and modes, and
most take `smoke` and `run`/`bench` modes. Submit them like this (this exact Stand5 smoke ran on 2026-10-04):

```bash
source slurm/_env.sh
slurm_clean sbatch --parsable slurm/repo20260923/gpu_v2_stand5_smoke.sbatch
```

Logs land in `/nfs/hpc/share/$USER/Humanoid_Lite/logs/<job name>-<job id>.out`. Scored runs are predeclared in
`SLURM_JOBS.md` before they are submitted. Do not start a scored run just to try something: use the driver or a
smoke.

## Gotchas

- **No rendering on CPU nodes.** They have neither OSMesa (`MUJOCO_GL=osmesa` fails with `'NoneType' object has no
  attribute 'glGetError'`) nor an EGL device (`MUJOCO_GL=egl` raises `EGLError`). Screenshots need `run.sh gpu`.
- **`srun --gres=gpu:1` without `-n 1` ran the command twice** (two tasks) from the desktop shell. Both copies write
  the same output files. `run.sh` always passes `-n 1`.
- **The desktop's `TMPDIR` (`/scratch/$USER/tmp`) is node-local** and does not exist on compute nodes. srun prints
  `error: Unable to create TMPDIR ... Setting TMPDIR to /tmp`, which is harmless; `run.sh` sets `TMPDIR=/tmp`. Never
  keep anything a later session needs under `/scratch`: another node cannot see it.
- **Do not run heavy Python on the interactive desktop node.** Its cgroup is small (6 GB on some allocations), and
  earlier sessions were OOM-killed there. Use `run.sh`.
- **`slurm_clean` is a shell function**, not a binary: `source slurm/_env.sh` first, and do not `exec` it.
- **EGL teardown noise**: a `mujoco.Renderer` left to the garbage collector prints an ignored `EGLError` traceback
  at exit. The driver closes its renderer explicitly; your own scripts should too.
- **pytest summaries get buried** under MuJoCo warnings (`WARNING: Nan, Inf or huge value in QVEL ...`), and
  `pytest ... | tail` reports tail's exit status, not pytest's. `run.sh test` writes to a file and records
  `PYTEST_EXIT`.
- **Reproducibility check**: scored turn-test results are in
  `results/repo-gpu-20260923/turngait-r12-20261001/turn-test-v2/<run>.json` (`turns[].yaw_deg`, protocol v2: stand
  3 s, then ±0.6 rad/s for 6 s, seeds 0–2). The driver matches them to 0.1°, so a mismatch means the code or the
  policy changed.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `DRIVER_ERROR no exported deploy.yaml for run name '...'` (exit 2) | Use a name from `run.sh cpu policies` (the part after the timestamp), add `--variant biped` for a legs-only policy (e.g. `dr-default-s0`), or pass `--deploy <path>`. |
| `AttributeError: 'NoneType' object has no attribute 'glGetError'` or `EGLError` before any output | Rendering on a CPU node: use `run.sh gpu`. |
| Every line of output appears twice | `srun` launched 2 tasks: add `-n 1` (`run.sh` does). |
| `srun: job ... queued and waiting for resources` for minutes | The GPU partitions are busy; `run.sh cpu` (no `--png`) usually starts at once. |
