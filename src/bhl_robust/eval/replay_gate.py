"""Real-checkpoint MuJoCo regression replay, with immutable input verification.

The oracle here is an independently repeated nominal trajectory. Detecting a
fault means detecting a replay regression, not establishing robot safety or
robustness to that fault. Physics uses the existing upstream observation/PD path.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

import mujoco
import numpy as np
import onnxruntime as ort
from omegaconf import OmegaConf

from berkeley_humanoid_lite_lowlevel.policy.rl_controller import RlController
from bhl_robust.eval.harness import EvalConfig, HeadlessMujocoEnv


FAULTS = ("observation_nan", "observation_shape", "quaternion_xyzw",
          "sensor_hold", "sensor_delay", "action_hold", "action_delay", "contact_score_omitted")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def verify_bundle(bundle: Path) -> dict:
    manifest = json.loads((bundle / "manifest.json").read_text())
    if manifest.get("schema") != "bhl-replay-v1" or not manifest.get("files_sha256"):
        raise ValueError("invalid replay manifest")
    for name, expected in manifest["files_sha256"].items():
        path = bundle / name
        if path.resolve().is_relative_to(bundle.resolve()) is False:
            raise ValueError(f"manifest path escapes bundle: {name}")
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"artifact mismatch or missing file: {name}")
    for package, expected in manifest["packages"].items():
        actual = importlib.metadata.version(package)
        compared = actual.split("+")[0] if package == "torch" else actual
        if compared != expected:
            raise ValueError(f"package mismatch: {package} {actual} != {expected}")
    return manifest


class TimedPolicy:
    """Single-threaded CPU ONNX inference; callable by upstream RlController."""

    def __init__(self, path: Path):
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        self.key = self.session.get_inputs()[0].name
        self.last_ns = 0

    def forward(self, observation):
        start = time.perf_counter_ns()
        result = self.session.run(None, {self.key: observation})[0]
        self.last_ns = time.perf_counter_ns() - start
        return result


def validate_observation(obs, cfg):
    expected = 11 + 2 * int(cfg.num_actions)
    if np.shape(obs) != (expected,):
        raise ValueError(f"observation_shape: expected {expected}, got {np.shape(obs)}")
    if not np.isfinite(obs).all():
        raise ValueError("observation_nonfinite")
    if abs(float(np.linalg.norm(obs[:4])) - 1.0) > 1e-3:
        raise ValueError("quaternion_not_unit")


def validate_target(target, cfg):
    if np.shape(target) != (int(cfg.num_actions),) or not np.isfinite(target).all():
        raise ValueError("target_shape_or_nonfinite")
    lo = np.asarray(cfg.action_limit_lower) * float(cfg.action_scale) + np.asarray(cfg.default_joint_positions)
    hi = np.asarray(cfg.action_limit_upper) * float(cfg.action_scale) + np.asarray(cfg.default_joint_positions)
    if np.any(target < lo - 1e-6) or np.any(target > hi + 1e-6):
        raise ValueError("target_outside_deployed_action_limits")


class RuntimeGuards:
    """Freeze immutable deployment limits once, outside the policy hot path.

    The original validators remain available as an independently timed baseline
    and differential decision oracle. This class changes no accepted ranges.
    """

    def __init__(self, cfg):
        self.num_actions = int(cfg.num_actions)
        self.observation_shape = (11 + 2 * self.num_actions,)
        self.target_shape = (self.num_actions,)
        self.lower = (np.asarray(cfg.action_limit_lower) * float(cfg.action_scale)
                      + np.asarray(cfg.default_joint_positions))
        self.upper = (np.asarray(cfg.action_limit_upper) * float(cfg.action_scale)
                      + np.asarray(cfg.default_joint_positions))
        self.lower.flags.writeable = self.upper.flags.writeable = False

    def observation(self, obs):
        if np.shape(obs) != self.observation_shape:
            raise ValueError(f"observation_shape: expected {self.observation_shape[0]}, got {np.shape(obs)}")
        if not np.isfinite(obs).all():
            raise ValueError("observation_nonfinite")
        if abs(float(np.linalg.norm(obs[:4])) - 1.0) > 1e-3:
            raise ValueError("quaternion_not_unit")

    def target(self, target):
        if np.shape(target) != self.target_shape or not np.isfinite(target).all():
            raise ValueError("target_shape_or_nonfinite")
        if np.any(target < self.lower - 1e-6) or np.any(target > self.upper + 1e-6):
            raise ValueError("target_outside_deployed_action_limits")


def load_runtime(bundle: Path):
    cfg = OmegaConf.load(bundle / "deploy.yaml")
    cfg.policy_checkpoint_path = str(bundle / "policy.onnx")
    policy = TimedPolicy(bundle / "policy.onnx")
    controller = RlController(cfg)
    controller.policy = policy
    env = HeadlessMujocoEnv(cfg, bundle / "scene/bhl_biped_scene.xml")
    return cfg, policy, controller, env


def command_for_step(step, scenario, cfg):
    # Frozen scenario commands are modest probes, not tuned locomotion gates.
    if step * float(cfg.policy_dt) < .6:
        return (0., 0., 0.)
    return tuple(scenario["command"])


def replay(bundle, scenario, seconds, fault=None):
    cfg, _, controller, env = load_runtime(bundle)
    guards = RuntimeGuards(cfg)
    env.reset(np.random.default_rng(scenario["seed"]), EvalConfig())
    controller.prev_actions[:] = 0
    controller.policy_observations[:] = 0
    trigger = int(1.0 / float(cfg.policy_dt))
    obs_history, target_history, trace, contact_steps = [], [], [], 0
    held_obs = held_target = None
    started = time.perf_counter()
    error = None
    for step in range(int(seconds / float(cfg.policy_dt))):
        obs = env.robot_observations(command_for_step(step, scenario, cfg))
        obs_history.append(obs.copy())
        if step >= trigger:
            if fault == "observation_nan":
                obs[4] = np.nan
            elif fault == "observation_shape":
                obs = obs[:-1]
            elif fault == "quaternion_xyzw":
                obs[:4] = np.roll(obs[:4], -1)
            elif fault == "sensor_hold":
                if held_obs is None:
                    held_obs = obs_history[max(0, trigger - 1)][:7].copy()
                obs[:7] = held_obs
            elif fault == "sensor_delay":
                obs[:7] = obs_history[max(0, step - 3)][:7]
        try:
            guards.observation(obs)
            target = controller.update(obs)
            target_history.append(target.copy())
            if step >= trigger:
                if fault == "action_hold":
                    if held_target is None:
                        held_target = target_history[max(0, trigger - 1)].copy()
                    target = held_target
                elif fault == "action_delay":
                    target = target_history[max(0, step - 2)]
            guards.target(target)
        except (ValueError, RuntimeError) as exc:
            error = str(exc)
            break
        # The existing PD law, plus raw per-physics-substep contact evidence.
        # A summary-only contact scorer cannot erase these independent samples.
        targets = np.zeros(env.num_joints, dtype=np.float32)
        targets[env.action_indices] = target
        contacts = []
        for _ in range(env.substeps):
            torque = env.joint_kp * (targets - env._joint_pos()) - env.joint_kd * env._joint_vel()
            env.mj_data.ctrl[:] = np.clip(torque, -env.effort_limits, env.effort_limits)
            mujoco.mj_step(env.mj_model, env.mj_data)
            contacts.append(int(env.mj_data.ncon))
        contact_steps += int(any(contacts))
        trace.append({"step": step, "qpos": env.mj_data.qpos.tolist(),
                      "qvel": env.mj_data.qvel.tolist(), "target": np.asarray(target).tolist(),
                      "contact_counts": contacts, "tilt_rad": env.tilt_rad,
                      "sink_m": env.sink_m})
    return {"scenario": scenario, "fault": fault, "runtime_error": error,
            "trace": trace, "scored_contact_steps": 0 if fault == "contact_score_omitted" else contact_steps,
            "wall_seconds": time.perf_counter() - started,
            "scope": "12-DoF trained biped; CPU MuJoCo simulation; no physical robot"}


def compare_replay(reference, candidate, atol=1e-9, rtol=1e-9):
    """Check raw evidence without consulting the injected fault label."""
    reasons, max_abs = [], 0.0
    if candidate["runtime_error"] is not None:
        reasons.append("runtime_validation_rejected")
    if len(candidate["trace"]) != len(reference["trace"]):
        reasons.append("trace_length_mismatch")
    raw_contact_steps = sum(any(t["contact_counts"]) for t in candidate["trace"])
    if candidate["scored_contact_steps"] != raw_contact_steps:
        reasons.append("contact_score_inconsistent_with_physics_samples")
    for a, b in zip(reference["trace"], candidate["trace"]):
        for key in ("qpos", "qvel", "target"):
            x, y = np.asarray(a[key]), np.asarray(b[key])
            if np.shape(x) != np.shape(y) or not np.isfinite(y).all():
                reasons.append(f"{key}_invalid")
                continue
            max_abs = max(max_abs, float(np.max(np.abs(x - y))))
            if not np.allclose(x, y, atol=atol, rtol=rtol):
                reasons.append(f"{key}_differs_from_nominal_replay")
        if a["contact_counts"] != b["contact_counts"]:
            reasons.append("physics_contact_stream_differs")
    return {"passed": not reasons, "reasons": sorted(set(reasons)), "max_state_or_target_abs_delta": max_abs,
            "atol": atol, "rtol": rtol}


def quantiles(values):
    a = np.asarray(values, dtype=float)
    return {"n": len(a), "p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)),
            "p99_ms": float(np.percentile(a, 99)), "max_ms": float(a.max())}


def hardware_context():
    """Best-effort Linux CPU model and actual runner affinity, without a shell."""
    model = None
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    return {"cpu_model": model, "affinity_cpu_ids": affinity,
            "affinity_count": len(affinity) if affinity is not None else None}


def benchmark(bundle, out, calls, warmup, guard_mode="cached", filename="timings.csv"):
    cfg, policy, controller, env = load_runtime(bundle)
    guards = RuntimeGuards(cfg)
    observation_guard = guards.observation if guard_mode == "cached" else lambda obs: validate_observation(obs, cfg)
    target_guard = guards.target if guard_mode == "cached" else lambda target: validate_target(target, cfg)
    env.reset(np.random.default_rng(94000), EvalConfig())
    # Gather changing observations from actual dynamics; inference is timed
    # against this recorded ring, not all-zero synthetic network inputs.
    ring = []
    for step in range(100):
        obs = env.robot_observations(command_for_step(step, {"command": [.3, 0., .2]}, cfg))
        ring.append(obs.copy())
        env.step(controller.update(obs))
    rows = []
    controller.prev_actions[:] = 0
    controller.policy_observations[:] = 0
    for i in range(warmup + calls):
        start = time.perf_counter_ns()
        # Read the original simulator sensor path and copy a captured frame to
        # vary the network input without timing simulated physics as real I/O.
        obs = env.robot_observations((.3, 0., .2))
        obs[:] = ring[i % len(ring)]
        observed = time.perf_counter_ns()
        observation_guard(obs)
        guarded = time.perf_counter_ns()
        target = controller.update(obs)
        controlled = time.perf_counter_ns()
        target_guard(target)
        end = time.perf_counter_ns()
        if i >= warmup:
            rows.append({"call": i - warmup, "observation_ms": (observed - start) / 1e6,
                         "input_guard_ms": (guarded - observed) / 1e6,
                         "controller_ms": (controlled - guarded) / 1e6,
                         "onnx_ms": policy.last_ns / 1e6,
                         "output_guard_ms": (end - controlled) / 1e6,
                         "total_ms": (end - start) / 1e6})
    with (out / filename).open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    period_ms = float(cfg.policy_dt) * 1000
    return {"calls": calls, "warmup_calls": warmup, "guard_mode": guard_mode, "policy_period_ms": period_ms,
            "lower_level_period_ms": float(cfg.control_dt) * 1000,
            "stages": {key: quantiles([r[key] for r in rows]) for key in rows[0] if key != "call"},
            "policy_deadline_misses": sum(r["total_ms"] > period_ms for r in rows),
            "half_policy_period_target_met": quantiles([r["total_ms"] for r in rows])["p99_ms"] <= period_ms / 2,
            "scope": "Warm serial CPU observation assembly + guards + upstream controller/ONNX; ONNX is nested within controller. "
                     "Captured MuJoCo input ring; excludes physics, sensor transport, OS real-time scheduling and model load. "
                     "Against policy_dt, not control_dt; no hard-real-time guarantee."}


def guard_decision_check(bundle):
    """Compare decisions on real dynamic observations/outputs and bad inputs."""
    cfg, _, controller, env = load_runtime(bundle)
    guards = RuntimeGuards(cfg)
    env.reset(np.random.default_rng(94000), EvalConfig())
    pairs = []
    for step in range(100):
        obs = env.robot_observations(command_for_step(step, {"command": [.3, 0., .2]}, cfg))
        target = controller.update(obs)
        pairs += [("observation", obs.copy()), ("target", target.copy())]
        env.step(target)
    obs = pairs[0][1]
    nonfinite_obs = obs.copy(); nonfinite_obs[4] = np.nan
    nonunit_obs = obs.copy(); nonunit_obs[:4] *= 2
    nonfinite_target = guards.lower.copy(); nonfinite_target[0] = np.inf
    out_of_range = guards.upper.copy(); out_of_range[0] += 2e-6
    pairs += [("observation", nonfinite_obs), ("observation", obs[:-1]),
              ("observation", nonunit_obs), ("target", nonfinite_target),
              ("target", guards.lower[:-1]), ("target", out_of_range),
              ("target", guards.lower.copy()), ("target", guards.upper.copy())]
    decisions = []
    for kind, value in pairs:
        outcomes = []
        for check in ((lambda v: validate_observation(v, cfg)) if kind == "observation" else (lambda v: validate_target(v, cfg)),
                      guards.observation if kind == "observation" else guards.target):
            try:
                check(value)
                outcomes.append(None)
            except ValueError as exc:
                outcomes.append(str(exc))
        decisions.append({"kind": kind, "baseline": outcomes[0], "cached": outcomes[1], "same": outcomes[0] == outcomes[1]})
    return {"cases": len(decisions), "decisions_match": all(d["same"] for d in decisions),
            "accepted": sum(d["baseline"] is None for d in decisions),
            "rejected": sum(d["baseline"] is not None for d in decisions), "decisions": decisions}


def paired_guard_benchmark(bundle, out, calls, warmup):
    # Two matched rounds, reversed order, with the same deterministic dynamic
    # input ring and fresh controller state in every phase. No physics timed.
    rounds = []
    per_phase = max(1, calls // 2)
    for i, mode in enumerate(("uncached", "cached", "cached", "uncached")):
        rounds.append(benchmark(bundle, out, per_phase, warmup, mode, f"paired-{i}-{mode}.csv"))
    paired = []
    for a, b in ((rounds[0], rounds[1]), (rounds[3], rounds[2])):
        before, after = a["stages"]["total_ms"], b["stages"]["total_ms"]
        paired.append({"uncached_total": before, "cached_total": after,
                       "p50_reduction_percent": 100 * (1 - after["p50_ms"] / before["p50_ms"]),
                       "p99_reduction_percent": 100 * (1 - after["p99_ms"] / before["p99_ms"])})
    return {"same_host": True, "phases": rounds, "paired_rounds": paired,
            "scope": "Caching deployment limits in the guards only; local serial warmed microbenchmark, not robot throughput."}


def run_suite(bundle, out, calls=10000, warmup=500):
    manifest = verify_bundle(bundle)
    out.mkdir(parents=True, exist_ok=False)
    receipt = {"manifest_sha256": digest(bundle / "manifest.json"), "argv": sys.argv,
               "host": platform.node(), "platform": platform.platform(), "python": sys.version, "cwd": str(Path.cwd()),
               "hardware_context": hardware_context(),
               "cpu": platform.processor(), "cpu_count": os.cpu_count(), "slurm_job_id": os.getenv("SLURM_JOB_ID"),
               "thread_settings": {k: os.getenv(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
               "packages": manifest["packages"], "installed_packages": {k: importlib.metadata.version(k) for k in manifest["packages"]},
               "source": manifest["source"], "unix_time": time.time()}
    write_new(out / "receipt.json", receipt)
    nominal, mutants = [], []
    for scenario in manifest["scenarios"]:
        ref = replay(bundle, scenario, manifest["seconds"])
        repeat = replay(bundle, scenario, manifest["seconds"])
        verdict = compare_replay(ref, repeat)
        valid_nominal = ref["runtime_error"] is None and len(ref["trace"]) == int(manifest["seconds"] / manifest["policy_dt"])
        if any(t["tilt_rad"] > .78 or t["sink_m"] > .25 for t in ref["trace"]):
            valid_nominal = False
        nominal.append({"seed": scenario["seed"], "valid_nominal": valid_nominal, **verdict})
        write_new(out / f"seed{scenario['seed']}-nominal.json", ref)
        write_new(out / f"seed{scenario['seed']}-repeat.json", repeat)
        for fault in FAULTS:
            candidate = replay(bundle, scenario, manifest["seconds"], fault)
            finding = compare_replay(ref, candidate)
            mutants.append({"seed": scenario["seed"], "fault": fault, "detected": not finding["passed"], **finding})
            write_new(out / f"seed{scenario['seed']}-{fault}.json", candidate)
    timing = benchmark(bundle, out, calls, warmup)
    decisions = guard_decision_check(bundle)
    write_new(out / "guard-decisions.json", decisions)
    paired_timing = paired_guard_benchmark(bundle, out, calls, warmup)
    write_new(out / "paired-guard-timing.json", paired_timing)
    verify_bundle(bundle)
    complete = all(n["valid_nominal"] and n["passed"] for n in nominal) and decisions["decisions_match"]
    detected = sum(m["detected"] for m in mutants)
    report = {"status": "PASS" if complete and detected == len(mutants) else "NEGATIVE",
              "nominal_replays": nominal, "fault_results": mutants,
              "nominal_repeats_passed": sum(n["passed"] and n["valid_nominal"] for n in nominal),
              "nominal_cases": len(nominal), "faults_detected": detected, "fault_cases": len(mutants),
              "timing": timing, "guard_decisions": {k: v for k, v in decisions.items() if k != "decisions"},
              "paired_guard_timing": paired_timing, "manifest_sha256": receipt["manifest_sha256"],
              "limitations": ["Five deterministic development/reset seeds are not a held-out robustness campaign.",
                              "A changed trajectory is a replay regression, not necessarily unsafe or a fall.",
                              "Repeated seeded traces do not measure long-term CI flake rates or cross-host determinism.",
                              "Contact scorer omission is tested against raw per-substep MuJoCo contact counts, including normal floor contacts.",
                              "No new training, physical deployment or navigation improvement is established by this suite."]}
    write_new(out / "report.json", report)
    print(json.dumps({k: report[k] for k in ("status", "nominal_repeats_passed", "nominal_cases", "faults_detected", "fault_cases")}, indent=2))
    return 0 if report["status"] == "PASS" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--calls", type=int, default=10000)
    parser.add_argument("--warmup", type=int, default=500)
    args = parser.parse_args()
    if args.calls < 1 or args.warmup < 0:
        parser.error("calls must be positive; warmup must be nonnegative")
    return run_suite(args.bundle.resolve(), args.out.resolve(), args.calls, args.warmup)


if __name__ == "__main__":
    raise SystemExit(main())
