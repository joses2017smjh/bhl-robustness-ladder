"""Fixed H3 latency grid and complete, paired six-cell scientific verdict.

Single-cell capture PASS means valid execution; only ``summarize`` can judge
the paired H3 gate. The grid and gates are frozen before training.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import gzip
import json
import math
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from bhl_robust.latency_history import file_sha256, POLICY_DT

COMMANDS = ((.3, 0., 0.), (.5, 0., 0.), (-.2, 0., 0.),
            (0., .2, 0.), (0., 0., .5), (.3, 0., .5))
RESET_SEEDS = tuple(range(201000, 201005))
DELAY_STEPS = (0, 1, 2, 3)
TRAINING_SEEDS = (0, 1, 2)
ARMS = ("history", "repeated_current")
EPISODE_S = 10.0
WARMUP_S = 1.0


def qualification(command, progress, *, fell, completed_steps, expected_steps=250):
    """Each active body-velocity component must realize half its signed integral."""
    active_s = EPISODE_S - WARMUP_S
    required = [abs(float(c)) * active_s * .5 for c in command]
    aligned = [math.copysign(1, c) * float(p) if c else 0.0
               for c, p in zip(command, progress, strict=True)]
    components = [c == 0 or aligned[i] >= required[i] for i, c in enumerate(command)]
    qualified = not fell and completed_steps == expected_steps and all(components)
    return dict(qualified=qualified, component_checks=components,
                required_progress=required, aligned_progress=aligned)


def evaluate_cell(deploy, *, arm, training_seed, out, upstream, smoke=False):
    from omegaconf import OmegaConf
    from bhl_robust.eval.harness import EvalConfig, HeadlessMujocoEnv, TILT_LIMIT_RAD, MAX_SINK_M
    from bhl_robust.eval.history_latency import make_history_controller
    from bhl_robust.eval.mjcf_assets import prepare_mjcf

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.load(deploy)
    cfg.physics_dt = .002
    cfg.policy_dt = POLICY_DT
    scene = prepare_mjcf(Path(upstream), out / "mjcf-cache", variant="biped")
    env = HeadlessMujocoEnv(cfg, scene)
    if env.mj_model.nu != 12 or env.substeps != 20:
        raise ValueError("wrong biped asset or physics decimation")
    seconds, warm = (1.0, .2) if smoke else (EPISODE_S, WARMUP_S)
    n_steps = round(seconds / POLICY_DT)
    warm_steps = round(warm / POLICY_DT)
    seeds = (100,) if smoke else RESET_SEEDS
    delays = (1,) if smoke else DELAY_STEPS
    commands = COMMANDS[:1] if smoke else COMMANDS
    rows = []
    trace_path = out / "latency-traces.jsonl.gz"
    with gzip.open(trace_path, "wt", encoding="utf-8") as trace:
        for delay in delays:
            controller = make_history_controller(cfg, arm=arm, delay_steps=delay)
            controller.load_policy()
            for command_index, command in enumerate(commands):
                for seed in seeds:
                    episode_id = f"{arm}-s{training_seed}-d{delay}-c{command_index}-r{seed}"
                    env.reset(np.random.default_rng(seed), EvalConfig())
                    controller.prev_actions[:] = 0
                    controller.policy_observations[:] = 0
                    controller.reset_packets()
                    obs = env.robot_observations(command)
                    fell = False
                    progress = np.zeros(3)
                    squared_errors = np.zeros(3)
                    samples = 0
                    for step in range(n_steps):
                        targets = controller.update(obs)
                        env.step(targets)
                        obs = env.robot_observations(command)
                        velocity = np.r_[env.base_lin_vel_yaw_frame(), env._base_ang_vel()[2]].astype(float)
                        tilt, sink = env.tilt_rad, env.sink_m
                        fell = tilt > TILT_LIMIT_RAD or sink > MAX_SINK_M
                        if step >= warm_steps:
                            progress += velocity * POLICY_DT
                            squared_errors += (velocity - command) ** 2
                            samples += 1
                        record = dict(episode_id=episode_id, tick=step, source_ticks=controller.last_packet_source_ticks,
                                      captured_imu=controller.last_captured_imu,
                                      actor_obs=controller.policy_observations[0].tolist(),
                                      actions=controller.prev_actions.tolist(), velocity=velocity.tolist(),
                                      tilt_rad=tilt, sink_m=sink, fell=fell)
                        trace.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")
                        if fell:
                            break
                    completed_steps = step + 1
                    row = dict(episode_id=episode_id, arm=arm, training_seed=int(training_seed),
                               delay_steps=delay, delay_ms=delay * 40, command_index=command_index,
                               command=list(command), reset_seed=seed, fell=bool(fell),
                               completed_steps=completed_steps, expected_steps=n_steps,
                               survival_s=completed_steps * POLICY_DT, progress=progress.tolist(),
                               active_samples=samples, tracking_rmse=(np.sqrt(squared_errors / samples).tolist()
                                                                      if samples else None),
                               scored=not smoke)
                    row.update(qualification(command, progress, fell=fell, completed_steps=completed_steps,
                                             expected_steps=n_steps) if not smoke else {"qualified": None})
                    rows.append(row)
                    print(f"H3 episode {episode_id}: fell={fell} qualified={row['qualified']}", flush=True)
    result = dict(schema="h3-latency-evaluation-v1", arm=arm, training_seed=int(training_seed),
                  phase="smoke" if smoke else "run", policy_dt=POLICY_DT, physics_dt=.002,
                  episode_s=seconds, warmup_s=warm, num_episodes=len(rows), episodes=rows,
                  deploy_sha256=file_sha256(deploy), policy_sha256=file_sha256(cfg.policy_checkpoint_path),
                  trace_sha256=file_sha256(trace_path), trace_path=trace_path.name,
                  mjcf_sha256=file_sha256(scene))
    (out / "latency-evaluation.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def validate_evaluation(data, *, arm, seed):
    problems = []
    expected = {(d, c, r) for d in DELAY_STEPS for c in range(len(COMMANDS)) for r in RESET_SEEDS}
    seen = set()
    if data.get("arm") != arm or data.get("training_seed") != seed or data.get("phase") != "run":
        problems.append("cell identity or phase mismatch")
    for key, value in (("policy_dt", .04), ("physics_dt", .002), ("episode_s", 10.), ("warmup_s", 1.)):
        if data.get(key) != value:
            problems.append(f"wrong {key}")
    rows = data.get("episodes", [])
    if len(rows) != 120 or data.get("num_episodes") != 120:
        problems.append("expected exactly 120 episodes")
    for row in rows:
        identity = (row.get("delay_steps"), row.get("command_index"), row.get("reset_seed"))
        if identity in seen or identity not in expected:
            problems.append(f"duplicate or unexpected episode {identity}")
            continue
        seen.add(identity)
        d, c, r = identity
        if row.get("arm") != arm or row.get("training_seed") != seed or row.get("scored") is not True:
            problems.append(f"episode identity mismatch {identity}")
        if row.get("command") != list(COMMANDS[c]) or row.get("delay_ms") != 40 * d:
            problems.append(f"command/delay mismatch {identity}")
        if row.get("expected_steps") != 250 or not isinstance(row.get("completed_steps"), int) or not 1 <= row["completed_steps"] <= 250:
            problems.append(f"invalid step count {identity}")
            continue
        progress = row.get("progress")
        if not isinstance(progress, list) or len(progress) != 3 or not all(isinstance(p, (int, float)) and math.isfinite(p) for p in progress):
            problems.append(f"invalid progress {identity}")
            continue
        if not isinstance(row.get("fell"), bool):
            problems.append(f"invalid fall flag {identity}")
            continue
        recomputed = qualification(COMMANDS[c], progress, fell=row["fell"], completed_steps=row["completed_steps"])
        if row.get("qualified") != recomputed["qualified"]:
            problems.append(f"qualification does not match measured progress {identity}")
        if not row["fell"] and row["completed_steps"] != 250:
            problems.append(f"incomplete upright episode {identity}")
        expected_samples = max(0, row["completed_steps"] - 25)
        if row.get("active_samples") != expected_samples:
            problems.append(f"active sample count mismatch {identity}")
    if seen != expected:
        problems.append("episode grid incomplete")
    return problems


def verify_trace(directory, data):
    """Recompute progress/gates and verify every actor IMU column against past captures."""
    directory = Path(directory)
    trace_path = directory / data.get("trace_path", "latency-traces.jsonl.gz")
    if not trace_path.is_file() or file_sha256(trace_path) != data.get("trace_sha256"):
        return ["trace missing or hash mismatch"]
    expected = {row["episode_id"]: row for row in data["episodes"]}
    packets, count, progress, squared_errors, final = {}, {}, {}, {}, {}
    problems = []
    try:
        with gzip.open(trace_path, "rt", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                identity = row.get("episode_id")
                if identity not in expected:
                    raise ValueError("unexpected trace episode")
                episode = expected[identity]
                tick = row.get("tick")
                if tick != count.get(identity, 0) or identity in final and final[identity]["fell"]:
                    raise ValueError("trace tick skipped/repeated or continued after a fall")
                packet = np.asarray(row["captured_imu"], dtype=np.float32)
                features = np.asarray(row["actor_obs"], dtype=np.float32)
                velocity = np.asarray(row["velocity"], dtype=float)
                if packet.shape != (6,) or features.shape != (63,) or velocity.shape != (3,) or not all(
                    np.isfinite(array).all() for array in (packet, features, velocity)):
                    raise ValueError("invalid trace feature/velocity dimensions or nonfinite values")
                history = packets.setdefault(identity, [])
                history.append(packet)
                age = episode["delay_steps"]
                expected_ticks = [max(0, tick - age - old) for old in (3, 2, 1, 0)] if episode["arm"] == "history" else [max(0, tick - age)] * 4
                if row["source_ticks"] != expected_ticks:
                    raise ValueError("packet source ticks violate frozen delay/history")
                expected_packets = np.asarray([history[source] for source in expected_ticks])
                if not np.array_equal(features[3:9], expected_packets[-1]) or not np.array_equal(
                    features[45:].reshape(3, 6), expected_packets[:-1]):
                    raise ValueError("actor contains wrong/future IMU packet")
                fell = row["tilt_rad"] > .78 or row["sink_m"] > .25
                if row.get("fell") is not bool(fell):
                    raise ValueError("trace fall flag differs from measured tilt/sink")
                if tick >= 25:
                    progress.setdefault(identity, np.zeros(3))[:] += velocity * .04
                    squared_errors.setdefault(identity, np.zeros(3))[:] += (velocity - episode["command"]) ** 2
                count[identity] = tick + 1
                final[identity] = row
        if set(count) != set(expected):
            raise ValueError("trace episode grid incomplete")
        for identity, episode in expected.items():
            if count[identity] != episode["completed_steps"] or final[identity]["fell"] != episode["fell"]:
                raise ValueError("episode count/fall summary differs from raw trace")
            measured = progress.get(identity, np.zeros(3))
            if not np.allclose(measured, episode["progress"], atol=1e-10, rtol=1e-10):
                raise ValueError("episode progress differs from raw trace")
            samples = max(0, count[identity] - 25)
            if samples and not np.allclose(np.sqrt(squared_errors[identity] / samples), episode["tracking_rmse"], atol=1e-10, rtol=1e-10):
                raise ValueError("episode tracking RMSE differs from raw trace")
    except (ValueError, OSError, KeyError, TypeError) as exc:
        problems.append(f"raw trace verification failed: {exc}")
    return problems


def paired_summary(cells):
    """No omissions, checkpoint selection or post-score gate changes permitted."""
    problems, evaluations, receipts = [], {}, {}
    for directory in map(Path, cells):
        try:
            receipt = json.loads((directory / "campaign_result.json").read_text())
            data = json.loads((directory / "latency-evaluation.json").read_text())
        except (OSError, ValueError) as exc:
            problems.append(f"{directory}: missing/unreadable cell: {exc}")
            continue
        identity = receipt.get("arm"), receipt.get("seed")
        if identity in evaluations or identity[0] not in ARMS or identity[1] not in TRAINING_SEEDS:
            problems.append(f"duplicate or unexpected cell {identity}")
            continue
        if receipt.get("phase") != "run" or receipt.get("execution_status") != "COMPLETE" or receipt.get("completed_iterations") != 500 or receipt.get("checkpoint_iteration") != 499:
            problems.append(f"{identity}: training is incomplete or wrong checkpoint")
        arm, seed = identity
        problems.extend(f"{identity}: {p}" for p in validate_evaluation(data, arm=arm, seed=seed))
        problems.extend(f"{identity}: {p}" for p in verify_trace(directory, data))
        if file_sha256(directory / "latency-evaluation.json") != receipt.get("evaluation_sha256"):
            problems.append(f"{identity}: evaluation receipt hash mismatch")
        for name, field in (("final-checkpoint.pt", "checkpoint_sha256"), ("policy.onnx", None)):
            artifact = directory / name
            wanted = receipt.get(field) if field else receipt.get("export", {}).get("policy_sha256")
            if not artifact.is_file() or file_sha256(artifact) != wanted:
                problems.append(f"{identity}: {name} missing or hash mismatch")
        evaluations[identity], receipts[identity] = data, receipt
    expected_cells = {(a, s) for a in ARMS for s in TRAINING_SEEDS}
    if set(evaluations) != expected_cells:
        problems.append("all six frozen cells are required")
    protocol_hashes = {r.get("protocol_sha256") for r in receipts.values()}
    if len(protocol_hashes) != 1 or not protocol_hashes or None in protocol_hashes:
        problems.append("cells do not share one frozen protocol")
    teacher_hashes = {r.get("teacher_sha256") for r in receipts.values()}
    if len(teacher_hashes) != 1 or not teacher_hashes or None in teacher_hashes:
        problems.append("cells do not share one teacher")
    counts, gates = {}, []
    if not problems:
        for (arm, seed), data in evaluations.items():
            counts[f"{arm}-s{seed}"] = {str(40 * d): dict(
                n=30, qualified=sum(r["qualified"] for r in data["episodes"] if r["delay_steps"] == d),
                survived=sum(not r["fell"] for r in data["episodes"] if r["delay_steps"] == d)) for d in DELAY_STEPS}
        for seed in TRAINING_SEEDS:
            h, f = counts[f"history-s{seed}"], counts[f"repeated_current-s{seed}"]
            checks = dict(nominal_survival=h["0"]["survived"] >= 27,
                          nominal_qualified_retention=h["0"]["qualified"] >= f["0"]["qualified"] - 1,
                          primary_80ms_gain=h["80"]["qualified"] - f["80"]["qualified"] >= 6)
            gates.append(dict(seed=seed, checks=checks, pass_gate=all(checks.values()),
                              primary_qualified_delta=h["80"]["qualified"] - f["80"]["qualified"]))
    n_pass = sum(g["pass_gate"] for g in gates)
    return dict(schema="h3-paired-scientific-verdict-v1", status="INCOMPLETE" if problems else
                "PASS" if n_pass >= 2 else "NEGATIVE", problems=problems,
                num_cells=len(evaluations), num_episodes=sum(len(d.get("episodes", [])) for d in evaluations.values()),
                paired_seed_passes=n_pass, required_paired_seed_passes=2, seed_gates=gates, counts=counts,
                protocol_sha256=next(iter(protocol_hashes)) if len(protocol_hashes) == 1 else None,
                interpretation="12-DoF simulation; delayed actor IMU packets only; current joints and commands; no hardware claim")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    capture = sub.add_parser("capture")
    capture.add_argument("--deploy", type=Path, required=True)
    capture.add_argument("--arm", choices=ARMS, required=True)
    capture.add_argument("--seed", type=int, required=True)
    capture.add_argument("--out", type=Path, required=True)
    capture.add_argument("--upstream", type=Path, required=True)
    capture.add_argument("--smoke", action="store_true")
    summarize = sub.add_parser("summarize")
    summarize.add_argument("--cells", type=Path, nargs="+", required=True)
    summarize.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "capture":
        evaluate_cell(args.deploy, arm=args.arm, training_seed=args.seed, out=args.out,
                      upstream=args.upstream, smoke=args.smoke)
        return 0
    result = paired_summary(args.cells)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 1 if result["status"] == "INCOMPLETE" else 0


if __name__ == "__main__":
    raise SystemExit(main())
