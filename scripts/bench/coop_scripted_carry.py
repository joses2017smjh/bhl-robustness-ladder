"""Scripted-arm cooperative cube carry in MuJoCo: score seeds, or render one.

    learned gait (frozen) + scripted arms + oracle cube pose

Two modes.

* score (default): run the predeclared seeds (0-9) for a crew of 2 (one pair)
  or 4 (two pairs, two cubes, one world), write one JSON with every episode,
  the per-pair summary under `scripted_carry.SUCCESS_RULE`, and print
  `SCRIPTED_CARRY_VERDICT crew=N verdict=PASS|NEGATIVE|INCOMPLETE`, computed
  from the JSON. The JSON is rewritten after every episode, so a wall-time
  kill keeps what finished.
* render (`--render-from SCORE.json`): read a score JSON; if it did not PASS,
  print `SCRIPTED_CARRY_RENDER=SKIPPED (NEGATIVE)` and write nothing. If it
  passed, re-simulate the median-by-completion-time successful seed with an
  offscreen renderer (needs OpenGL: MUJOCO_GL=egl on a GPU node), write the
  mp4, a GIF within the 5 MiB budget (`panels.write_gif`) and a JSON sidecar
  with source hashes, labels, metrics and playback speed. `--no-render` runs
  the same pipeline with blank frames for a login-node check.

Protocols (`--protocol`, default carry):

* carry: the original experiment above (`scripted_carry.SUCCESS_RULE`,
  30 s, lift then carry), unchanged.
* lift_hold: the separate cooperative LIFT-AND-HOLD experiment
  (`scripted_carry.LIFT_HOLD_RULE`, 20 s): same harness, layout, hand pads,
  kp-30 grasping arm and label, the carry phase removed (zero velocity command
  all episode). Score mode refuses an existing --out and prints
  `COOP-LIFT-HOLD crew N: PASS|NEGATIVE|INCOMPLETE ...` computed from the
  written JSON. Render mode renders the median-by-hold-duration successful
  seed of a PASS, refuses to overwrite any output, and labels frame, banner,
  sidecar and caption "cooperative lift and hold — carry not achieved"
  (`scripted_carry.LIFT_HOLD_NOTE`) with the full label.
* lift_place: the separate cooperative LIFT, HOLD and PLACE experiment
  (`scripted_carry.LIFT_PLACE_RULE`, 20 s, `scripted_carry.PlaceParams`):
  same harness, layout, hand pads, kp-30 grasping arm, cube and label; after
  the lift the pair holds briefly, lowers the cube back onto its plinth,
  opens the hands and stands. Scored seeds are 10-19 ONLY: the protocol
  refuses any seed 0-9 (scored for the other protocols). Score mode refuses
  an existing --out and prints `COOP-LIFT-PLACE crew N: ...` from the written
  JSON. Render mode renders the median-by-hold successful seed of a PASS,
  refuses to overwrite, and labels everything with
  `scripted_carry.LIFT_PLACE_NOTE` ("cooperative lift, hold and place —
  scripted arms, frozen learned gait; carry not achieved").

Clip speed: the frame header states simulated time only (no speed claim). The
mp4 is encoded at round(fps) (12 fps for 12.5 sim frames/s at stride 2, i.e.
0.96x, `playback_speed_mp4`); the GIF badge states the true GIF speed
(gif speed x mp4 speed, e.g. "GIF 1.92x"), also in the sidecar
(`sim_seconds_per_gif_second`). A GIF over the 5 MiB budget is deleted, never
left in docs/gifs, and gets no sidecar.

Everything the script decides is in `scripted_carry.CarryParams` and
`KEYFRAMES_LEFT`; both are copied into every output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = HERE.parents[1]

from bhl_robust.eval import panels                                    # noqa: E402
from bhl_robust.eval import scripted_carry as sc                      # noqa: E402

TASK = "coop_scripted_carry_v1"
TASK_LIFT_HOLD = "coop_lift_hold_v1"
TASK_LIFT_PLACE = "coop_lift_place_v1"


def sha256(path) -> str | None:
    path = Path(path)
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=20).stdout.strip() or None
    except Exception:                                                  # noqa: BLE001
        return None


def parse_seeds(raw: str) -> list[int]:
    out = []
    for part in raw.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def load(args):
    from omegaconf import OmegaConf
    from team_airlock import CpuPolicy

    cfg = OmegaConf.load(args.deploy)
    if cfg.num_actions != 22 or cfg.num_joints != 22 or cfg.num_observations != 75:
        raise SystemExit("requires the full 22-DoF, 75-observation humanoid locomotion policy")
    policy = CpuPolicy(cfg.policy_checkpoint_path)
    p = sc.CarryParams()
    model, slots, pairs = sc.build_carry(args.upstream, args.cache_dir, args.crew // 2, p)
    return cfg, policy, p, model, slots, pairs


def provenance(args, cfg, p) -> dict:
    import mujoco
    return {
        "task": TASK, "label": sc.LABEL, "label_detail": sc.LABEL_DETAIL,
        "learned": "22-DoF locomotion gait (legs + each robot's outer arm), frozen",
        "scripted": "each robot's grasping arm: joint keyframes reach/squeeze/lift/hold",
        "oracle": "cube pose (carry gate + stop) and robot base poses (pair sync) from the simulator",
        "modelling_choices": [
            "hand pads: one box collision geom per hand = hand-mesh AABB (upstream hands are visual-only)",
            f"grasping-arm PD kp {p.grasp_kp} (deploy.yaml 10); kd and the 4 Nm arm effort cap unchanged",
            "grasping arm spawned in the script's rest pose",
            "the cube may rotate in the hands; the rule scores its centre height",
        ],
        "rule": sc.SUCCESS_RULE, "params": sc.params_dict(p), "keyframes_left": sc.KEYFRAMES_LEFT,
        "crew": args.crew, "pairs": args.crew // 2,
        "simulator": f"MuJoCo {mujoco.__version__}", "physics_dt": float(cfg.physics_dt),
        "policy_dt": float(cfg.policy_dt),
        "deploy": str(Path(args.deploy).resolve()), "checkpoint": str(cfg.policy_checkpoint_path),
        "checkpoint_sha256": sha256(cfg.policy_checkpoint_path),
        "module_sha256": sha256(REPO / "src/bhl_robust/eval/scripted_carry.py"),
        "script_sha256": sha256(Path(__file__)), "git_head": git_head(),
    }


def provenance_for(args, cfg, p) -> dict:
    """`provenance` for the chosen protocol (carry: exactly `provenance`)."""
    out = provenance(args, cfg, p)
    if getattr(args, "protocol", "carry") == "lift_place":
        q = sc.PlaceParams()
        out.update({
            "task": TASK_LIFT_PLACE, "protocol": "lift_place", "note": sc.LIFT_PLACE_NOTE,
            "label_detail": sc.LIFT_PLACE_LABEL_DETAIL,
            "scripted": "each robot's grasping arm: joint keyframes reach/squeeze/lift, hold, lower "
                        "(PlaceParams.lower_to, reached from the lift keyframe), open, rest",
            "oracle": ("cube pose from the simulator, used only to score; robot base poses drive "
                       "station keeping" if q.station_keep else
                       "cube pose from the simulator, used only to score; every velocity command is zero"),
            "modelling_choices": out["modelling_choices"] + [
                f"lowering starts at {sc.PlaceScript(p, q).t_lower_start:.2f} s so the cube is on the plinth "
                "before the earliest grip slip seen under lift_hold on exploration seeds 100-119",
                f"lowering target (LEFT arm) {list(sc.PlaceScript(p, q).lower_target)} "
                + sc.describe_lower_target(sc.PlaceScript(p, q).lower_target),
                "station keeping " + ("ON (oracle base poses)" if q.station_keep else "OFF"),
            ],
            "rule": sc.LIFT_PLACE_RULE, "place_params": sc.place_params_dict(q),
        })
        return out
    if getattr(args, "protocol", "carry") == "lift_hold":
        out.update({
            "task": TASK_LIFT_HOLD, "protocol": "lift_hold", "note": sc.LIFT_HOLD_NOTE,
            "label_detail": sc.LIFT_HOLD_LABEL_DETAIL,
            "scripted": "each robot's grasping arm: joint keyframes reach/squeeze/lift, then held",
            "oracle": "cube pose from the simulator, used only to score (no carry gate, no sync); "
                      "every velocity command is zero",
            "rule": sc.LIFT_HOLD_RULE,
        })
    return out


def verdict_of(summary: dict) -> str:
    if not summary["complete"]:
        return "INCOMPLETE"
    return "PASS" if summary["pass"] else "NEGATIVE"


# ------------------------------------------------------------------ score

def score(args) -> int:
    if getattr(args, "protocol", "carry") == "lift_hold":
        return score_lift_hold(args)
    if getattr(args, "protocol", "carry") == "lift_place":
        return score_lift_place(args)
    cfg, policy, p, model, slots, pairs = load(args)
    rule = dict(sc.SUCCESS_RULE)
    seeds = parse_seeds(args.seeds)
    if args.seconds is not None:
        # a shortened episode is a pipeline check, never a scored run
        rule["episode_s"] = float(args.seconds)
    payload = provenance(args, cfg, p)
    payload["scored_run"] = args.seconds is None and seeds == list(sc.SUCCESS_RULE["seeds"])
    payload["episodes"] = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        t0 = time.time()
        ep = sc.run_episode(model, slots, pairs, cfg, policy, seed, p, rule=rule)
        ep["wall_s"] = round(time.time() - t0, 1)
        payload["episodes"].append(ep)
        for r in ep["pairs"]:
            print(json.dumps({"crew": args.crew, "seed": seed, "pair": r["pair"],
                              "success": r["success"], "first_failed_check": r["first_failed_check"],
                              "lift_peak_m": r["lift_peak_m"], "lift_hold_s": r["lift_hold_s"],
                              "carry_m": r["carry_m"], "max_tilt_rad": r["max_tilt_rad"],
                              "floor": r["cube_floor_contact"], "completion_s": r["completion_s"],
                              "wall_s": ep["wall_s"]}), flush=True)
        payload["summary"] = sc.summarize(payload["episodes"], args.crew // 2, rule)
        payload["summary"]["median_seed"] = sc.median_seed(payload["episodes"])
        payload["summary"]["verdict"] = verdict_of(payload["summary"])
        if not payload["scored_run"]:
            payload["summary"]["verdict"] = "PIPELINE_CHECK"
            payload["summary"]["pass"] = False
        args.out.write_text(json.dumps(payload, indent=1, allow_nan=False) + "\n")
    s = payload["summary"]
    counts = " ".join(f"pair{pp['pair']}={pp['successes']}/{pp['episodes']}" for pp in s["per_pair"])
    print(f"SCRIPTED_CARRY_VERDICT crew={args.crew} verdict={s['verdict']} {counts} "
          f"median_seed={s['median_seed']} json={args.out}", flush=True)
    return 0


def score_lift_hold(args) -> int:
    """Score mode of the lift-and-hold protocol. Never overwrites --out."""
    if args.out.exists():
        raise SystemExit(f"COOP-LIFT-HOLD: REFUSED ({args.out} exists; score JSONs are never overwritten)")
    rule = dict(sc.LIFT_HOLD_RULE)
    seeds = parse_seeds(args.seeds)
    if args.seconds is not None:
        # a shortened episode is a pipeline check, never a scored run
        rule["episode_s"] = float(args.seconds)
    cfg, policy, p, model, slots, pairs = load(args)
    payload = provenance_for(args, cfg, p)
    payload["scored_run"] = args.seconds is None and seeds == list(sc.LIFT_HOLD_RULE["seeds"])
    payload["episodes"] = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        t0 = time.time()
        ep = sc.run_episode(model, slots, pairs, cfg, policy, seed, p, rule=rule, protocol="lift_hold")
        ep["wall_s"] = round(time.time() - t0, 1)
        payload["episodes"].append(ep)
        for r in ep["pairs"]:
            print(json.dumps({"protocol": "lift_hold", "crew": args.crew, "seed": seed, "pair": r["pair"],
                              "success": r["success"], "first_failed_check": r["first_failed_check"],
                              "lift_peak_m": r["lift_peak_m"], "lift_hold_s": r["lift_hold_s"],
                              "hold_start_s": r["hold_start_s"], "hold_end_s": r["hold_end_s"],
                              "max_tilt_rad": r["max_tilt_rad"], "floor": r["cube_floor_contact"],
                              "final_lift_m": r["final_lift_m"], "horiz_max_m": r["horiz_max_m"],
                              "wall_s": ep["wall_s"]}), flush=True)
        payload["summary"] = sc.summarize_lift_hold(payload["episodes"], args.crew // 2, rule)
        payload["summary"]["median_seed"] = sc.median_seed_by_hold(payload["episodes"])
        payload["summary"]["verdict"] = verdict_of(payload["summary"])
        if not payload["scored_run"]:
            payload["summary"]["verdict"] = "PIPELINE_CHECK"
            payload["summary"]["pass"] = False
        args.out.write_text(json.dumps(payload, indent=1, allow_nan=False) + "\n")
    # the verdict line is computed from the JSON as written
    print(sc.lift_hold_verdict_line(json.loads(args.out.read_text()), args.crew) + f" | json={args.out}",
          flush=True)
    return 0


def score_lift_place(args) -> int:
    """Score mode of the lift-hold-place protocol. Never overwrites --out and
    never runs a seed 0-9 (those are scored for the carry / lift_hold rules)."""
    if args.out.exists():
        raise SystemExit(f"COOP-LIFT-PLACE: REFUSED ({args.out} exists; score JSONs are never overwritten)")
    seeds = parse_seeds(args.seeds)
    if any(0 <= s < 10 for s in seeds):
        raise SystemExit("COOP-LIFT-PLACE: REFUSED (seeds 0-9 are scored for the carry / lift_hold "
                         "protocols; lift_place is scored on seeds 10-19 only)")
    reserved = list(sc.LIFT_PLACE_RULE["seeds"])
    if any(s in reserved for s in seeds) and not (seeds == reserved and args.seconds is None):
        raise SystemExit("COOP-LIFT-PLACE: REFUSED (seeds 10-19 are reserved for the scored run: "
                         "all of 10-19, full-length episodes, or none of them)")
    rule = dict(sc.LIFT_PLACE_RULE)
    if args.seconds is not None:
        # a shortened episode is a pipeline check, never a scored run
        rule["episode_s"] = float(args.seconds)
    cfg, policy, p, model, slots, pairs = load(args)
    q = sc.PlaceParams()
    payload = provenance_for(args, cfg, p)
    payload["scored_run"] = args.seconds is None and seeds == list(sc.LIFT_PLACE_RULE["seeds"])
    payload["episodes"] = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        t0 = time.time()
        ep = sc.run_place_episode(model, slots, pairs, cfg, policy, seed, p, q=q, rule=rule)
        ep["wall_s"] = round(time.time() - t0, 1)
        payload["episodes"].append(ep)
        for r in ep["pairs"]:
            print(json.dumps({"protocol": "lift_place", "crew": args.crew, "seed": seed, "pair": r["pair"],
                              "success": r["success"], "first_failed_check": r["first_failed_check"],
                              "lift_peak_m": r["lift_peak_m"], "lift_hold_s": r["lift_hold_s"],
                              "hold_end_s": r["hold_end_s"], "max_tilt_rad": r["max_tilt_rad"],
                              "floor": r["cube_floor_contact"], "final_dz_m": r["final_dz_m"],
                              "final_speed_mps": r["final_speed_mps"],
                              "final_offset_xy_m": r["final_offset_xy_m"],
                              "final_robot_contact": r["final_robot_contact"],
                              "station_disp_max_m": r["station_disp_max_m"],
                              "wall_s": ep["wall_s"]}), flush=True)
        payload["summary"] = sc.summarize_lift_place(payload["episodes"], args.crew // 2, rule)
        payload["summary"]["median_seed"] = sc.median_seed_by_hold(payload["episodes"])
        payload["summary"]["verdict"] = verdict_of(payload["summary"])
        if not payload["scored_run"]:
            payload["summary"]["verdict"] = "PIPELINE_CHECK"
            payload["summary"]["pass"] = False
        args.out.write_text(json.dumps(payload, indent=1, allow_nan=False) + "\n")
    # the verdict line is computed from the JSON as written
    print(sc.lift_place_verdict_line(json.loads(args.out.read_text()), args.crew) + f" | json={args.out}",
          flush=True)
    return 0


# ------------------------------------------------------------------ flush-pad variant (C1, opt-in)
#
# `--flushpad-stage {probe,hold,place,smoke}` only (no flag = every path above, unchanged). A MODIFIED
# END-EFFECTOR, not the stock robot: `scripted_carry.FlushPadParams` (flush hand pads, condim 4, torsional
# friction 0.04 m) plus the HARNESS CHANGE (elliptic cone, impratio 10), and for lift_place the frozen
# reverse-keyframe lowering (`FlushPadPlaceParams`). Labels: LEARNED gait (frozen arms-dr1.0-s0) + SCRIPTED arms +
# ORACLE cube pose (scoring only). Seeds are fixed per stage: probe 120-124 (crew 2), hold 20-29 and place 30-39
# (crews 2 and 4), smoke only on the declared tuning seeds 100-119. Score JSONs are never overwritten; the stage
# verdict JSONs are written from them by `flushpad_verdict_cli` (slurm/repo20260923/cpu_coop_flushpad.sbatch).

TASKS_FLUSHPAD = {"probe": "coop_flushpad_probe_v1", "hold": "coop_flushpad_lift_hold_v1",
                  "place": "coop_flushpad_lift_place_v1", "smoke": "coop_flushpad_smoke_v1"}
FLUSHPAD_STAGE_HELP = ("OPT-IN C1 flush-pad variant (MODIFIED END-EFFECTOR + harness change): probe (120-124), "
                       "hold (20-29), place (30-39) or smoke (tuning seeds 100-119); no flag = stock paths")


def flushpad_guard(args) -> tuple:
    """(rule, protocol, seeds) of a --flushpad-stage run; SystemExit before anything is loaded or written."""
    stage = args.flushpad_stage
    if stage not in sc.FLUSHPAD_STAGES:
        raise SystemExit(f"COOP-FLUSHPAD: REFUSED (unknown stage {stage!r})")
    if getattr(args, "render_from", None):
        raise SystemExit("COOP-FLUSHPAD: REFUSED (the flush-pad variant has no render mode)")
    if args.out is None:
        raise SystemExit("COOP-FLUSHPAD: REFUSED (--flushpad-stage needs --out)")
    if args.out.exists():
        raise SystemExit(f"COOP-FLUSHPAD: REFUSED ({args.out} exists; score JSONs are never overwritten)")
    seeds = parse_seeds(args.seeds)
    if stage == "smoke":
        if not seeds or any(s not in sc.FLUSHPAD_SMOKE_SEEDS for s in seeds):
            raise SystemExit("COOP-FLUSHPAD: REFUSED (smoke runs only on the declared tuning seeds 100-119; "
                             "120-124 are the probe's and 20-39 the scored stage's)")
        if args.protocol not in ("lift_hold", "lift_place"):
            raise SystemExit("COOP-FLUSHPAD: REFUSED (smoke needs --protocol lift_hold or lift_place)")
        if args.crew not in (2, 4):
            raise SystemExit("COOP-FLUSHPAD: REFUSED (crew 2 or 4)")
        return sc.flushpad_stage_rule("smoke", args.protocol, seeds), args.protocol, seeds
    rule = sc.flushpad_stage_rule(stage)
    if args.protocol != rule["protocol"]:
        raise SystemExit(f"COOP-FLUSHPAD: REFUSED (stage {stage} is protocol {rule['protocol']})")
    if seeds != list(rule["seeds"]) or args.seconds is not None:
        raise SystemExit(f"COOP-FLUSHPAD: REFUSED (stage {stage} runs exactly seeds {rule['seeds'][0]}-"
                         f"{rule['seeds'][-1]}, full-length episodes)")
    if args.crew not in rule["crews"]:
        raise SystemExit(f"COOP-FLUSHPAD: REFUSED (stage {stage} runs crew {rule['crews']})")
    return rule, rule["protocol"], seeds


def load_flushpad(args):
    """`load` with `FlushPadParams` (the stock `load` is unchanged)."""
    from omegaconf import OmegaConf
    from team_airlock import CpuPolicy

    cfg = OmegaConf.load(args.deploy)
    if cfg.num_actions != 22 or cfg.num_joints != 22 or cfg.num_observations != 75:
        raise SystemExit("requires the full 22-DoF, 75-observation humanoid locomotion policy")
    policy = CpuPolicy(cfg.policy_checkpoint_path)
    p = sc.FlushPadParams()
    model, slots, pairs = sc.build_carry(args.upstream, args.cache_dir, args.crew // 2, p)
    return cfg, policy, p, model, slots, pairs


def provenance_flushpad(args, cfg, p, model, stage: str, rule: dict, protocol: str) -> dict:
    out = provenance(args, cfg, p)
    place = protocol == "lift_place"
    q = sc.FlushPadPlaceParams()
    out.update({
        "task": TASKS_FLUSHPAD[stage], "variant": sc.FLUSHPAD_VARIANT, "stage": stage, "protocol": protocol,
        "note": sc.LIFT_PLACE_NOTE if place else sc.LIFT_HOLD_NOTE,
        "label": sc.FLUSHPAD_LABEL, "label_detail": sc.FLUSHPAD_LABEL_DETAIL,
        "end_effector": sc.FLUSHPAD_NOTE, "harness_change": sc.FLUSHPAD_HARNESS_NOTE,
        "learned": "22-DoF locomotion gait arms-dr1.0-s0 (legs + each robot's outer arm), frozen",
        "scripted": ("each robot's grasping arm: joint keyframes reach/squeeze/lift (unchanged), "
                     + ("hold, lower (reverse keyframe: lift -> squeeze), open, rest" if place else "then held")),
        "oracle": "cube pose from the simulator, used only to score; every velocity command is zero",
        "modelling_choices": [
            "MODIFIED END-EFFECTOR: one box collision pad per hand = the hand-mesh AABB (stock size and centre), "
            "re-oriented in the hand-link frame so its faces are parallel to the cube faces at the squeeze pose "
            "(squeeze keyframe, shoulder roll stopped where the pad meets the cube face plane); see flush_pad_design",
            f"pad contact condim {p.pad_condim}, friction [{p.pad_friction}, {p.pad_torsional_friction}, 0.0001] "
            "(torsional 0.04 m)",
            f"HARNESS CHANGE: cone {p.solver_cone}, impratio {p.solver_impratio} (global MuJoCo options; they also "
            "change the foot-floor contact; robot-fall clauses unchanged)",
            f"grasping-arm PD kp {p.grasp_kp} (deploy.yaml 10); kd and the 4 Nm arm effort cap unchanged",
            "grasping arm spawned in the script's rest pose; arm keyframes and timeline unchanged",
            "cube tilt (scoring only) = angle of the cube's z axis from world z at every policy-step state",
            "numerical blow-ups: MuJoCo's automatic reset (qpos0, time 0, on a NaN or |x| > 1e10 in qpos/qvel/qacc) "
            "is detected after every policy step (episode field sim_reset) and fails the episode (no_sim_reset); "
            "a non-finite stop's NaNs are written as null (nonfinite_values_nulled) and it is scored as a failure",
        ] + ([f"lowering target: {sc.describe_lower_target(sc.PlaceScript(p, q).lower_target)} (FROZEN "
              "2026-10-02 before any probe or scored episode)"] if place else []),
        "rule": rule, "rule_text": (sc.FLUSHPAD_PROBE_RULE_TEXT if stage == "probe" else sc.FLUSHPAD_SCORED_RULE_TEXT),
        "clauses_as_applied": list(sc.FLUSHPAD_CLAUSES_AS_APPLIED),
        "seed_evidence": list(sc.FLUSHPAD_SEED_EVIDENCE),
        "base_rule": sc.LIFT_PLACE_RULE if place else sc.LIFT_HOLD_RULE,
        "flush_pad_design": sc.flushpad_design(args.upstream, args.cache_dir, p),
        "model_check": sc.flushpad_model_check(model),
    })
    if place:
        out["place_params"] = sc.place_params_dict(q)
    return out


def score_flushpad(args) -> int:
    """Score mode of the flush-pad variant. Never overwrites --out; refuses any seed outside its stage."""
    rule, protocol, seeds = flushpad_guard(args)
    if args.seconds is not None:
        # smoke only (the guard refuses --seconds for probe/hold/place): a pipeline check, never scored
        rule = dict(rule, episode_s=float(args.seconds))
    cfg, policy, p, model, slots, pairs = load_flushpad(args)
    payload = provenance_flushpad(args, cfg, p, model, args.flushpad_stage, rule, protocol)
    if not sc.flushpad_model_ok(payload["model_check"], len(slots)):
        raise SystemExit(f"COOP-FLUSHPAD: model check failed: {payload['model_check']}")
    payload["scored_run"] = args.flushpad_stage != "smoke"
    payload["episodes"] = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    dt = float(cfg.policy_dt)
    for seed in seeds:
        t0 = time.time()
        ep = sc.run_flushpad_episode(model, slots, pairs, cfg, policy, seed, p, rule=rule)
        ep["wall_s"] = round(time.time() - t0, 1)
        payload["episodes"].append(ep)
        for k, r in enumerate(ep["pairs"]):
            fc = sc.flushpad_clauses(rule, ep, k, dt)
            print(json.dumps({"variant": sc.FLUSHPAD_VARIANT, "stage": args.flushpad_stage, "protocol": protocol,
                              "crew": args.crew, "seed": seed, "pair": r["pair"],
                              "flushpad_success": fc["success"], "flushpad_first_failed": fc["first_failed_check"],
                              "base_rule_success": r["success"], "base_first_failed": r["first_failed_check"],
                              "cube_tilt_max_rad": max([v for v in r["cube_tilt_series_rad"] if v is not None],
                                                       default=None),
                              "lift_peak_m": r["lift_peak_m"], "lift_hold_s": r["lift_hold_s"],
                              "max_robot_tilt_rad": r["max_tilt_rad"], "floor": r["cube_floor_contact"],
                              "wall_s": ep["wall_s"]}), flush=True)
        payload["summary"] = sc.flushpad_file_summary(payload)
        args.out.write_text(json.dumps(payload, indent=1, allow_nan=False) + "\n")
    # the line is computed from the JSON as written
    print(sc.flushpad_verdict_line(sc.flushpad_file_summary(json.loads(args.out.read_text())))
          + f" | json={args.out}", flush=True)
    return 0


def flushpad_verdict_cli(argv) -> int:
    """Stage verdict of the flush-pad launcher, from score JSONs only (`python -c` entry of
    cpu_coop_flushpad.sbatch): argv = [STAGE, VERDICT_JSON, KEY=PATH, ...] with STAGE
      probe          KEY probe (the probe score JSON)                    -> PROCEED|NEGATIVE|INCOMPLETE|INVALID
      scored         KEYS hold_crew2 hold_crew4 place_crew2 place_crew4  -> PASS|NEGATIVE|INCOMPLETE|INVALID
      scored_not_run KEY probe_verdict (the probe verdict JSON)          -> NOT_RUN
      smoke          KEY step_status (the launcher's step exit statuses, unit tests included) and any other
                     KEYS (smoke score JSONs)                            -> SMOKE_PASS|SMOKE_FAIL
    A missing or unreadable input reads as missing. Refuses to overwrite VERDICT_JSON. Returns 0 whatever the
    verdict: the launcher reads the verdict from the JSON, never from this exit status."""
    if len(argv) < 3:
        raise SystemExit("usage: STAGE VERDICT_JSON KEY=PATH ...")
    stage, out = argv[0], Path(argv[1])
    if out.exists():
        raise SystemExit(f"COOP-FLUSHPAD-VERDICT: REFUSED ({out} exists; verdicts are never overwritten)")
    paths = {}
    for kv in argv[2:]:
        k, _, v = kv.partition("=")
        if not k or not v:
            raise SystemExit(f"COOP-FLUSHPAD-VERDICT: bad input {kv!r} (KEY=PATH)")
        paths[k] = Path(v)

    def read(path):
        try:
            return json.loads(Path(path).read_text())
        except (OSError, ValueError):
            return None

    if stage == "probe":
        v = sc.flushpad_probe_verdict(read(paths["probe"]) if "probe" in paths else None)
    elif stage == "scored":
        v = sc.flushpad_scored_verdict({k: read(paths[k]) if k in paths else None for k in sc.FLUSHPAD_SCORED_KEYS})
    elif stage == "scored_not_run":
        v = sc.flushpad_not_run_verdict(read(paths["probe_verdict"]) if "probe_verdict" in paths else None)
    elif stage == "smoke":
        status = read(paths["step_status"]) if "step_status" in paths else None
        step_problems = sc.flushpad_smoke_step_problems(status)
        unit_ok = isinstance(status, dict) and type(status.get("pytest")) is int and status["pytest"] == 0
        checks = {k: sc.flushpad_smoke_check(read(path)) for k, path in paths.items() if k != "step_status"}
        ok = bool(checks) and all(c["verdict"] == "SMOKE_PASS" for c in checks.values()) and not step_problems
        v = {**sc._flushpad_head("smoke"), "verdict": "SMOKE_PASS" if ok else "SMOKE_FAIL",
             "step_status": status, "unit_tests_passed": bool(unit_ok),
             "episodes": [r for c in checks.values() for r in c["episodes"]],
             "problems": [f"step_status: {p}" for p in step_problems]
             + [f"{k}: {p}" for k, c in checks.items() for p in c["problems"]], "checks": checks,
             "note": "pipeline check on a declared tuning seed; outcomes are diagnostics, never a verdict"}
    else:
        raise SystemExit(f"COOP-FLUSHPAD-VERDICT: unknown stage {stage!r}")
    v["inputs"] = {k: {"path": str(path), "sha256": sha256(path)} for k, path in paths.items()}
    v["code_sha256"] = {"scripted_carry.py": sha256(REPO / "src/bhl_robust/eval/scripted_carry.py"),
                        "coop_scripted_carry.py": sha256(Path(__file__))}
    v["git_head"] = git_head()
    v["written_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(v, indent=1, allow_nan=False) + "\n")
    print(sc.flushpad_verdict_line(v) + f" | verdict_json={out}", flush=True)
    return 0


# --- coop-wristhold --- (W, opt-in; 2026-10-03) ----------------------------------------------------------------------
#
# The W variant's own entry point, `wristhold_main(argv)`, which slurm/repo20260923/cpu_coop_wristhold.sbatch calls as
#   python -c '...; import coop_scripted_carry as m; sys.exit(m.wristhold_main(sys.argv[1:]))' --wristhold-stage ...
# so `main()` and its flags stay byte-identical (no flag is added there). The flush-pad variant UNCHANGED (MODIFIED
# END-EFFECTOR + HARNESS CHANGE, reverse-keyframe lowering) PLUS the kinematic wrist-orientation hold
# (`scripted_carry.WristHoldParams`, `WristHold`). Labels: LEARNED gait (frozen arms-dr1.0-s0) + SCRIPTED arms (with a
# kinematic wrist-orientation hold) + ORACLE cube pose (scoring only). Stages: probe (crew 2, lift_hold, seeds
# 125-129), hold (crews 2/4, lift_hold, seeds 20-29), place (crews 2/4, lift_place, seeds 30-39), smoke (throw-away
# seeds 140-144; crew 2 or 4; lift_hold or lift_place) and smoke_nohold (the smoke's no-hold baseline: the flush-pad
# variant unchanged, crew 2, lift_hold, seeds 140-144, the hand's twist measured only). Full-length episodes only.
# Score JSONs are never overwritten; the stage verdict JSONs are written from them by `wristhold_verdict_cli`.

TASKS_WRISTHOLD = {"probe": "coop_wristhold_probe_v1", "hold": "coop_wristhold_lift_hold_v1",
                   "place": "coop_wristhold_lift_place_v1", "smoke": "coop_wristhold_smoke_v1",
                   "smoke_nohold": "coop_wristhold_smoke_nohold_v1"}


def wristhold_guard(args) -> tuple:
    """(rule, protocol, seeds) of a --wristhold-stage run; SystemExit before anything is loaded or written."""
    stage = args.wristhold_stage
    if stage not in sc.WRISTHOLD_STAGES:
        raise SystemExit(f"COOP-WRISTHOLD: REFUSED (unknown stage {stage!r})")
    if args.out is None:
        raise SystemExit("COOP-WRISTHOLD: REFUSED (--wristhold-stage needs --out)")
    if args.out.exists():
        raise SystemExit(f"COOP-WRISTHOLD: REFUSED ({args.out} exists; score JSONs are never overwritten)")
    seeds = parse_seeds(args.seeds)
    if stage in ("smoke", "smoke_nohold"):
        if not seeds or any(s not in sc.WRISTHOLD_SMOKE_SEEDS for s in seeds):
            raise SystemExit("COOP-WRISTHOLD: REFUSED (smoke runs only on the throw-away seeds 140-144; 125-129 are "
                             "the probe's and 20-39 the scored stage's)")
        if stage == "smoke_nohold":
            if args.protocol != "lift_hold" or args.crew != 2:
                raise SystemExit("COOP-WRISTHOLD: REFUSED (the no-hold baseline is crew 2, lift_hold)")
        else:
            if args.protocol not in ("lift_hold", "lift_place"):
                raise SystemExit("COOP-WRISTHOLD: REFUSED (smoke needs --protocol lift_hold or lift_place)")
            if args.crew not in (2, 4):
                raise SystemExit("COOP-WRISTHOLD: REFUSED (crew 2 or 4)")
        return sc.wristhold_stage_rule(stage, args.protocol, seeds), args.protocol, seeds
    rule = sc.wristhold_stage_rule(stage)
    if args.protocol != rule["protocol"]:
        raise SystemExit(f"COOP-WRISTHOLD: REFUSED (stage {stage} is protocol {rule['protocol']})")
    if seeds != list(rule["seeds"]):
        raise SystemExit(f"COOP-WRISTHOLD: REFUSED (stage {stage} runs exactly seeds {rule['seeds'][0]}-"
                         f"{rule['seeds'][-1]})")
    if args.crew not in rule["crews"]:
        raise SystemExit(f"COOP-WRISTHOLD: REFUSED (stage {stage} runs crew {rule['crews']})")
    return rule, rule["protocol"], seeds


def load_wristhold(args, nohold: bool):
    """`load` with `WristHoldParams` (or `FlushPadParams` for the no-hold baseline); the stock `load` is unchanged."""
    from omegaconf import OmegaConf
    from team_airlock import CpuPolicy

    cfg = OmegaConf.load(args.deploy)
    if cfg.num_actions != 22 or cfg.num_joints != 22 or cfg.num_observations != 75:
        raise SystemExit("requires the full 22-DoF, 75-observation humanoid locomotion policy")
    policy = CpuPolicy(cfg.policy_checkpoint_path)
    p = sc.FlushPadParams() if nohold else sc.WristHoldParams()
    model, slots, pairs = sc.build_carry(args.upstream, args.cache_dir, args.crew // 2, p)
    return cfg, policy, p, model, slots, pairs


def provenance_wristhold(args, cfg, p, model, stage: str, rule: dict, protocol: str) -> dict:
    out = provenance(args, cfg, p)
    place = protocol == "lift_place"
    nohold = stage == "smoke_nohold"
    q = sc.FlushPadPlaceParams()
    hold_line = ("NO HOLD (baseline): the flush-pad variant unchanged; each grasping hand's rotation about the pinch "
                 "axis is measured only" if nohold else sc.WRISTHOLD_HOLD_NOTE)
    design = sc.wristhold_design(args.upstream, args.cache_dir, sc.WristHoldParams())
    out.update({
        "task": TASKS_WRISTHOLD[stage], "variant": sc.WRISTHOLD_VARIANT, "stage": stage, "protocol": protocol,
        "note": sc.LIFT_PLACE_NOTE if place else sc.LIFT_HOLD_NOTE,
        "label": sc.WRISTHOLD_NOHOLD_LABEL if nohold else sc.WRISTHOLD_LABEL,
        "label_detail": sc.FLUSHPAD_LABEL_DETAIL if nohold else sc.WRISTHOLD_LABEL_DETAIL,
        "end_effector": sc.FLUSHPAD_NOTE, "harness_change": sc.FLUSHPAD_HARNESS_NOTE, "wrist_hold": hold_line,
        "learned": "22-DoF locomotion gait arms-dr1.0-s0 (legs + each robot's outer arm), frozen",
        "scripted": ("each robot's grasping arm: joint keyframes reach/squeeze/lift (unchanged), "
                     + ("hold, lower (reverse keyframe: lift -> squeeze), open, rest" if place else "then held")
                     + ("" if nohold else "; its wrist (elbow roll) servoed by the kinematic hold from the grasp "
                        "state on (lift_place: through the release, then withdrawn over the retract segment)")),
        "oracle": "cube pose from the simulator, used only to score; every velocity command is zero",
        "modelling_choices": [
            "MODIFIED END-EFFECTOR: one box collision pad per hand = the hand-mesh AABB (stock size and centre), "
            "re-oriented in the hand-link frame so its faces are parallel to the cube faces at the squeeze pose "
            "(squeeze keyframe, shoulder roll stopped where the pad meets the cube face plane); see flush_pad_design",
            f"pad contact condim {p.pad_condim}, friction [{p.pad_friction}, {p.pad_torsional_friction}, 0.0001] "
            "(torsional 0.04 m)",
            f"HARNESS CHANGE: cone {p.solver_cone}, impratio {p.solver_impratio} (global MuJoCo options; they also "
            "change the foot-floor contact; robot-fall clauses unchanged)",
            f"grasping-arm PD kp {p.grasp_kp} (deploy.yaml 10); kd and the 4 Nm arm effort cap unchanged",
            "grasping arm spawned in the script's rest pose; arm keyframes and timeline unchanged",
            "cube tilt (scoring only) = angle of the cube's z axis from world z at every policy-step state",
            "numerical blow-ups: MuJoCo's automatic reset (qpos0, time 0, on a NaN or |x| > 1e10 in qpos/qvel/qacc) "
            "is detected after every policy step (episode field sim_reset) and fails the episode (no_sim_reset); "
            "a non-finite stop's NaNs are written as null (nonfinite_values_nulled) and it is scored as a failure",
        ] + ([] if nohold else [
            "WRIST-ORIENTATION HOLD: from the grasp state (first policy-step state at or after the end of the squeeze, "
            "4.52 s) each grasping arm's elbow roll target is the clamped solution of a small IK (damped Gauss-Newton "
            "with the exact twist derivative from the MuJoCo rotational Jacobian, steps accepted only if |twist| "
            "decreases) that brings the hand's twist about the pinch axis (world x) back to its grasp-time value, or as "
            "close as the range allows, from the robot's own measured state (never the cube); every control step "
            "(0.04 s); lift_hold to the end, lift_place through the release then withdrawn over the retract",
            "KINEMATIC PREDICTION (wrist_hold_design, computed before any episode): " + design["prediction"],
        ]) + ([f"lowering target: {sc.describe_lower_target(sc.PlaceScript(p, q).lower_target)} (FROZEN "
               "2026-10-02 before any probe or scored episode of the flush-pad variant)"] if place else []),
        "rule": rule, "rule_text": (sc.WRISTHOLD_PROBE_RULE_TEXT if stage in ("probe", "smoke_nohold")
                                    or (stage == "smoke" and not place) else sc.WRISTHOLD_SCORED_RULE_TEXT),
        "clauses_as_applied": list(sc.WRISTHOLD_CLAUSES_AS_APPLIED),
        "seed_evidence": list(sc.WRISTHOLD_SEED_EVIDENCE),
        "base_rule": sc.LIFT_PLACE_RULE if place else sc.LIFT_HOLD_RULE,
        "flush_pad_design": sc.flushpad_design(args.upstream, args.cache_dir, p),
        "wrist_hold_design": design,
        "model_check": sc.flushpad_model_check(model),
    })
    if place:
        out["place_params"] = sc.place_params_dict(q)
    return out


def score_wristhold(args) -> int:
    """Score mode of the W variant. Never overwrites --out; refuses any seed outside its stage."""
    rule, protocol, seeds = wristhold_guard(args)
    nohold = args.wristhold_stage == "smoke_nohold"
    cfg, policy, p, model, slots, pairs = load_wristhold(args, nohold)
    payload = provenance_wristhold(args, cfg, p, model, args.wristhold_stage, rule, protocol)
    if not sc.flushpad_model_ok(payload["model_check"], len(slots)):
        raise SystemExit(f"COOP-WRISTHOLD: model check failed: {payload['model_check']}")
    payload["scored_run"] = args.wristhold_stage in ("probe", "hold", "place")
    payload["episodes"] = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    dt = float(cfg.policy_dt)
    for seed in seeds:
        t0 = time.time()
        ep = sc.run_wristhold_episode(model, slots, pairs, cfg, policy, seed, p, rule=rule)
        ep["wall_s"] = round(time.time() - t0, 1)
        payload["episodes"].append(ep)
        for k, r in enumerate(ep["pairs"]):
            fc = sc.flushpad_clauses(rule, ep, k, dt)
            print(json.dumps({"variant": sc.WRISTHOLD_VARIANT, "stage": args.wristhold_stage, "protocol": protocol,
                              "crew": args.crew, "seed": seed, "pair": r["pair"],
                              "wristhold_success": fc["success"], "wristhold_first_failed": fc["first_failed_check"],
                              "base_rule_success": r["success"], "base_first_failed": r["first_failed_check"],
                              "cube_tilt_max_rad": max([v for v in r["cube_tilt_series_rad"] if v is not None],
                                                       default=None),
                              "lift_peak_m": r["lift_peak_m"], "lift_hold_s": r["lift_hold_s"],
                              "max_robot_tilt_rad": r["max_tilt_rad"], "floor": r["cube_floor_contact"],
                              "hands": [{x: h.get(x) for x in ("side", "twist_at_lift_end_rad", "twist_max_abs_rad",
                                                               "swing_max_rad", "target_clamped_steps",
                                                               "pd_demand_above_cap_steps")}
                                        for h in sc.wristhold_pair_summary(ep, k)],
                              "wall_s": ep["wall_s"]}), flush=True)
        payload["summary"] = sc.wristhold_file_summary(payload)
        args.out.write_text(json.dumps(payload, indent=1, allow_nan=False) + "\n")
    # the line is computed from the JSON as written
    print(sc.wristhold_verdict_line(sc.wristhold_file_summary(json.loads(args.out.read_text())))
          + f" | json={args.out}", flush=True)
    return 0


def wristhold_verdict_cli(argv) -> int:
    """Stage verdict of the W launcher, from score JSONs only (`python -c` entry of cpu_coop_wristhold.sbatch):
    argv = [STAGE, VERDICT_JSON, KEY=PATH, ...] with STAGE
      probe          KEY probe (the probe score JSON)                    -> PROCEED|NEGATIVE|INCOMPLETE|INVALID
      scored         KEYS hold_crew2 hold_crew4 place_crew2 place_crew4  -> PASS|NEGATIVE|INCOMPLETE|INVALID
      scored_not_run KEY probe_verdict (the probe verdict JSON)          -> NOT_RUN
      smoke          KEY step_status (the launcher's step exit statuses, unit tests included), hold_crew2,
                     place_crew4 and nohold_crew2 (smoke score JSONs)     -> SMOKE_PASS|SMOKE_FAIL (+ measurement)
    A missing or unreadable input reads as missing. Refuses to overwrite VERDICT_JSON. Returns 0 whatever the
    verdict: the launcher reads the verdict from the JSON, never from this exit status."""
    if len(argv) < 3:
        raise SystemExit("usage: STAGE VERDICT_JSON KEY=PATH ...")
    stage, out = argv[0], Path(argv[1])
    if out.exists():
        raise SystemExit(f"COOP-WRISTHOLD-VERDICT: REFUSED ({out} exists; verdicts are never overwritten)")
    paths = {}
    for kv in argv[2:]:
        k, _, v = kv.partition("=")
        if not k or not v:
            raise SystemExit(f"COOP-WRISTHOLD-VERDICT: bad input {kv!r} (KEY=PATH)")
        paths[k] = Path(v)

    def read(path):
        try:
            return json.loads(Path(path).read_text())
        except (OSError, ValueError):
            return None

    if stage == "probe":
        v = sc.wristhold_probe_verdict(read(paths["probe"]) if "probe" in paths else None)
    elif stage == "scored":
        v = sc.wristhold_scored_verdict({k: read(paths[k]) if k in paths else None
                                         for k in sc.WRISTHOLD_SCORED_KEYS})
    elif stage == "scored_not_run":
        v = sc.wristhold_not_run_verdict(read(paths["probe_verdict"]) if "probe_verdict" in paths else None)
    elif stage == "smoke":
        status = read(paths["step_status"]) if "step_status" in paths else None
        step_problems = sc.wristhold_smoke_step_problems(status)
        unit_ok = isinstance(status, dict) and type(status.get("pytest")) is int and status["pytest"] == 0
        loaded = {k: read(path) for k, path in paths.items() if k != "step_status"}
        checks = {k: sc.wristhold_smoke_check(pl) for k, pl in loaded.items()}
        missing = [k for k in ("hold_crew2", "place_crew4", "nohold_crew2") if k not in checks]
        ok = (bool(checks) and not missing and all(c["verdict"] == "SMOKE_PASS" for c in checks.values())
              and not step_problems)
        v = {**sc._wristhold_head("smoke"), "verdict": "SMOKE_PASS" if ok else "SMOKE_FAIL",
             "step_status": status, "unit_tests_passed": bool(unit_ok),
             "episodes": [r for c in checks.values() for r in c["episodes"]],
             "problems": [f"step_status: {p}" for p in step_problems] + [f"{k}: no input" for k in missing]
             + [f"{k}: {p}" for k, c in checks.items() for p in c["problems"]], "checks": checks,
             "measurement": sc.wristhold_smoke_measurement(loaded.get("hold_crew2"), loaded.get("nohold_crew2"),
                                                           loaded.get("place_crew4")),
             "note": "pipeline check on a throw-away seed; outcomes are diagnostics, never a verdict"}
    else:
        raise SystemExit(f"COOP-WRISTHOLD-VERDICT: unknown stage {stage!r}")
    v["inputs"] = {k: {"path": str(path), "sha256": sha256(path)} for k, path in paths.items()}
    v["code_sha256"] = {"scripted_carry.py": sha256(REPO / "src/bhl_robust/eval/scripted_carry.py"),
                        "coop_scripted_carry.py": sha256(Path(__file__))}
    v["git_head"] = git_head()
    v["written_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(v, indent=1, allow_nan=False) + "\n")
    print(sc.wristhold_verdict_line(v) + f" | verdict_json={out}", flush=True)
    return 0


def wristhold_main(argv=None) -> int:
    """Command line of the W variant (score mode only; no render)."""
    ap = argparse.ArgumentParser(description="OPT-IN W variant: the flush-pad variant + a kinematic wrist-orientation "
                                             "hold (scripted_carry.WristHoldParams); see the coop-wristhold section")
    ap.add_argument("--deploy", type=Path, required=True)
    ap.add_argument("--upstream", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--crew", type=int, choices=(2, 4), default=2)
    ap.add_argument("--protocol", choices=("lift_hold", "lift_place"), required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--wristhold-stage", choices=sc.WRISTHOLD_STAGES, required=True,
                    help="probe (125-129), hold (20-29), place (30-39), smoke or smoke_nohold (throw-away 140-144)")
    return score_wristhold(ap.parse_args(argv))

# --- end coop-wristhold ---


# ------------------------------------------------------------------ render

class Recorder:
    """Frame hook: overhead-oblique view that follows the cube(s), plus a
    text panel with the live numbers and the labels."""

    def __init__(self, args, model, pairs, policy_dt: float, out_mp4: Path, png_dir: Path):
        from maze_record import FrameSink
        import mujoco

        self.args, self.model, self.pairs = args, model, pairs
        self.lift_hold = getattr(args, "protocol", "carry") == "lift_hold"
        self.lift_place = getattr(args, "protocol", "carry") == "lift_place"
        self.run_steps = np.zeros(len(pairs), dtype=int)    # current >= 5 cm run, per pair
        self.policy_dt = policy_dt
        self.render = not args.no_render
        self.w, self.h = args.width, args.height
        self.side_w = 330
        self.fps = 1.0 / policy_dt
        self.sink = FrameSink(out_mp4, png_dir, self.fps / max(1, args.stride))
        self.frames = 0
        self.look = None
        if self.render:
            self.r = mujoco.Renderer(model, height=self.h, width=self.w)
            self.cam = mujoco.MjvCamera()
            self.cam.distance = args.distance
            self.cam.azimuth, self.cam.elevation = args.azimuth, args.elevation
        self.last = None

    def __call__(self, *, step, t, runner, script, pairs, lift, horiz, carry_on, carry_done, commands,
                 **extra):
        up = np.asarray(lift) >= sc.LIFT_HOLD_RULE["lift_hold_m"]
        self.run_steps = np.where(up, self.run_steps + 1, 0)
        if self.args.stride > 1 and step % self.args.stride:
            return
        centre = np.mean([runner.d.xpos[pr.cube_body] for pr in pairs], axis=0)
        target = np.array([centre[0], centre[1] - 0.15, 0.35])
        self.look = target if self.look is None else 0.9 * self.look + 0.1 * target
        if self.render:
            self.cam.lookat[:] = self.look
            self.r.update_scene(runner.d, camera=self.cam)
            main = self.r.render()
        else:
            main = np.full((self.h, self.w, 3), 40, dtype=np.uint8)
        panel = self._panel(t, script.phase(t), runner, pairs, lift, horiz, carry_on, carry_done)
        # no playback-speed claim in the frame: the same frame is played by the
        # mp4 (0.96x at stride 2) and the GIF (badge states its true speed)
        header = (f"MuJoCo | {self.args.crew} BHL humanoids, {len(pairs)} cube(s) 0.28 m / 0.5 kg | "
                  f"seed {self.args.seed_used} | sim t = {t:5.2f} s | {script.phase(t)}")
        footer = sc.LABEL + " | hand pads added | scripted arm kp 30"
        if self.lift_hold:
            footer = sc.LIFT_HOLD_NOTE + " | " + footer
        if self.lift_place:
            # shorter than NOTE + LABEL so it fits the 960+330 px frame (the note already says
            # "scripted arms, frozen learned gait"; the remaining label part is the oracle cube pose)
            footer = sc.LIFT_PLACE_NOTE + " | oracle cube pose | hand pads added, arm kp 30"
        frame = panels.compose_frame(main, [panel], header, footer, side_w=self.side_w)
        self.last = frame
        self.sink.add(np.ascontiguousarray(frame))
        self.frames += 1

    def _panel(self, t, phase, runner, pairs, lift, horiz, carry_on, carry_done):
        from PIL import Image, ImageDraw
        img = Image.new("RGB", (self.side_w, self.h), panels.PANEL_BG)
        d = ImageDraw.Draw(img)
        top = panels._title(d, self.side_w, "live numbers (simulator)")
        f, fs = panels.load_font(14), panels.load_font(12)
        y = top + 8
        if self.lift_hold:
            return self._panel_lift_hold(d, img, y, f, fs, phase, runner, pairs, lift)
        if self.lift_place:
            return self._panel_lift_place(d, img, y, f, fs, phase, runner, pairs, lift)
        for k, pr in enumerate(pairs):
            F = runner.cube_forces(pr)
            state = "carrying" if carry_on[k] else ("stopped" if carry_done[k] else phase)
            lines = [f"pair {k}: {state}",
                     f"  cube lift   {100 * lift[k]:6.1f} cm",
                     f"  carried     {horiz[k]:6.2f} m",
                     f"  squeeze b/a {F['b']:4.1f} / {F['a']:4.1f} N",
                     f"  robot tilt  {max(runner.tilt(pr.robot_a), runner.tilt(pr.robot_b)):5.2f} rad"]
            for ln in lines:
                d.text((10, y), ln, font=f, fill=panels.TEXT)
                y += 20
            y += 8
        rule = sc.SUCCESS_RULE
        for ln in ["rule (predeclared):", f"  lift >= {100 * rule['lift_peak_m']:.0f} cm,",
                   f"  >= {100 * rule['lift_hold_m']:.0f} cm for {rule['lift_hold_s']:.0f} s,",
                   f"  carried >= {rule['carry_m']:.1f} m, no fall,", "  no floor contact, 30 s",
                   "", "learned: gait (legs + outer arm)", "scripted: grasping arm",
                   "oracle: cube + robot poses", "contact: MuJoCo, no welds"]:
            d.text((10, y), ln, font=fs, fill=panels.DIM)
            y += 17
        return img

    def _panel_lift_hold(self, d, img, y, f, fs, phase, runner, pairs, lift):
        warn = (255, 196, 90)
        for ln in ("cooperative lift and hold \u2014", "carry not achieved"):
            d.text((10, y), ln, font=f, fill=warn)
            y += 20
        y += 6
        for k, pr in enumerate(pairs):
            F = runner.cube_forces(pr)
            lines = [f"pair {k}: {phase}",
                     f"  cube lift   {100 * lift[k]:6.1f} cm",
                     f"  held >= 5cm {self.run_steps[k] * self.policy_dt:6.1f} s",
                     f"  squeeze b/a {F['b']:4.1f} / {F['a']:4.1f} N",
                     f"  robot tilt  {max(runner.tilt(pr.robot_a), runner.tilt(pr.robot_b)):5.2f} rad"]
            for ln in lines:
                d.text((10, y), ln, font=f, fill=panels.TEXT)
                y += 20
            y += 8
        rule = sc.LIFT_HOLD_RULE
        for ln in ["rule (predeclared):", f"  lift >= {100 * rule['lift_peak_m']:.0f} cm,",
                   f"  >= {100 * rule['lift_hold_m']:.0f} cm for {rule['lift_hold_s']:.0f} s continuously,",
                   f"  no fall, no floor contact, {rule['episode_s']:.0f} s",
                   "  no carry: zero velocity command",
                   "", "learned: gait (legs + outer arm)", "scripted: grasping arm",
                   "oracle: cube pose (scoring only)", "contact: MuJoCo, no welds"]:
            d.text((10, y), ln, font=fs, fill=panels.DIM)
            y += 17
        return img

    def _panel_lift_place(self, d, img, y, f, fs, phase, runner, pairs, lift):
        warn = (255, 196, 90)
        for ln in ("cooperative lift, hold and place", "\u2014 carry not achieved"):
            d.text((10, y), ln, font=f, fill=warn)
            y += 20
        y += 6
        for k, pr in enumerate(pairs):
            F = runner.cube_forces(pr)
            c = runner.d.xpos[pr.cube_body]
            off = c[:2] - self.model.geom_pos[pr.plinth_geom][:2]
            lines = [f"pair {k}: {phase}",
                     f"  cube lift   {100 * lift[k]:6.1f} cm",
                     f"  held >= 5cm {self.run_steps[k] * self.policy_dt:6.1f} s",
                     f"  off plinth  {100 * off[0]:+5.1f},{100 * off[1]:+5.1f} cm",
                     f"  squeeze b/a {F['b']:4.1f} / {F['a']:4.1f} N",
                     f"  robot tilt  {max(runner.tilt(pr.robot_a), runner.tilt(pr.robot_b)):5.2f} rad"]
            for ln in lines:
                d.text((10, y), ln, font=f, fill=panels.TEXT)
                y += 20
            y += 8
        rule = sc.LIFT_PLACE_RULE
        for ln in ["rule (predeclared):", f"  lift >= {100 * rule['lift_peak_m']:.0f} cm, >= "
                   f"{100 * rule['lift_hold_m']:.0f} cm for {rule['lift_hold_s']:.0f} s,",
                   f"  ends on the plinth (+/-{100 * rule['final_height_tol_m']:.0f} cm, centre on top),",
                   f"  at rest, released, no fall, no floor, {rule['episode_s']:.0f} s",
                   "", "learned: gait (legs + outer arm)", "scripted: grasping arm",
                   "oracle: cube pose (scoring only)", "contact: MuJoCo, no welds"]:
            d.text((10, y), ln, font=fs, fill=panels.DIM)
            y += 17
        return img

    def hold(self, seconds: float, banner: str, sub: str | None = None):
        if self.last is None:
            return
        from PIL import Image, ImageDraw
        img = Image.fromarray(self.last.copy())
        d = ImageDraw.Draw(img)
        font = panels.load_font(28)
        tw = d.textlength(banner, font=font)
        x, y = (img.size[0] - self.side_w - tw) / 2, 56
        d.rectangle((x - 14, y - 8, x + tw + 14, y + 38), fill=(20, 110, 60))
        d.text((x, y), banner, font=font, fill=(255, 255, 255))
        if sub:
            f2 = panels.load_font(20)
            sw = d.textlength(sub, font=f2)
            x2, y2 = (img.size[0] - self.side_w - sw) / 2, y + 52
            d.rectangle((x2 - 12, y2 - 6, x2 + sw + 12, y2 + 28), fill=(120, 70, 10))
            d.text((x2, y2), sub, font=f2, fill=(255, 255, 255))
        arr = np.asarray(img)
        for _ in range(int(seconds * self.fps / max(1, self.args.stride))):
            self.sink.add(np.ascontiguousarray(arr))

    def close(self):
        if self.render:
            self.r.close()
        return self.sink.close()


def render(args) -> int:
    if getattr(args, "protocol", "carry") == "lift_place":
        return render_lift_place(args)
    lift_hold = getattr(args, "protocol", "carry") == "lift_hold"
    tag = "COOP_LIFT_HOLD_RENDER" if lift_hold else "SCRIPTED_CARRY_RENDER"
    scored = json.loads(Path(args.render_from).read_text())
    s = scored.get("summary", {})
    if scored.get("protocol", "carry") != getattr(args, "protocol", "carry"):
        raise SystemExit(f"--protocol {args.protocol} but {args.render_from} is protocol "
                         f"{scored.get('protocol', 'carry')}")
    if args.pipeline_check_seed is not None:
        # login-node check of the render path only: blank frames, no GIF, any verdict
        if not args.no_render:
            raise SystemExit("--pipeline-check-seed requires --no-render")
        s = dict(s, median_seed=args.pipeline_check_seed, verdict="PIPELINE_CHECK")
        scored = dict(scored, scored_run=True)
    elif not (scored.get("scored_run") and s.get("verdict") == "PASS" and s.get("median_seed") is not None):
        print(f"{tag}=SKIPPED verdict={s.get('verdict')} crew={scored.get('crew')} "
              f"(no success clip is rendered for a run that did not pass)", flush=True)
        return 0
    if int(scored["crew"]) != args.crew:
        raise SystemExit(f"--crew {args.crew} but {args.render_from} is crew {scored['crew']}")
    stem = f"coop_lift_scripted_{args.crew}" if lift_hold else f"carry_scripted_{args.crew}"
    mp4 = args.out_dir / f"{stem}.mp4"
    if lift_hold:
        if scored.get("rule") != sc.LIFT_HOLD_RULE and args.pipeline_check_seed is None:
            raise SystemExit("score JSON rule differs from scripted_carry.LIFT_HOLD_RULE")
        taken = [f for f in (args.gif, Path(args.gif).with_suffix(".json") if args.gif else None,
                             mp4, args.out_dir / f"{stem}.json") if f is not None and Path(f).exists()]
        if taken:
            raise SystemExit(f"COOP-LIFT-HOLD-RENDER: REFUSED ({', '.join(map(str, taken))} exists; "
                             f"outputs are never overwritten)")
    seed = int(s["median_seed"])
    args.seed_used = seed
    want = next((e for e in scored["episodes"] if e["seed"] == seed), scored["episodes"][0])
    cfg, policy, p, model, slots, pairs = load(args)
    if sc.params_dict(p) != scored["params"] or sc.KEYFRAMES_LEFT != scored["keyframes_left"]:
        raise SystemExit("scripted_carry parameters changed since the scored run; re-score first")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rec = Recorder(args, model, pairs, float(cfg.policy_dt), mp4, Path(args.frames_root) / stem)
    t0 = time.time()
    if lift_hold:
        ep = sc.run_episode(model, slots, pairs, cfg, policy, seed, p, frame_hook=rec,
                            rule=sc.LIFT_HOLD_RULE, protocol="lift_hold")
    else:
        ep = sc.run_episode(model, slots, pairs, cfg, policy, seed, p, frame_hook=rec)
    ok = all(r["success"] for r in ep["pairs"])
    if lift_hold:
        hold_s = min(r["lift_hold_s"] for r in ep["pairs"])
        peak = min(r["lift_peak_m"] for r in ep["pairs"])
        rec.hold(1.5, f"LIFTED {100 * peak:.0f} cm, HELD {hold_s:.1f} s  (seed {seed})" if ok
                 else f"NOT REPRODUCED (seed {seed})", sub=sc.LIFT_HOLD_NOTE)
    else:
        done_t = max((r["completion_s"] or 0.0) for r in ep["pairs"])
        rec.hold(1.5, f"CARRIED  {min(r['carry_m'] for r in ep['pairs']):.2f} m  (seed {seed})" if ok
                 else f"NOT REPRODUCED (seed {seed})")
    sink = rec.close()
    if sink["mp4"] is None and sink["frames"] > 0:
        png = Path(sink["png_dir"])
        for enc, extra in (("libx264", ["-preset", "veryfast", "-crf", "20"]), ("mpeg4", ["-q:v", "3"])):
            r = subprocess.run([panels.ffmpeg_exe(), "-y", "-loglevel", "error", "-framerate",
                                str(round(rec.fps / max(1, args.stride))), "-i", str(png / "frame_%04d.png"),
                                "-c:v", enc, *extra, "-pix_fmt", "yuv420p", str(mp4)])
            if r.returncode == 0 and mp4.is_file():
                sink["mp4"] = str(mp4)
                break
    if lift_hold:
        matches = [{"pair": a["pair"], "success": a["success"] == b["success"],
                    "lift_hold_s": a["lift_hold_s"], "scored_lift_hold_s": b["lift_hold_s"]}
                   for a, b in zip(ep["pairs"], want["pairs"])]
    else:
        matches = [{"pair": a["pair"], "success": a["success"] == b["success"],
                    "completion_s": a["completion_s"], "scored_completion_s": b["completion_s"]}
                   for a, b in zip(ep["pairs"], want["pairs"])]
    speed_mp4 = round(round(rec.fps / max(1, args.stride)) / (rec.fps / max(1, args.stride)), 4)
    sim_per_gif = round(args.gif_speed * speed_mp4, 4)
    result = {
        "name": stem, "seed": seed, "crew": args.crew, "reproduced_success": ok,
        "matches_scored_episode": matches, "episode": {k: v for k, v in ep.items() if k != "trace"},
        "frames": sink["frames"], "mp4": sink["mp4"], "mp4_sha256": sha256(mp4) if sink["mp4"] else None,
        "no_render": args.no_render, "video_fps": rec.fps / max(1, args.stride),
        # FrameSink encodes at round(fps): at stride 2 that is 12 fps for 12.5 frames per sim second
        "encoded_fps": round(rec.fps / max(1, args.stride)),
        "playback_speed_mp4": speed_mp4,
        # the frame header states sim time only; clip speeds are these fields and the GIF badge
        "frame_header_speed": None,
        "camera": {"type": "free, follows mean cube xy", "distance": args.distance,
                   "azimuth": args.azimuth, "elevation": args.elevation, "size": [args.width, args.height]},
        "scored_json": os.path.relpath(Path(args.render_from).resolve(), REPO),
        "scored_json_sha256": sha256(args.render_from),
        "label": sc.LABEL, "label_detail": sc.LIFT_HOLD_LABEL_DETAIL if lift_hold else sc.LABEL_DETAIL,
        "wall_seconds": round(time.time() - t0, 1),
    }
    if lift_hold:
        result.update({"protocol": "lift_hold", "note": sc.LIFT_HOLD_NOTE})
    if args.gif and result["mp4"] and ok and not args.no_render:
        g = panels.write_gif(mp4, args.gif, fps=args.gif_fps, speed=args.gif_speed, width=args.gif_width,
                             badge=f"GIF {sim_per_gif:.2f}x")
        side = Path(args.gif).with_suffix(".json")
        gate = sc.enforce_gif_budget(g, args.gif, side if lift_hold else None)
        result["gif"] = g
        if not gate["kept"]:
            # over the 5 MiB budget: deleted, never left in docs/gifs, and no sidecar
            result["gif_deleted_over_budget"] = gate["deleted"]
        else:
            common = {
                "output": os.path.relpath(Path(args.gif).resolve(), REPO), "output_sha256": sha256(args.gif),
                "output_mb": g["mb"], "within_budget": g["within_budget"],
                "source_clip": os.path.relpath(mp4.resolve(), REPO), "source_sha256": result["mp4_sha256"],
                "evidence": result["scored_json"], "evidence_sha256": result["scored_json_sha256"],
            }
            speeds = {
                "playback_speed": sim_per_gif, "gif_speed_setting": args.gif_speed,
                "playback_speed_mp4": speed_mp4, "sim_seconds_per_gif_second": sim_per_gif,
                "frame_header_speed": None, "badge": g.get("badge"),
                "gif": g,
                "checkpoint": scored["checkpoint"], "checkpoint_sha256": scored["checkpoint_sha256"],
                "module_sha256": scored["module_sha256"],
            }
            if lift_hold:
                sidecar = {
                    **common, "protocol": "lift_hold", "note": sc.LIFT_HOLD_NOTE,
                    "caption": (f"{sc.LIFT_HOLD_NOTE}: {args.crew} BHL humanoids lift a 0.28 m / 0.5 kg cube "
                                f"together and hold it (seed {seed}: lifted {100 * peak:.1f} cm, held >= 5 cm "
                                f"for {hold_s:.1f} s; median hold duration of the successful seeds). "
                                f"{sc.LABEL}. Hand pads added; grasping arm scripted at kp 30; the robots "
                                f"never walk (zero velocity command)."),
                    "label": sc.LABEL, "label_detail": sc.LIFT_HOLD_LABEL_DETAIL,
                    "modelling_choices": scored["modelling_choices"],
                    "success_rule": scored["rule"], "summary": s,
                    "episode": [{k: r[k] for k in ("pair", "success", "lift_peak_m", "lift_hold_s",
                                                   "hold_start_s", "hold_end_s", "max_tilt_rad",
                                                   "cube_floor_contact")} for r in ep["pairs"]],
                    **speeds,
                }
            else:
                sidecar = {
                    **common,
                    "caption": (f"{args.crew} BHL humanoids lift a 0.28 m / 0.5 kg cube together and carry it "
                                f"(seed {seed}, median completion of the successful seeds). "
                                f"{sc.LABEL}. Hand pads added; grasping arm scripted at kp 30."),
                    "label": sc.LABEL, "label_detail": sc.LABEL_DETAIL,
                    "modelling_choices": scored["modelling_choices"],
                    "success_rule": scored["rule"], "summary": s,
                    "episode": [{k: r[k] for k in ("pair", "success", "lift_peak_m", "lift_hold_s", "carry_m",
                                                   "max_tilt_rad", "completion_s")} for r in ep["pairs"]],
                    **speeds,
                }
            side.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n")
    elif args.gif:
        result["gif"] = None
        result["gif_skipped"] = ("--no-render" if args.no_render else
                                 "re-simulated episode did not succeed" if not ok else "no mp4")
    (args.out_dir / f"{stem}.json").write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")
    gif_path = result["gif"]["gif"] if (result.get("gif") and not result.get("gif_deleted_over_budget")) else None
    if lift_hold:
        print(f"{tag} crew={args.crew} seed={seed} reproduced={ok} hold_s={hold_s:.2f} "
              f"frames={sink['frames']} gif={gif_path} | {sc.LIFT_HOLD_NOTE}", flush=True)
    else:
        print(f"{tag} crew={args.crew} seed={seed} reproduced={ok} "
              f"completion_s={done_t:.2f} frames={sink['frames']} gif={gif_path}", flush=True)
    return 0


def render_lift_place(args) -> int:
    """Render mode of the lift-hold-place protocol: only on a scored PASS, the
    median-by-hold successful seed, nothing ever overwritten."""
    tag = "COOP_LIFT_PLACE_RENDER"
    scored = json.loads(Path(args.render_from).read_text())
    s = scored.get("summary", {})
    if scored.get("protocol") != "lift_place":
        raise SystemExit(f"--protocol lift_place but {args.render_from} is protocol "
                         f"{scored.get('protocol', 'carry')}")
    if args.pipeline_check_seed is not None:
        # login-node check of the render path only: blank frames, no GIF, any verdict
        if not args.no_render:
            raise SystemExit("--pipeline-check-seed requires --no-render")
        if 0 <= args.pipeline_check_seed < 20:
            raise SystemExit("--pipeline-check-seed must not be a scored seed (0-19)")
        s = dict(s, median_seed=args.pipeline_check_seed, verdict="PIPELINE_CHECK")
    else:
        line = sc.lift_place_verdict_line(scored, scored.get("crew"))
        # the seed is recomputed from the episodes, never taken from the stored summary
        median = sc.median_seed_by_hold([e for e in scored.get("episodes", [])
                                         if e["seed"] in sc.LIFT_PLACE_RULE["seeds"]])
        s = dict(s, median_seed=median)
        if not (line.split(" | ")[0].endswith(": PASS") and median is not None):
            print(f"{tag}=SKIPPED verdict={line.split(' | ')[0].split(': ')[-1]} crew={scored.get('crew')} "
                  f"(no success clip is rendered for a run that did not pass)", flush=True)
            return 0
        if scored.get("rule") != sc.LIFT_PLACE_RULE:
            raise SystemExit("score JSON rule differs from scripted_carry.LIFT_PLACE_RULE")
    if int(scored["crew"]) != args.crew:
        raise SystemExit(f"--crew {args.crew} but {args.render_from} is crew {scored['crew']}")
    stem = f"coop_lift_place_scripted_{args.crew}"
    mp4 = args.out_dir / f"{stem}.mp4"
    taken = [f for f in (args.gif, Path(args.gif).with_suffix(".json") if args.gif else None,
                         mp4, args.out_dir / f"{stem}.json") if f is not None and Path(f).exists()]
    if taken:
        raise SystemExit(f"COOP-LIFT-PLACE-RENDER: REFUSED ({', '.join(map(str, taken))} exists; "
                         f"outputs are never overwritten)")
    seed = int(s["median_seed"])
    args.seed_used = seed
    want = next((e for e in scored["episodes"] if e["seed"] == seed), None)
    cfg, policy, p, model, slots, pairs = load(args)
    q = sc.PlaceParams()
    if (sc.params_dict(p) != scored["params"] or sc.KEYFRAMES_LEFT != scored["keyframes_left"]
            or json.loads(json.dumps(sc.place_params_dict(q))) != scored["place_params"]):
        raise SystemExit("scripted_carry parameters changed since the scored run; re-score first")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rec = Recorder(args, model, pairs, float(cfg.policy_dt), mp4, Path(args.frames_root) / stem)
    t0 = time.time()
    rule = sc.LIFT_PLACE_RULE
    if args.seconds is not None:
        if args.pipeline_check_seed is None:
            raise SystemExit("--seconds is only for --pipeline-check-seed")
        rule = dict(rule, episode_s=float(args.seconds))
    ep = sc.run_place_episode(model, slots, pairs, cfg, policy, seed, p, q=q, frame_hook=rec, rule=rule)
    ok = all(r["success"] for r in ep["pairs"])
    hold_s = min(r["lift_hold_s"] for r in ep["pairs"])
    peak = min(r["lift_peak_m"] for r in ep["pairs"])
    rec.hold(1.5, f"LIFTED {100 * peak:.0f} cm, HELD {hold_s:.1f} s, PLACED  (seed {seed})" if ok
             else f"NOT REPRODUCED (seed {seed})", sub="cooperative lift, hold and place \u2014 carry not achieved")
    sink = rec.close()
    if sink["mp4"] is None and sink["frames"] > 0:
        png = Path(sink["png_dir"])
        for enc, extra in (("libx264", ["-preset", "veryfast", "-crf", "20"]), ("mpeg4", ["-q:v", "3"])):
            r = subprocess.run([panels.ffmpeg_exe(), "-y", "-loglevel", "error", "-framerate",
                                str(round(rec.fps / max(1, args.stride))), "-i", str(png / "frame_%04d.png"),
                                "-c:v", enc, *extra, "-pix_fmt", "yuv420p", str(mp4)])
            if r.returncode == 0 and mp4.is_file():
                sink["mp4"] = str(mp4)
                break
    matches = [{"pair": a["pair"], "success": a["success"] == b["success"],
                "lift_hold_s": a["lift_hold_s"], "scored_lift_hold_s": b["lift_hold_s"],
                "final_offset_xy_m": a["final_offset_xy_m"], "scored_final_offset_xy_m": b["final_offset_xy_m"]}
               for a, b in zip(ep["pairs"], want["pairs"])] if want else None
    speed_mp4 = round(round(rec.fps / max(1, args.stride)) / (rec.fps / max(1, args.stride)), 4)
    sim_per_gif = round(args.gif_speed * speed_mp4, 4)
    result = {
        "name": stem, "protocol": "lift_place", "note": sc.LIFT_PLACE_NOTE, "seed": seed, "crew": args.crew,
        "reproduced_success": ok, "matches_scored_episode": matches,
        "episode": {k: v for k, v in ep.items() if k != "trace"},
        "frames": sink["frames"], "mp4": sink["mp4"], "mp4_sha256": sha256(mp4) if sink["mp4"] else None,
        "no_render": args.no_render, "video_fps": rec.fps / max(1, args.stride),
        "encoded_fps": round(rec.fps / max(1, args.stride)), "playback_speed_mp4": speed_mp4,
        "frame_header_speed": None,
        "camera": {"type": "free, follows mean cube xy", "distance": args.distance,
                   "azimuth": args.azimuth, "elevation": args.elevation, "size": [args.width, args.height]},
        "scored_json": os.path.relpath(Path(args.render_from).resolve(), REPO),
        "scored_json_sha256": sha256(args.render_from),
        "label": sc.LABEL, "label_detail": sc.LIFT_PLACE_LABEL_DETAIL,
        "wall_seconds": round(time.time() - t0, 1),
    }
    if args.gif and result["mp4"] and ok and not args.no_render:
        g = panels.write_gif(mp4, args.gif, fps=args.gif_fps, speed=args.gif_speed, width=args.gif_width,
                             badge=f"GIF {sim_per_gif:.2f}x")
        side = Path(args.gif).with_suffix(".json")
        gate = sc.enforce_gif_budget(g, args.gif, side)
        result["gif"] = g
        if not gate["kept"]:
            result["gif_deleted_over_budget"] = gate["deleted"]
        else:
            sidecar = {
                "output": os.path.relpath(Path(args.gif).resolve(), REPO), "output_sha256": sha256(args.gif),
                "output_mb": g["mb"], "within_budget": g["within_budget"],
                "source_clip": os.path.relpath(mp4.resolve(), REPO), "source_sha256": result["mp4_sha256"],
                "evidence": result["scored_json"], "evidence_sha256": result["scored_json_sha256"],
                "protocol": "lift_place", "note": sc.LIFT_PLACE_NOTE,
                "caption": (f"{sc.LIFT_PLACE_NOTE}: {args.crew} BHL humanoids lift a 0.28 m / 0.5 kg cube "
                            f"together, hold it and set it back on its plinth (seed {seed}: lifted "
                            f"{100 * peak:.1f} cm, held >= 5 cm for {hold_s:.1f} s, placed and released; "
                            f"median hold of the successful seeds). {sc.LABEL}. Hand pads added; grasping "
                            f"arm scripted at kp 30; the robots never walk. Plays at {sim_per_gif:.2f}x."),
                "label": sc.LABEL, "label_detail": sc.LIFT_PLACE_LABEL_DETAIL,
                "learned": scored.get("learned"), "scripted": scored.get("scripted"),
                "oracle": scored.get("oracle"),
                "modelling_choices": scored["modelling_choices"], "place_params": scored["place_params"],
                "success_rule": scored["rule"], "summary": s,
                "episode": [{k: r[k] for k in ("pair", "success", "lift_peak_m", "lift_hold_s", "hold_start_s",
                                               "hold_end_s", "max_tilt_rad", "cube_floor_contact", "final_dz_m",
                                               "final_speed_mps", "final_offset_xy_m", "final_robot_contact")}
                            for r in ep["pairs"]],
                "playback_speed": sim_per_gif, "gif_speed_setting": args.gif_speed,
                "playback_speed_mp4": speed_mp4, "sim_seconds_per_gif_second": sim_per_gif,
                "frame_header_speed": None, "badge": g.get("badge"), "gif": g,
                "checkpoint": scored["checkpoint"], "checkpoint_sha256": scored["checkpoint_sha256"],
                "module_sha256": scored["module_sha256"],
            }
            side.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n")
    elif args.gif:
        result["gif"] = None
        result["gif_skipped"] = ("--no-render" if args.no_render else
                                 "re-simulated episode did not succeed" if not ok else "no mp4")
    (args.out_dir / f"{stem}.json").write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")
    gif_path = result["gif"]["gif"] if (result.get("gif") and not result.get("gif_deleted_over_budget")) else None
    print(f"{tag} crew={args.crew} seed={seed} reproduced={ok} hold_s={hold_s:.2f} "
          f"frames={sink['frames']} gif={gif_path} | {sc.LIFT_PLACE_NOTE}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deploy", type=Path, required=True)
    ap.add_argument("--upstream", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--crew", type=int, choices=(2, 4), default=2)
    ap.add_argument("--protocol", choices=sc.PROTOCOLS, default="carry",
                    help="carry (default, SUCCESS_RULE), lift_hold (LIFT_HOLD_RULE, no carry phase) or "
                         "lift_place (LIFT_PLACE_RULE, seeds 10-19 only)")
    ap.add_argument("--seeds", default="0-9", help="e.g. 0-9 or 100,101")
    ap.add_argument("--seconds", type=float, default=None,
                    help="shorten episodes (pipeline check only; the verdict becomes PIPELINE_CHECK)")
    ap.add_argument("--out", type=Path, help="score mode: JSON output")
    ap.add_argument("--render-from", type=Path, help="render mode: a score JSON")
    ap.add_argument("--out-dir", type=Path, help="render mode: mp4 + result JSON directory")
    ap.add_argument("--gif", type=Path)
    ap.add_argument("--frames-root", default=os.path.join(os.environ.get("TMPDIR", "/tmp"), "scripted-carry-frames"))
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--distance", type=float, default=3.4)
    ap.add_argument("--azimuth", type=float, default=135.0)
    ap.add_argument("--elevation", type=float, default=-38.0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--pipeline-check-seed", type=int, default=None,
                    help="with --no-render: exercise the render path on this seed regardless of verdict")
    ap.add_argument("--gif-speed", type=float, default=2.0)
    ap.add_argument("--gif-fps", type=int, default=8)
    ap.add_argument("--gif-width", type=int, default=860)
    ap.add_argument("--flushpad-stage", choices=sc.FLUSHPAD_STAGES, default=None, help=FLUSHPAD_STAGE_HELP)
    args = ap.parse_args()
    if args.flushpad_stage is not None:
        return score_flushpad(args)
    if args.render_from:
        if not args.out_dir:
            ap.error("--render-from needs --out-dir")
        return render(args)
    if not args.out:
        ap.error("score mode needs --out")
    return score(args)


if __name__ == "__main__":
    sys.exit(main())
