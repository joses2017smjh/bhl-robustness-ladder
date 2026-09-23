"""Render one Mission 7 route episode to MP4 with a hashed sidecar.

Re-runs a probe configuration deterministically (same DebugEnv, same
RouteHandoffController) and captures a tracking-camera frame after every
25 Hz physics window.  The MP4 is a rendering of a known result; the sidecar
records the configuration, the outcome of THIS run, and the sha256 of the
result record it reproduces, so a clip can never be mistaken for evidence on
its own.  Needs a GPU node with EGL (MUJOCO_GL=egl) and ffmpeg.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from bhl_robust.eval.video import EpisodeRecorder
from bhl_robust.mission.approach_debug import DebugEnv
from mission7_route_handoff_probe import RouteHandoffController


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="output MP4 path (inside the repo)")
    p.add_argument("--stage", choices=("doors", "transport"), required=True)
    p.add_argument("--index", type=int, required=True)
    p.add_argument("--caption", default="")
    p.add_argument("--max-seconds", type=float, default=None, help="stop recording early (render only)")
    p.add_argument("--evidence", type=Path, default=None, help="result record this run reproduces")
    p.add_argument("--handoff", choices=("switch", "early"), default="early")
    p.add_argument("--rejoin-fix", choices=("none", "forward_pulse", "prev_actions_reset"), default="none")
    p.add_argument("--stall-min-s", type=float, default=.8)
    p.add_argument("--stage-activate", action="store_true")
    p.add_argument("--stage-wait-open", type=float, default=2.0)
    p.add_argument("--pre-point", type=float, default=.30)
    p.add_argument("--cross-clear", type=float, default=None)
    p.add_argument("--cross-kick", action="store_true")
    p.add_argument("--align-yaw", action="store_true")
    p.add_argument("--rejoin-advance", action="store_true")
    p.add_argument("--exit-ramp", type=float, default=0.)
    p.add_argument("--exit-ramp-center", action="store_true")
    a = p.parse_args()
    a.repo = a.repo.resolve(); a.out = a.out.resolve()
    if not a.out.is_relative_to(a.repo):
        p.error("output must remain inside the repository")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    env = DebugEnv(a.repo, a.out.parent / f"{a.out.stem}-cache", stage=a.stage, split="validation", seed=1000)
    env.reset(a.index)
    controller = RouteHandoffController(
        env, handoff_mode=a.handoff, rejoin_fix=a.rejoin_fix, stall_min_s=a.stall_min_s,
        stage_activate=a.stage_activate, stage_wait_open_s=a.stage_wait_open, pre_point_m=a.pre_point,
        cross_clear_m=a.cross_clear, cross_kick=a.cross_kick, align_yaw=a.align_yaw,
        rejoin_advance=a.rejoin_advance, exit_ramp_s=a.exit_ramp, exit_ramp_center=a.exit_ramp_center)
    recorder = EpisodeRecorder(env.model, a.out, fps=25.0, caption=a.caption, track_body=f"{env.slot.prefix}base")
    runner_step = env.runner.step

    def recorded_step(targets):
        runner_step(targets)
        flash = "fall" if env.runner.tilt(0) >= .78 else None
        recorder.capture(env.runner.d, flash=flash)

    env.runner.step = recorded_step
    done = False
    while not done:
        env.runner.contact_trace = []
        action = controller.action()
        _, _, done, _ = env.step(action)
        controller.observe_contacts(env.runner.contact_trace)
        controller.observe_step(action)
        if a.max_seconds is not None and env.runner.d.time >= a.max_seconds:
            break
    err = recorder.close()
    row = env.metrics()
    sidecar = {
        "output": str(a.out.relative_to(a.repo)), "output_sha256": sha256(a.out) if a.out.exists() else None,
        "renderer": "MuJoCo EpisodeRecorder, tracking camera, 25 fps", "ffmpeg_error": err,
        "stage": a.stage, "layout_index": a.index, "split": "validation",
        "configuration": {k: v for k, v in vars(a).items() if k not in ("repo", "out", "evidence", "caption")},
        "outcome_of_this_run": {"success": bool(row["success"]), "failure": row["failure"], "elapsed_s": row["elapsed_s"],
                                "recording_stopped_early": a.max_seconds is not None and row["elapsed_s"] >= a.max_seconds and row["failure"] is None and not row["success"]},
        "evidence": str(a.evidence) if a.evidence else None,
        "evidence_sha256": sha256(a.evidence) if a.evidence and a.evidence.exists() else None,
        "caption": a.caption,
        "scope": "A rendering of one development layout under one controller configuration; not a route-rate claim and not a replay-gate result.",
        "git_commit": subprocess.check_output(["git", "-C", str(a.repo), "rev-parse", "HEAD"], text=True).strip(),
    }
    a.out.with_suffix(".json").write_text(json.dumps(sidecar, indent=2) + "\n")
    print(json.dumps({"out": sidecar["output"], "outcome": sidecar["outcome_of_this_run"], "ffmpeg_error": err}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
