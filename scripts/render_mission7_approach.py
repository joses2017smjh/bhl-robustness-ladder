"""Render one privileged Approach episode (Mission 7) to MP4 with a hashed sidecar.

Deterministic rerun of one balanced-matrix layout under one controller arm from
scripts/mission7_approach_followup.py, captured after every 25 Hz physics
window.  The sidecar names the arm, layout, split, this run's outcome and the
sha256 of the result record it reproduces.  GPU node with EGL and ffmpeg.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from bhl_robust.eval.video import EpisodeRecorder
from bhl_robust.mission.approach_debug import DebugEnv
from mission7_approach_followup import CONTROLLERS, balanced_layouts, controller_note, run_controller


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--controller", choices=sorted(CONTROLLERS), required=True)
    p.add_argument("--split", choices=("test", "validation"), default="test")
    p.add_argument("--index", type=int, required=True)
    p.add_argument("--caption", default="")
    p.add_argument("--evidence", type=Path, default=None)
    p.add_argument("--replay-order", default=None,
                   help="direction key such as '-1,+0': reset through the balanced-matrix selection in the "
                        "evaluator's order up to the target layout so the RNG-driven spawn noise matches the "
                        "evaluated episode (a fresh env draws the first sample and does not reproduce it)")
    a = p.parse_args()
    a.repo = a.repo.resolve(); a.out = a.out.resolve()
    if not a.out.is_relative_to(a.repo):
        p.error("output must remain inside the repository")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    env = DebugEnv(a.repo, a.out.parent / f"{a.out.stem}-cache", stage="approach", split=a.split, seed=2300, approach_distance=.65)
    replayed = []
    if a.replay_order:
        for entry in balanced_layouts()[a.replay_order]:
            if entry["split"] != a.split:
                continue
            env.reset(entry["index"]); replayed.append(entry["index"])
            if entry["index"] == a.index:
                break
        if not replayed or replayed[-1] != a.index:
            raise SystemExit(f"layout {a.index} is not in the {a.split} part of direction {a.replay_order}")
    else:
        env.reset(a.index)
    controller = CONTROLLERS[a.controller](env)
    recorder = EpisodeRecorder(env.model, a.out, fps=25.0, caption=a.caption, track_body=f"{env.slot.prefix}base")
    runner_step = env.runner.step

    def recorded_step(targets):
        runner_step(targets)
        recorder.capture(env.runner.d, flash="fall" if env.runner.tilt(0) >= .78 else None)

    env.runner.step = recorded_step
    row = run_controller(env, controller)
    err = recorder.close()
    sidecar = {
        "output": str(a.out.relative_to(a.repo)), "output_sha256": sha256(a.out) if a.out.exists() else None,
        "renderer": "MuJoCo EpisodeRecorder, tracking camera, 25 fps", "ffmpeg_error": err,
        "task": "Mission 7 privileged Approach, 0.65 m spawn, balanced-matrix layout", "split": a.split, "layout_index": a.index,
        "controller_arm": a.controller, "controller": controller_note(a.controller, controller),
        "reset_sequence": replayed or [a.index],
        "reproduces_evaluated_episode": bool(replayed),
        "outcome_of_this_run": {"success": bool(row["success"]), "failure": row["failure"], "collisions": row["collisions"], "elapsed_s": row["elapsed_s"]},
        "evidence": str(a.evidence) if a.evidence else None, "evidence_sha256": sha256(a.evidence) if a.evidence and a.evidence.exists() else None,
        "caption": a.caption, "labels": "privileged pose (oracle), frozen learned gait, free-standing, native MuJoCo",
        "scope": "One layout under one controller arm; the gate result is the 64-episode matrix in the evidence file, not this clip.",
        "git_commit": subprocess.check_output(["git", "-C", str(a.repo), "rev-parse", "HEAD"], text=True).strip(),
    }
    a.out.with_suffix(".json").write_text(json.dumps(sidecar, indent=2) + "\n")
    print(json.dumps({"out": sidecar["output"], "outcome": sidecar["outcome_of_this_run"], "ffmpeg_error": err}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
