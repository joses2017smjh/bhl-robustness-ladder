"""Small route-to-PlateStage handoff probe on selected validation layouts.

This probe changes one factor relative to the completed route evaluation:
when the unchanged PlateSafeRouteController enters its existing ``switch``
phase, the guarded PlateStage command stream takes over until that bounded
stage completes.  Route navigation, plate geometry, activation semantics,
termination, and the fall predicate remain unchanged.  The selected layout
indices are intentionally small and are supplied explicitly by the submitter.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bhl_robust.mission.approach_debug import DebugEnv, PlateSafeRouteController
from mission7_plate_stage import PlateStage


def physical_to_action(command, base_action):
    scales = np.asarray([0.4, 0.35, 0.4], dtype=float)
    action = np.asarray(base_action, dtype=float).copy()
    action[:3] = np.arctanh(np.clip(np.asarray(command, dtype=float) / scales, -0.999999, 0.999999))
    return action


class RouteHandoffController:
    """PlateSafe routing with only the existing switch-to-stage handoff changed."""

    def __init__(self, env):
        self.env = env
        self.route = PlateSafeRouteController(env, contact_hold_s=1.0)
        self.stage = PlateStage(env)
        self.handoff_seen = False
        self.phase_history = []

    def action(self):
        base_action = self.route.action()
        recorded = np.tanh(base_action[:3]) * np.asarray([0.4, 0.35, 0.4])
        now = float(self.env.runner.d.time)
        if self.env.phase == "switch" or self.stage.phase != "recorded":
            if not self.handoff_seen:
                self.handoff_seen = True
            command, phase = self.stage.command(recorded)
            if phase != "recorded":
                self.env.phase = "stage_" + phase
                self.phase_history.append((now, self.env.phase))
                return physical_to_action(command, base_action)
        return base_action


def run(args):
    args.out.mkdir(parents=True, exist_ok=True)
    stages = (args.stage,) if args.stage != "both" else ("doors", "transport")
    indices = [int(value) for value in args.indices.split(",") if value.strip()]
    rows = []
    for stage in stages:
        for index in indices:
            env = DebugEnv(args.repo, args.out / f"{stage}-{index}-cache",
                           stage=stage, split="validation", seed=1000)
            env.reset(index)
            controller = RouteHandoffController(env)
            while True:
                _, _, done, _ = env.step(controller.action())
                if done:
                    break
            row = env.metrics()
            row.update(
                layout_index=index,
                stage=stage,
                controller="PlateSafeRouteController plus guarded PlateStage on switch handoff",
                route_plate_contacts_seen=sorted(set(controller.route.plate_contacts_seen)),
                guarded_stage_activated=controller.handoff_seen,
                guarded_stage_history=controller.stage.history,
                guarded_stage_phase_history=controller.phase_history,
                guarded_stage_completed=bool(controller.stage.done),
            )
            rows.append(row)
            (args.out / f"{stage}-{index}.json").write_text(json.dumps(row, indent=2) + "\n")
    result = {
        "complete": True,
        "status": "COMPLETED_TARGETED_HANDOFF_PROBE",
        "controller": "PlateSafeRouteController plus guarded PlateStage on switch handoff",
        "selected_indices": indices,
        "stages": list(stages),
        "geometry_unchanged": True,
        "activation_semantics_unchanged": True,
        "fall_predicate_unchanged": True,
        "route_episode_count": len(rows),
        "guarded_stage_activations": sum(row["guarded_stage_activated"] for row in rows),
        "guarded_stage_completions": sum(row["guarded_stage_completed"] for row in rows),
        "episodes": rows,
    }
    (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "status": result["status"],
        "episodes": len(rows),
        "guarded_stage_activations": result["guarded_stage_activations"],
        "guarded_stage_completions": result["guarded_stage_completions"],
        "out": str(args.out),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage", choices=("doors", "transport", "both"), required=True)
    parser.add_argument("--indices", required=True, help="comma-separated validation layout indices")
    args = parser.parse_args()
    args.repo = args.repo.resolve()
    args.out = args.out.resolve()
    if not args.out.is_relative_to(args.repo):
        parser.error("output must remain inside repository")
    run(args)
