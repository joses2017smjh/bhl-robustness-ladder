"""G-T3: verdict on the Tier 3 gate -- 22 DoF on stairs, depth on, arm deviation off.

Three rows, two real iterations each at 4,096 envs, logs in one directory:

  ppo-stairs.log     rsl-rl PPO through slurm/inner/train.sh
  mappo-stairs.log   skrl MAPPO, limb4 (arm_left 5, arm_right 5, leg_left 6, leg_right 6)
  limb1-stairs.log   skrl IPPO, one agent owning all 22 joints

Passes only if, from what each run printed about itself:

  * the reward manager built without `joint_deviation_shoulder` and
    `joint_deviation_elbow`, and with `joint_deviation_hip` (so the table was
    printed at all) -- the ablation the first block's PPO control never got;
  * the observation is 331 wide (75 of proprioception + 256 of depth) on the
    skrl rows, with the depth term, and PPO's actor takes 331 inputs;
  * limb4 builds four agents of 5 / 5 / 6 / 6 whose first joints are the left
    shoulder, the right shoulder, the left hip and the right hip, and limb1 one
    agent of 22;
  * the skrl rows reach 48 of 48 timesteps on the rsl-rl settings and log
    terrain level; PPO logs two iterations.

Usage: arms_tier3_check.py <gate_dir> <repo>
"""

from __future__ import annotations

import ast
import glob
import re
import sys
from pathlib import Path

OBS = 331
TAG = "Info / Curriculum/terrain_levels"
LIMB4 = {"arm_left": 5, "arm_right": 5, "leg_left": 6, "leg_right": 6}
FIRST = {"arm_left": "arm_left_shoulder_pitch_joint", "arm_right": "arm_right_shoulder_pitch_joint",
         "leg_left": "leg_left_hip_roll_joint", "leg_right": "leg_right_hip_roll_joint"}


def tags_of(log_dir: Path) -> set[str]:
    from tensorboard.backend.event_processing import event_accumulator

    events = sorted(glob.glob(str(log_dir / "**" / "events.out.tfevents.*"), recursive=True))
    if not events:
        return set()
    ea = event_accumulator.EventAccumulator(events[-1], size_guidance={"scalars": 0})
    ea.Reload()
    return set(ea.Tags()["scalars"])


def reward_table(text: str, problems: list[str]) -> None:
    if "joint_deviation_hip" not in text:
        problems.append("no reward-term table in the log")
    for term in ("joint_deviation_shoulder", "joint_deviation_elbow"):
        if re.search(rf"\|\s*{term}\s*\|", text):
            problems.append(f"{term} still active")


def check_marl(text: str, repo: Path, part: str, problems: list[str]) -> None:
    m = re.search(r"\[marl-gate\] task=(\S+) partition=(\S+) n_dof=(\d+) agents=(\d+) "
                  r"act=(\{.*?\}) obs=(\d+) state=(\d+) critic=(\w+) policy_terms=(\[.*?\]) joints=(\{.*\})", text)
    if not m:
        problems.append("no [marl-gate] env line")
        return
    _, got, n_dof, agents, act, obs, state, critic, terms, joints = m.groups()
    want = LIMB4 if part == "limb4" else {"whole": 22}
    if got != part or int(n_dof) != 22 or ast.literal_eval(act) != want:
        problems.append(f"partition {got} n_dof {n_dof} act {act}")
    if int(obs) != OBS or int(state) != OBS + 3 or critic != "privileged" or "depth" not in ast.literal_eval(terms):
        problems.append(f"obs {obs} state {state} terms {terms}")
    if part == "limb4":
        j = ast.literal_eval(joints)
        bad = {a: j.get(a, [""])[0] for a in FIRST if j.get(a, [""])[0] != FIRST[a]}
        if bad:
            problems.append(f"joint order wrong: {bad}")
    t = re.search(r"\[marl-gate\] trainer_agent=(\w+) log_dir=(\S+)", text)
    if not t or t.group(1) != ("MAPPO" if part == "limb4" else "IPPO"):
        problems.append("trainer missing or wrong")
    else:
        if TAG not in tags_of(repo / t.group(2)):
            problems.append("no terrain level in the event file")
    if not re.search(r"48/48 \[", text):
        problems.append("did not reach 48/48")
    h = re.search(r"\[marl-gate\] hparams=(\S+) actor=(\[.*?\])", text)
    if not h or h.group(1) != "rsl" or ast.literal_eval(h.group(2)) != [256, 128, 128]:
        problems.append("hparams not rsl-matched")


def check_ppo(text: str, problems: list[str]) -> None:
    if not re.search(rf"Linear\(in_features={OBS},", text):
        problems.append(f"PPO actor does not take {OBS} inputs")
    if not re.search(r"Learning iteration 1/2", text):
        problems.append("PPO did not log its second iteration")


def main() -> None:
    gate_dir = Path(sys.argv[1])
    repo = Path(sys.argv[2] if len(sys.argv) > 2 else ".")
    rows = {"ppo-stairs": None, "mappo-stairs": "limb4", "limb1-stairs": "limb1"}
    ok_all = True
    for name, part in rows.items():
        log = gate_dir / f"{name}.log"
        problems: list[str] = []
        if not log.exists():
            problems.append("no log")
        else:
            text = log.read_text(errors="replace")
            reward_table(text, problems)
            if "Tier 3 ids NOT registered" in text:
                problems.append("task failed to register")
            if part is None:
                check_ppo(text, problems)
            else:
                check_marl(text, repo, part, problems)
        ok_all &= not problems
        print(f"  {name:14} {'PASS' if not problems else 'FAIL'}  {'; '.join(problems) or 'ok'}")
    print(f"G-T3 {'PASS' if ok_all else 'FAIL'}")


if __name__ == "__main__":
    main()
