"""G-B4t: verdict on the Tier 1 terrain gate, from the logs and event files it left.

The gate job trains every Tier 1 MARL row for two real iterations at the full
4,096 envs. This reads what each run *printed about itself* and what it *wrote
to TensorBoard*, and passes only if, for every row:

  (i)   the constructed env is the 12-DoF biped, split as asked: legs2 gives two
        agents of 6 whose first joints are the left and the right hip, limb1
        gives one agent of 12;
  (ii)  the algorithm reached the trainer -- skrl built a MAPPO or IPPO object
        matching the row, not the default;
  (iii) every agent sees the full single-agent policy observation, depth term
        included, at the same width on all four terrains, and MAPPO's critic
        state is that same vector;
  (iv)  training progressed (48 of 48 timesteps) and the event file carries
        `Info / Curriculum/terrain_levels` -- the work order's primary metric,
        which no skrl run logged before `environment_info="log"` and the
        float-to-tensor conversion in `limb_marl.loggable` (`21328444`);
  (v)   the PPO settings and network are the rsl-rl control's (`--hparams rsl`),
        not skrl's defaults the first block ran with.

§5 lost three ablation arms to flags that never reached the env, and they
trained as copies of the control. That is why this checks the constructed
objects rather than the command lines.

Usage: marl_terrain_check.py <gate_dir> <repo> <expected rows>   (one <row>.log per run)
"""

from __future__ import annotations

import ast
import glob
import re
import sys
from pathlib import Path

ROWS = {  # row token -> (algo, partition, agents, widths, trainer class)
    "mappo": ("mappo", "legs2", 2, {"leg_left": 6, "leg_right": 6}, "MAPPO"),
    "ippo": ("ippo", "legs2", 2, {"leg_left": 6, "leg_right": 6}, "IPPO"),
    "limb1": ("ippo", "limb1", 1, {"whole": 12}, "IPPO"),
}
TAG = "Info / Curriculum/terrain_levels"


def tags_of(log_dir: Path) -> set[str]:
    from tensorboard.backend.event_processing import event_accumulator

    events = sorted(glob.glob(str(log_dir / "**" / "events.out.tfevents.*"), recursive=True))
    if not events:
        return set()
    ea = event_accumulator.EventAccumulator(events[-1], size_guidance={"scalars": 0})
    ea.Reload()
    return set(ea.Tags()["scalars"])


def check(log: Path, repo: Path) -> tuple[bool, str, int | None]:
    text = log.read_text(errors="replace")
    row = log.stem.split("-")[0]
    algo, part, n_agents, widths, cls = ROWS[row]
    problems = []

    m = re.search(r"\[marl-gate\] task=(\S+) partition=(\S+) n_dof=(\d+) agents=(\d+) "
                  r"act=(\{.*?\}) obs=(\d+) state=(\d+) policy_terms=(\[.*?\]) joints=(\{.*\})", text)
    obs = None
    if not m:
        problems.append("no [marl-gate] env line (env never built)")
    else:
        _, got_part, n_dof, agents, act, obs, state, terms, joints = m.groups()
        obs, state = int(obs), int(state)
        if got_part != part:
            problems.append(f"partition {got_part} != {part}")
        if int(n_dof) != 12:
            problems.append(f"n_dof {n_dof} != 12")
        if int(agents) != n_agents or ast.literal_eval(act) != widths:
            problems.append(f"agents {agents} act {act} != {n_agents} {widths}")
        if "depth" not in ast.literal_eval(terms):
            problems.append(f"no depth term in policy obs {terms}")
        if obs != state:
            problems.append(f"obs {obs} != state {state}")
        j = ast.literal_eval(joints)
        if part == "legs2" and (j.get("leg_left", [""])[0] != "leg_left_hip_roll_joint"
                                or j.get("leg_right", [""])[0] != "leg_right_hip_roll_joint"):
            problems.append(f"joint order wrong: {j}")

    t = re.search(r"\[marl-gate\] trainer_agent=(\w+) log_dir=(\S+)", text)
    if not t:
        problems.append("no trainer line (agent never built)")
    else:
        if t.group(1) != cls:
            problems.append(f"trainer {t.group(1)} != {cls}")
        if not re.search(r"48/48 \[", text):
            problems.append("did not reach 48/48 timesteps")
        tags = tags_of(repo / t.group(2))
        if TAG not in tags:
            problems.append(f"event file lacks {TAG!r} ({len(tags)} tags)")
    h = re.search(r"\[marl-gate\] hparams=(\S+) actor=(\[.*?\]) critic=(\[.*?\]) act=(\w+) "
                  r"normalize=(\w+) (\{.*\})", text)
    if not h:
        problems.append("no hparams line")
    else:
        hp = h.group(6)
        if (h.group(1) != "rsl" or ast.literal_eval(h.group(2)) != [256, 128, 128]
                or "KLAdaptiveLR" not in hp or "'entropy_loss_scale': 0.008" not in hp):
            problems.append(f"hparams not rsl-matched: {h.group(0)[:160]}")
    if re.search(r"CUDA out of memory|OutOfMemoryError", text):
        problems.append("CUDA OOM at 4096 envs")
    return (not problems), "; ".join(problems) or "ok", obs


def main() -> None:
    gate_dir, repo = Path(sys.argv[1]), Path(sys.argv[2] if len(sys.argv) > 2 else ".")
    expected = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    logs = sorted(gate_dir.glob("*.log"))
    ok_all, widths = bool(logs), set()
    for log in logs:
        ok, why, obs = check(log, repo)
        ok_all &= ok
        if obs is not None:
            widths.add(obs)
        print(f"  {log.stem:24} {'PASS' if ok else 'FAIL'}  obs={obs}  {why}")
    if len(widths) > 1:
        print(f"  observation width differs across terrains: {sorted(widths)}")
        ok_all = False
    print(f"G-B4t {'PASS' if ok_all and len(logs) == expected else 'FAIL'}  "
          f"{len(logs)} of {expected} rows logged, obs widths {sorted(widths)}")


if __name__ == "__main__":
    main()
