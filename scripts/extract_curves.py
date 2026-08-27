"""Flatten TensorBoard event files into one tidy CSV for plotting."""
import csv, os, re, sys
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

LOGROOTS = [Path(a) for a in sys.argv[1:-1]]
OUT = Path(sys.argv[-1])
TAGS = {
    "Train/mean_reward": "reward",
    "Train/mean_episode_length": "ep_len",
    "Episode_Termination/base_orientation": "fall_frac",
    "Episode_Termination/fallen": "fall_frac",
    "Curriculum/push_levels": "push_mps",
    "Curriculum/terrain_levels": "terrain_level",
    "Episode_Reward/reaching_fine": "pinch",
    "Episode_Reward/lifting_object": "lift_bonus",
    "Curriculum/lift_height": "lift_h",
    "Curriculum/stage_lift": "stage_lift",
}
rows = []
dirs = [d for root in LOGROOTS if root.is_dir() for d in sorted(root.iterdir())]

# One label per run. The directory name carries a timestamp and the label does
# not, so a rerun of the same arm produces a second directory with the *same*
# label -- and the downstream summariser keys on label alone, which silently
# concatenated the two runs' scalars into one curve. That is how a 16,000-step
# curve came out of a 6,000-iteration arm, and how an occlusion arm that lifts
# to 21 cm was averaged into one that never left the floor.
#
# Newest wins, because a rerun exists to supersede. Anything dropped is named,
# so a superseded run is a line in the log rather than a silent merge.
newest = {}
for d in dirs:
    if not d.is_dir() or d.name == "isaaclab":
        continue
    label = re.sub(r"^[0-9-]+_[0-9-]+_", "", d.name)
    if label in newest:
        print(f"  superseded: {newest[label].name} <- {d.name}")
    newest[label] = d
dirs = [newest[k] for k in sorted(newest)]

for d in dirs:
    label = re.sub(r"^[0-9-]+_[0-9-]+_", "", d.name)
    ea = EventAccumulator(str(d), size_guidance={"scalars": 10000})
    try:
        ea.Reload()
    except Exception as e:
        print(f"  skip {label}: {e}"); continue
    avail = set(ea.Tags().get("scalars", []))
    for tag, short in TAGS.items():
        if tag not in avail:
            continue
        for ev in ea.Scalars(tag):
            rows.append({"label": label, "metric": short, "step": ev.step, "value": ev.value})
    print(f"  {label}: {sorted(avail & set(TAGS))}")

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["label", "metric", "step", "value"])
    w.writeheader(); w.writerows(rows)
print(f"wrote {len(rows)} rows -> {OUT}")
