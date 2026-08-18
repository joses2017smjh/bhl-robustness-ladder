"""Flatten TensorBoard event files into one tidy CSV for plotting."""
import csv, os, re, sys
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

LOGROOT = Path(sys.argv[1])
OUT = Path(sys.argv[2])
TAGS = {
    "Train/mean_reward": "reward",
    "Train/mean_episode_length": "ep_len",
    "Episode_Termination/base_orientation": "fall_frac",
    "Curriculum/push_levels": "push_mps",
    "Curriculum/terrain_levels": "terrain_level",
}
rows = []
for d in sorted(LOGROOT.iterdir()):
    if not d.is_dir() or d.name == "isaaclab":
        continue
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
