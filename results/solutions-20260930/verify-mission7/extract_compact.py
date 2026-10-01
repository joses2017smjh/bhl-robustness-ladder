"""Read-only: extract compact per-sample traces from Mission 7 replay-gate episode files.

Streams the indent=2 JSON line by line (the 1 GB files do not fit a 6 GB job when json.load'ed) and writes
<label>.pkl next to this script: per episode the header fields (stage history etc.), per-sample
time/xy/yaw/tilt/phase/effective+recorded command/body velocity, and plate contact events
(time, plate geom, body, normal force, distance).
"""
import gzip
import json
import pickle
import sys
from pathlib import Path

R = Path("/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder")
C = R / "results/mission7-campaign-20260923"
F = R / "results/mission7-approach-followup-20260922"
GATES = {
    "gref_21397732": F / "plate-stage-cn-c22-guarded/episodes.json",
    "waitopen2_21400863": C / "replay-gate-waitopen2/episodes.json.gz",
    "v2align_21405537": C / "replay-gate-v2-align/episodes.json.gz",
    "v3s190_21405539": C / "replay-gate-v3-settle190/episodes.json.gz",
    "v4s080_21405541": C / "replay-gate-v4-settle080/episodes.json.gz",
    "v3s190a_21405543": C / "replay-gate-v3-settle190-align/episodes.json.gz",
    "v4s080a_21405545": C / "replay-gate-v4-settle080-align/episodes.json.gz",
}
OUT = Path(__file__).resolve().parent


def compact_sample(x):
    plate_ev = []
    for ev in x["contact_events"]:
        if ev["world_geom"].startswith("plate_"):
            body = ev.get("body2") if ev.get("body1") == "world" else ev.get("body1")
            plate_ev.append((x["time_s"], ev["world_geom"], body, ev["normal_force_N"], ev.get("distance_m")))
    return dict(t=x["time_s"], xy=x["xy"], yaw=x["yaw"], tilt=x["tilt"], phase=x["phase"],
                eff=x["effective_command"], rec=x["recorded_command"], vb=x["velocity_body"],
                yaw_rate=x.get("yaw_rate"), contacts=x.get("contacts")), plate_ev


def stream(path):
    op = gzip.open if str(path).endswith(".gz") else open
    episodes = []
    complete = None
    with op(path, "rt") as fh:
        state = "top"
        header, buf = [], []
        cur = None
        for line in fh:
            if state == "top":
                if line.startswith('  "complete"'):
                    complete = json.loads("{" + line.strip().rstrip(",") + "}")["complete"]
                elif line.rstrip() == "    {":
                    state, header = "header", []
                continue
            if state == "header":
                if line.startswith('      "samples": ['):
                    text = "{" + "".join(header).rstrip().rstrip(",") + "}"
                    cur = json.loads(text)
                    cur.update(t=[], xy=[], yaw=[], tilt=[], phase=[], eff=[], rec=[], vb=[], yaw_rate=[],
                               contacts=[], plate_ev=[])
                    state = "samples"
                else:
                    header.append(line)
                continue
            if state == "samples":
                stripped = line.rstrip()
                if stripped == "        {":
                    buf = [line]
                    state = "sample"
                elif stripped == "      ]":
                    state = "after"
                continue
            if state == "sample":
                buf.append(line)
                if line.rstrip() in ("        }", "        },"):
                    x = json.loads("".join(buf).rstrip().rstrip(","))
                    cs, pev = compact_sample(x)
                    for k, v in cs.items():
                        cur[k].append(v)
                    cur["plate_ev"].extend(pev)
                    state = "samples"
                continue
            if state == "after":
                if line.rstrip() in ("    }", "    },"):
                    episodes.append(cur)
                    cur = None
                    state = "top"
                continue
    return complete, episodes


def main(labels):
    for label in labels:
        complete, eps = stream(GATES[label])
        with open(OUT / f"{label}.pkl", "wb") as fh:
            pickle.dump(dict(label=label, path=str(GATES[label]), complete=complete, episodes=eps), fh)
        print(label, "complete", complete, "episodes", len(eps), "layouts", [e["layout_index"] for e in eps],
              "samples", [len(e["t"]) for e in eps], flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or list(GATES))
