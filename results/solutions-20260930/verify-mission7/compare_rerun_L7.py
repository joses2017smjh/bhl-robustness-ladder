"""Read-only: compare the investigator's login-node rerun of v2-align layout 7 with the cn-c22 gate trace, sample by sample."""
import gzip, json, sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
GATE = "/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder/results/mission7-campaign-20260923/replay-gate-v2-align/episodes.json.gz"
RERUN = "/nfs/hpc/share/sanchej7/Humanoid_Lite/solutions-20260930/mission7/replay_v2align_L7/episodes.json"

def episode_samples(path, want_layout):
    op = gzip.open if path.endswith(".gz") else open
    state, header, buf, samples, cur_layout = "top", [], [], [], None
    with op(path, "rt") as fh:
        for line in fh:
            if state == "top":
                if line.rstrip() == "    {":
                    state, header = "header", []
                continue
            if state == "header":
                if line.startswith('      "samples": ['):
                    hdr = json.loads("{" + "".join(header).rstrip().rstrip(",") + "}")
                    cur_layout = hdr["layout_index"]
                    state = "samples"
                else:
                    header.append(line)
                continue
            if state == "samples":
                s = line.rstrip()
                if s == "        {":
                    buf, state = [line], "sample"
                elif s == "      ]":
                    if cur_layout == want_layout:
                        return hdr, samples
                    state, samples = "top", []
                continue
            if state == "sample":
                buf.append(line)
                if line.rstrip() in ("        }", "        },"):
                    if cur_layout == want_layout:
                        samples.append(json.loads("".join(buf).rstrip().rstrip(",")))
                    state = "samples"
    return None, None

h1, a = episode_samples(GATE, 7)
h2, b = episode_samples(RERUN, 7)
print("gate samples", len(a), "rerun samples", len(b), "gate first_fall", h1["first_fall_s"], "rerun first_fall", h2["first_fall_s"])
ndiff = 0
for i, (x, y) in enumerate(zip(a, b)):
    if x != y:
        ndiff += 1
        if ndiff <= 3:
            keys = [k for k in x if x.get(k) != y.get(k)]
            print("diff at", i, x["time_s"], keys)
print("samples differing:", ndiff, "of", min(len(a), len(b)))
hk = [k for k in h1 if k not in ("baseline",) and h1[k] != h2.get(k)]
print("header keys differing:", hk)

import numpy as np
def flat(v, prefix=""):
    out = {}
    if isinstance(v, dict):
        for k, w in v.items():
            out.update(flat(w, prefix + "." + k))
    elif isinstance(v, list):
        for i, w in enumerate(v):
            out.update(flat(w, prefix + f"[{i}]"))
    else:
        out[prefix] = v
    return out
maxd = {}
nonnum = set()
for x, y in zip(a, b):
    fx, fy = flat(x), flat(y)
    for k in set(fx) | set(fy):
        vx, vy = fx.get(k), fy.get(k)
        if vx == vy:
            continue
        kk = k.split("[")[0]
        if isinstance(vx, (int, float)) and isinstance(vy, (int, float)):
            maxd[kk] = max(maxd.get(kk, 0.), abs(vx - vy))
        else:
            nonnum.add(kk)
print("max abs diff per field:", {k: float(v) for k, v in sorted(maxd.items())})
print("non-numeric differing fields:", sorted(nonnum))
print("header maximum_tilt", h1["maximum_tilt"], h2["maximum_tilt"], "peak_tan", h1["peak_plate_tangent_force_N"], h2["peak_plate_tangent_force_N"])
