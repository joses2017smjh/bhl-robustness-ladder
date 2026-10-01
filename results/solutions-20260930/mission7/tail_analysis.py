"""Open-loop tail analysis: after hand-back the replay feeds the RECORDED body-frame
commands.  Compare the replay's pose/yaw with the recording at hand-back and measure how far
the tail drives the robot from where the recorded command was meant to be applied.
"""
import json
from pathlib import Path

import numpy as np
from analyze_gates import GATES, load, wrap

REPO = Path("/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder")
src = json.loads((REPO / "results/mission7-replay-smoke-20260921/fullroute/legacy-doors.json").read_text())
ref = {i: e for i, e in enumerate(src["episodes"])}

out = []
for name, path in GATES.items():
    data = load(path)
    for ep in data["episodes"]:
        idx = ep["layout_index"]
        hist = ep["stage_history"]
        rec = [h for h in hist if h["phase"] == "recorded"]
        if not rec:
            continue
        t_hb = rec[0]["time_s"]
        s = ep["samples"]
        r = ref[idx]["diagnostic_trace"]
        i_hb = next(i for i, x in enumerate(s) if x["time_s"] >= t_hb - 1e-9)
        dyaw_hb = wrap(s[i_hb]["yaw"] - r[i_hb]["yaw"])
        dxy_hb = float(np.linalg.norm(np.array(s[i_hb]["xy"]) - np.array(r[i_hb]["xy"])))
        # world-frame direction of recorded command in the recording vs as applied in the replay
        fall_t = ep["first_fall_s"]
        end = len(s) if fall_t is None or fall_t < t_hb else next(i for i, x in enumerate(s) if x["time_s"] >= fall_t)
        angs = []
        for i in range(i_hb, min(end, i_hb + 100)):
            c = np.array(s[i]["recorded_command"][:2])
            if np.linalg.norm(c) < .1:
                continue
            w_rep = np.array([[np.cos(s[i]["yaw"]), -np.sin(s[i]["yaw"])], [np.sin(s[i]["yaw"]), np.cos(s[i]["yaw"])]]) @ c
            w_ref = np.array([[np.cos(r[i]["yaw"]), -np.sin(r[i]["yaw"])], [np.sin(r[i]["yaw"]), np.cos(r[i]["yaw"])]]) @ c
            angs.append(abs(wrap(np.arctan2(w_rep[1], w_rep[0]) - np.arctan2(w_ref[1], w_ref[0]))))
        post_fall = fall_t is not None and fall_t > t_hb
        out.append(dict(gate=name, layout=idx, handback_s=round(t_hb, 2), yaw_diff_at_handback=round(dyaw_hb, 2),
                        pos_diff_at_handback_m=round(dxy_hb, 2),
                        median_cmd_world_angle_error_first4s_deg=None if not angs else round(float(np.degrees(np.median(angs))), 1),
                        post_stage_fall=post_fall, fall_s=fall_t, window_end=round(s[-1]["time_s"], 2),
                        tail_s=round(s[-1]["time_s"] - t_hb, 1)))
Path(__file__).with_name("tail_analysis.json").write_text(json.dumps(out, indent=1))
for o in out:
    print(f"{o['gate'][:22]:22s} L{o['layout']:2d} hb={o['handback_s']:6.2f} tail={o['tail_s']:5.1f}s dyaw={o['yaw_diff_at_handback']:+.2f} "
          f"dpos={o['pos_diff_at_handback_m']:.2f} cmd_angle_err={o['median_cmd_world_angle_error_first4s_deg']} "
          f"{'POSTFALL@%.2f' % o['fall_s'] if o['post_stage_fall'] else ''}")
