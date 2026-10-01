"""turn_test v2 protocol (reset seeds 0-2, +/-0.6 rad/s after a 3 s settle, 6 s; walk (0.35,0,0) seed 0 after 1 s)
replayed with Isaac's TRAINING observation corruption (uniform: ang vel 0.3, gravity 0.05, joint pos 0.05,
joint vel 2.0) and no action noise. Report-only diagnostic (not the predeclared gate)."""
import json, sys
import mj_probe2 as P

runs = sys.argv[1].split(",")
mult = float(sys.argv[2])
out = {}
for r in runs:
    ck = P.ckpt_path(r, None)
    turns = []
    for s in (0, 1, 2):
        for w in (0.6, -0.6):
            o = P.run(ck, w, 3.0, 6.0, s, 0.0, 0.0, "all", 1e9, 3.0, mult)
            turns.append(o)
    walk = P.run(ck, 0.0, 1.0, 6.0, 0, 0.0, 0.35, "all", 1e9, 3.0, mult)
    ok_t = sum(1 for o in turns if o["fell_at_s"] is None and o["yaw_deg_cmd_dir"] >= 150.0)
    ok_w = walk["fell_at_s"] is None and abs(walk["yaw_deg_cmd_dir"]) <= 15.0
    out[r] = {"turns": turns, "walk": walk, "n_turn_ok": ok_t, "walk_ok": ok_w, "v2_pass": ok_t == 6 and ok_w}
    print(f"{r:28s} obs-noise x{mult}: turns {[o['yaw_deg_cmd_dir'] for o in turns]} ok {ok_t}/6 | walk drift {walk['yaw_deg_cmd_dir']} fell {walk['fell_at_s']} | v2 {'PASS' if out[r]['v2_pass'] else 'FAIL'}", flush=True)
json.dump(out, open(f"v2_obsnoise_x{mult}.json", "w"), indent=1)
