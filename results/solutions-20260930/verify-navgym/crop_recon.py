"""verify-navgym: ideal-raycast reconstruction of the 3x24x24 ego-map crop from the SAVED s6 physics trace
poses on maze 50009 (trace every 0.2 s; gym ray caster on the true wall boxes; no policy stepped, no
episode run, nothing scored). Question: when s6 stands in the route's branch cell (4,0), does the crop
show the opening up into (4,1) (the route) and the closed top of the pocket (5,1)?"""
import json, math, sys
import numpy as np
from bhl_robust.eval import random_maze as rm
from bhl_robust.navgym import env as ng
from PIL import Image, ImageDraw
RES = "/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder/results/navgym-v4-transfer-20260930"
mz = rm.generate(6, 6, 50009, extra_openings=1); boxes = ng.wall_boxes(mz)
angles = np.linspace(-np.pi, np.pi, ng.LIDAR_RAYS, endpoint=False)
tr = json.load(open(f"{RES}/armV4-s6/seed50009.json"))["trace"]
em = ng.EgoMap(mz.bounds())
size, half, res = ng.MAP_CROP, ng.MAP_CROP // 2, ng.MAP_RES
u = (half - np.arange(size)) * res
U, V = np.meshgrid(u, u, indexing="ij")
print("crop forward offsets %.1f..%.1f m, lateral %.1f..%.1f m" % (u.min(), u.max(), u.min(), u.max()))
rows, saved = [], 0
for k, t in enumerate(tr):
    x, y = t["xy"]; yaw = t["yaw"]
    em.update(x, y, yaw, angles, ng.cast_rays(boxes, (x, y), yaw + angles, ng.LIDAR_RANGE))
    if not (4.9 < x < 6.3 and y < 0.7):
        continue
    crop = em.crop(x, y, yaw)
    c, s = math.cos(yaw), math.sin(yaw)
    wx = x + U * c - V * s; wy = y + U * s + V * c
    in41 = (wx > 5.0) & (wx < 6.2) & (wy > 0.8) & (wy < 2.0)          # interior of route cell (4,1)
    in42 = (wx > 5.0) & (wx < 6.2) & (wy > 2.15) & (wy < 3.4)         # interior of route cell (4,2)
    top51 = (wx > 6.35) & (wx < 7.65) & (wy > 2.0) & (wy < 2.25)      # the pocket (5,1)'s closed top wall
    turn52 = (wx > 6.35) & (wx < 7.65) & (wy > 2.15) & (wy < 3.4)     # route cell (5,2) (the turn east)
    r = {"t": t["t"], "x": x, "y": y, "yaw_deg": round(math.degrees(yaw), 1),
         "cells_41_in_crop": int(in41.sum()), "free_41": round(float(crop[1][in41].mean()), 2) if in41.any() else None,
         "cells_42_in_crop": int(in42.sum()), "free_42": round(float(crop[1][in42].mean()), 2) if in42.any() else None,
         "cells_52_in_crop": int(turn52.sum()),
         "cells_top51_in_crop": int(top51.sum()), "occ_top51": round(float(crop[0][top51].mean()), 2) if top51.any() else None}
    rows.append(r)
    if saved < 2 and r["free_41"] is not None and r["free_41"] > 0.5 and t["t"] > 40:
        img = Image.new("RGB", (size * 10, size * 10)); d = ImageDraw.Draw(img)
        for i in range(size):
            for j in range(size):
                col = (30, 30, 30) if crop[0, i, j] else ((235, 235, 235) if crop[1, i, j] else (140, 140, 160))
                if in41[i, j]: col = (120, 200, 120) if crop[1, i, j] else (60, 120, 60)
                if top51[i, j] and crop[0, i, j]: col = (200, 60, 60)
                d.rectangle((j * 10, i * 10, j * 10 + 9, i * 10 + 9), fill=col)   # row i = forward offset (up), col j = left->right
        img.save(f"crop_s6_50009_t{t['t']:.1f}.png"); saved += 1
fr41 = [r["free_41"] for r in rows if r["free_41"] is not None]
print("samples in (4,0):", len(rows), "| (4,1) in crop:", sum(r["cells_41_in_crop"] > 0 for r in rows),
      "| mean free fraction of (4,1) when in crop: %.2f" % np.mean(fr41),
      "| (4,2) in crop:", sum(r["cells_42_in_crop"] > 0 for r in rows), "| (5,2) in crop:", sum(r["cells_52_in_crop"] > 0 for r in rows),
      "| pocket top wall in crop:", sum(r["cells_top51_in_crop"] > 0 for r in rows),
      "| occ when in crop (after t>18 s, pocket visited): %.2f" % np.mean([r["occ_top51"] for r in rows if r["occ_top51"] is not None and r["t"] > 18]))
for r in rows[::max(1, len(rows) // 8)]:
    print(r)
json.dump(rows, open(sys.argv[1], "w"), indent=1)
