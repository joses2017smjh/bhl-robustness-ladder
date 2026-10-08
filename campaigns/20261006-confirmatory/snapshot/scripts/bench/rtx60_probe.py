"""Does the RTX renderer produce actual pixels on Isaac Sim 6.0?

This is the single question that decides whether porting the task overlays from
Isaac Lab 2.x to 3.x is worth doing. Isaac Sim 5.1 segfaults on this cluster
inside `omni.usd.create_hydra_engine`, which is why every vision result in this
repo uses Warp ray-casting -- geometry only, no materials, no lighting, no
colour. If 6.0 renders, RGB and rendered depth become available and section 6's
central constraint lifts.

An earlier probe showed 6.0.1 surviving the call that kills 5.1. Surviving a
call is not rendering, so this one opens a stage, puts a lit object in front of
a camera, and reads the buffer back. It checks that the image is not uniform,
because a renderer that returns a constant grey frame has technically not
crashed and has also not worked.
"""
import os

from isaacsim import SimulationApp

app = SimulationApp({"headless": True, "renderer": "RayTracedLighting",
                     "width": 256, "height": 256})

import numpy as np                                   # noqa: E402
import omni.replicator.core as rep                   # noqa: E402
import omni.usd                                      # noqa: E402
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade   # noqa: E402

try:
    from isaacsim.core.version import get_version
    ver = get_version()[0]
except Exception:
    ver = "?"

omni.usd.get_context().new_stage()
stage = omni.usd.get_context().get_stage()
print(f"PROBE | isaacsim {ver} | stage opened without segfault", flush=True)

UsdGeom.Xform.Define(stage, "/World")
cube = UsdGeom.Cube.Define(stage, "/World/cube")
cube.CreateSizeAttr(60.0)
UsdGeom.Xformable(cube).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
# A bare UsdGeom.Cube has no material, and an unshaded prim renders black no
# matter how well the renderer works. The first version of this probe reported
# "no image" on a frame whose DEPTH was correct to the centimetre, which is a
# scene bug wearing a renderer bug's clothes.
mat = UsdShade.Material.Define(stage, "/World/mat")
shader = UsdShade.Shader.Define(stage, "/World/mat/surface")
shader.CreateIdAttr("UsdPreviewSurface")
shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.9, 0.45, 0.2))
shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
UsdShade.MaterialBindingAPI(cube.GetPrim()).Bind(mat)

# Aim the key light at the cube rather than leaving it on its default axis, and
# add a dome so nothing depends on a single direction being right.
light = UsdLux.DistantLight.Define(stage, "/World/key")
light.CreateIntensityAttr(6000.0)
UsdGeom.Xformable(light).AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 0.0, 45.0))
dome = UsdLux.DomeLight.Define(stage, "/World/dome")
dome.CreateIntensityAttr(1200.0)

cam = UsdGeom.Camera.Define(stage, "/World/cam")
UsdGeom.Xformable(cam).AddTranslateOp().Set(Gf.Vec3d(0.0, -300.0, 0.0))
UsdGeom.Xformable(cam).AddRotateXYZOp().Set(Gf.Vec3f(90.0, 0.0, 0.0))
cam.CreateClippingRangeAttr(Gf.Vec2f(1.0, 10000.0))

rp = rep.create.render_product("/World/cam", (256, 256))
rgb = rep.AnnotatorRegistry.get_annotator("rgb")
dep = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
rgb.attach(rp)
dep.attach(rp)
for _ in range(20):
    rep.orchestrator.step(rt_subframes=4)

c = np.asarray(rgb.get_data())[..., :3].astype(np.float32)
d = np.asarray(dep.get_data()).astype(np.float32)
finite = np.isfinite(d) & (d < 1e30)

lines = [
    f"RTX {ver} | rgb {c.shape} mean={c.mean():.1f} std={c.std():.1f} "
    f"unique={len(np.unique(c.reshape(-1, 3), axis=0))}",
    f"RTX {ver} | depth {d.shape} finite={finite.mean():.3f} "
    f"range=[{d[finite].min():.1f}, {d[finite].max():.1f}]" if finite.any()
    else f"RTX {ver} | depth all non-finite",
]
# A crashed renderer returns nothing; a broken one returns a flat frame. Both
# have to fail, so the test is variance rather than existence.
# Depth and colour are reported separately because they fail separately: an
# unlit scene gives correct depth and a black frame, which is a scene problem,
# while a dead renderer gives neither.
depth_ok = finite.mean() > 0.1 and np.isfinite(d[finite]).all()
rgb_ok = c.std() > 1.0
ok = depth_ok and rgb_ok
lines.append(f"VERDICT | depth={'OK' if depth_ok else 'DEAD'} "
             f"rgb={'OK' if rgb_ok else 'BLACK'} -> "
             f"{'RTX FULLY RENDERS ON 6.0' if ok else ('RTX GEOMETRY OK, SHADING NOT' if depth_ok else 'RTX DEAD')}")
for ln in lines:
    print(ln, flush=True)
with open(os.environ.get("BENCH_OUT", "/tmp/rtx60.txt"), "a") as f:
    f.write("\n".join(lines) + "\n")
app.close()
