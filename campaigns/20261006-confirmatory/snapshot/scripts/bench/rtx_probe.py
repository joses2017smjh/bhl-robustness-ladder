"""Can Isaac Sim 6.0.1 render on this cluster, and what would it cost in a loop?

§6 established that the RTX renderer segfaults under Isaac Sim 5.1 here, and
that geometric depth makes the crash irrelevant *for depth*. It left one thing
unmeasured, and it is the thing that decides whether a renderer-in-the-loop
pipeline is worth building: 5.1 was observed to die inside
`createHydraEngine`, and 6.0.1 was observed to get *through* that call. Getting
through a call is not the same as producing an image. A renderer that
initialises and then hands back a constant is exactly the failure mode §6 warns
about for warp ray-casting, and it would look like success in every log.

So this asserts on pixels, not on survival:

1. boot with the renderer enabled and report what Kit thinks it is running on;
2. create a render product and pull RGB and `distance_to_image_plane` back;
3. check the frame is not degenerate -- a real scene has more than one colour,
   and its depth has finite structure with a spread;
4. move a light and re-render, because a static frame can be a cached buffer;
5. time it at several camera counts and resolutions, which is the number the
   in-the-loop question actually turns on.

Run under the 6.0.1 interpreter (`venv60`), not the locked 5.1 one. It imports
no Isaac Lab: Isaac Lab 2.3 does not support Isaac Sim 6.0 and Isaac Lab 3.0
does not support 5.1, so a probe that needed Isaac Lab could not be a probe of
the boundary itself.
"""

from __future__ import annotations

import argparse
import json
import time


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--counts", type=int, nargs="+", default=(1, 4, 16, 64))
    p.add_argument("--res", type=int, nargs="+", default=(64, 128))
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--frames", type=int, default=24)
    p.add_argument("--json", default=None)
    args = p.parse_args()

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "renderer": "RaytracedLighting"})

    import carb
    import numpy as np
    import omni.replicator.core as rep
    from pxr import Gf, UsdGeom, UsdLux

    import omni.usd

    report: dict = {"stages": {}}

    def stage_ok(name, **kv):
        report["stages"][name] = dict(ok=True, **kv)
        print(f"  PASS  {name}  " + "  ".join(f"{k}={v}" for k, v in kv.items()),
              flush=True)

    def stage_fail(name, why):
        report["stages"][name] = dict(ok=False, why=str(why))
        print(f"  FAIL  {name}  {why}", flush=True)

    print("=== 1. what is Kit running ===")
    settings = carb.settings.get_settings()
    ver = settings.get("/app/version") or "?"
    renderer = settings.get("/renderer/active") or "?"
    stage_ok("boot", kit_version=ver, active_renderer=renderer)

    print("\n=== 2. a scene, a render product, and a readback ===")
    ctx = omni.usd.get_context()
    ctx.new_stage()
    stage = ctx.get_stage()

    # Deliberately high-contrast and non-symmetric geometry. A cube and a sphere
    # at different depths mean a correct depth image has two distinct plateaus,
    # so "did it render" can be answered from the histogram instead of by eye.
    UsdGeom.Xform.Define(stage, "/World")
    ground = UsdGeom.Cube.Define(stage, "/World/ground")
    ground.GetSizeAttr().Set(1.0)
    UsdGeom.Xformable(ground).AddScaleOp().Set(Gf.Vec3f(20.0, 20.0, 0.05))
    box = UsdGeom.Cube.Define(stage, "/World/box")
    box.GetSizeAttr().Set(1.0)
    UsdGeom.Xformable(box).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.6))
    ball = UsdGeom.Sphere.Define(stage, "/World/ball")
    ball.GetRadiusAttr().Set(0.5)
    UsdGeom.Xformable(ball).AddTranslateOp().Set(Gf.Vec3d(2.2, 0.4, 0.5))
    light = UsdLux.DistantLight.Define(stage, "/World/sun")
    light.GetIntensityAttr().Set(3000.0)

    try:
        cam = rep.create.camera(position=(-4.0, 0.0, 1.8), look_at=(0.0, 0.0, 0.5))
        rp = rep.create.render_product(cam, (args.res[0], args.res[0]))
        rgb_a = rep.AnnotatorRegistry.get_annotator("rgb")
        dep_a = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
        rgb_a.attach(rp)
        dep_a.attach(rp)
        for _ in range(args.warmup):
            rep.orchestrator.step(rt_subframes=1)
        rgb = np.asarray(rgb_a.get_data())
        dep = np.asarray(dep_a.get_data())
        stage_ok("render_product", rgb=str(rgb.shape), depth=str(dep.shape))
    except Exception as e:  # noqa: BLE001
        stage_fail("render_product", e)
        app.close()
        raise SystemExit(1)

    print("\n=== 3. is the frame an image or a constant ===")
    try:
        colours = len(np.unique(rgb.reshape(-1, rgb.shape[-1]), axis=0))
        finite = np.isfinite(dep) & (dep > 0)
        frac = float(finite.mean())
        spread = float(np.ptp(dep[finite])) if finite.any() else 0.0
        if colours < 8:
            raise AssertionError(f"only {colours} distinct colours; this is a fill")
        if frac < 0.5:
            raise AssertionError(f"{frac:.0%} finite depth pixels")
        if spread < 0.10:
            raise AssertionError(f"depth spread {spread:.3f} m; no structure")
        stage_ok("not_degenerate", colours=colours,
                 finite=f"{frac:.0%}", depth_spread_m=round(spread, 3))
    except Exception as e:  # noqa: BLE001
        stage_fail("not_degenerate", e)

    print("\n=== 4. does it re-render, or is that a cached buffer ===")
    try:
        light.GetIntensityAttr().Set(300.0)
        for _ in range(4):
            rep.orchestrator.step(rt_subframes=1)
        rgb2 = np.asarray(rgb_a.get_data())
        delta = float(np.abs(rgb2.astype(np.int32) - rgb.astype(np.int32)).mean())
        if delta < 0.5:
            raise AssertionError(f"mean pixel delta {delta:.3f} after a 10x "
                                 "light change; the buffer is stale")
        stage_ok("responds_to_scene", mean_pixel_delta=round(delta, 2))
    except Exception as e:  # noqa: BLE001
        stage_fail("responds_to_scene", e)

    print("\n=== 5. what it costs: ms per rendered frame vs camera count ===")
    # The comparison this feeds is §6's physics-only budget: 187.5 ms per policy
    # step at 4,096 envs. Anything here that approaches that per step is a
    # renderer that halves training throughput on its own.
    print(f"{'cameras':>8}{'res':>7}{'ms/frame':>11}{'ms/camera':>11}")
    timings = []
    for res in args.res:
        for n in args.counts:
            try:
                cams = [rep.create.camera(position=(-4.0, 0.6 * i, 1.8),
                                          look_at=(0.0, 0.0, 0.5))
                        for i in range(n)]
                prods = [rep.create.render_product(c, (res, res)) for c in cams]
                ann = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
                for q in prods:
                    ann.attach(q)
                for _ in range(args.warmup):
                    rep.orchestrator.step(rt_subframes=1)
                t0 = time.perf_counter()
                for _ in range(args.frames):
                    rep.orchestrator.step(rt_subframes=1)
                ms = (time.perf_counter() - t0) / args.frames * 1e3
                print(f"{n:>8}{res:>7}{ms:>11.2f}{ms / n:>11.3f}", flush=True)
                timings.append(dict(cameras=n, res=res, ms_per_frame=round(ms, 3),
                                    ms_per_camera=round(ms / n, 4)))
                for q in prods:
                    ann.detach(q)
                    q.destroy()
            except Exception as e:  # noqa: BLE001
                print(f"{n:>8}{res:>7}   failed: {type(e).__name__}: {e}", flush=True)
                timings.append(dict(cameras=n, res=res, error=str(e)))
    report["timings"] = timings

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"\njson -> {args.json}")

    bad = [k for k, v in report["stages"].items() if not v["ok"]]
    print(f"\n{'ALL PASS' if not bad else 'FAILED: ' + ', '.join(bad)}")
    app.close()
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
