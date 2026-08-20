"""Scripted squat-and-pick on the 22-DoF humanoid.

This is not a trained policy. The robot has no fingers and the shoulders cannot
adduct past a ~36 cm hand spacing, so the object is a wide bin lifted from the
sides — the grasp that this morphology can actually close. Joint targets are
interpolated and the floating base is held upright so a 6 Nm machine can finish
the motion; a weld would fight itself with two hands on one free body, so after
the grasp the bin is parented to the hands.

The clip exists because a walking GIF does not show that the 22-DoF model has
arms, and a locomotion policy will never squat.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

from bhl_robust.eval.mjcf_assets import prepare_mjcf
from bhl_robust.eval.video import EpisodeRecorder

JOINTS = [
    "arm_left_shoulder_pitch_joint", "arm_left_shoulder_roll_joint",
    "arm_left_shoulder_yaw_joint", "arm_left_elbow_pitch_joint",
    "arm_left_elbow_roll_joint",
    "arm_right_shoulder_pitch_joint", "arm_right_shoulder_roll_joint",
    "arm_right_shoulder_yaw_joint", "arm_right_elbow_pitch_joint",
    "arm_right_elbow_roll_joint",
    "leg_left_hip_roll_joint", "leg_left_hip_yaw_joint",
    "leg_left_hip_pitch_joint", "leg_left_knee_pitch_joint",
    "leg_left_ankle_pitch_joint", "leg_left_ankle_roll_joint",
    "leg_right_hip_roll_joint", "leg_right_hip_yaw_joint",
    "leg_right_hip_pitch_joint", "leg_right_knee_pitch_joint",
    "leg_right_ankle_pitch_joint", "leg_right_ankle_roll_joint",
]

DT = 0.002
SUBSTEPS = 10
POLICY_DT = DT * SUBSTEPS


def _vec(**kw) -> np.ndarray:
    q = np.zeros(22, dtype=np.float64)
    q[12] = q[18] = -0.20
    q[13] = q[19] = 0.40
    q[14] = q[20] = -0.30
    idx = {n: i for i, n in enumerate(JOINTS)}
    for k, v in kw.items():
        q[idx[k]] = v
    return q


STAND = _vec()
# Max adduct (left roll lo = -0.26). Any more positive abducts; the bin is
# sized to that 36 cm span because the shoulders cannot close further.
_ARMS_HOLD = dict(
    arm_left_shoulder_roll_joint=-0.26, arm_right_shoulder_roll_joint=0.26,
    arm_left_shoulder_pitch_joint=-0.15, arm_right_shoulder_pitch_joint=0.15,
    arm_left_elbow_pitch_joint=0.45, arm_right_elbow_pitch_joint=-0.45,
)
_LEGS_SQUAT = dict(
    leg_left_hip_pitch_joint=-1.35, leg_right_hip_pitch_joint=-1.35,
    leg_left_knee_pitch_joint=2.05, leg_right_knee_pitch_joint=2.05,
    leg_left_ankle_pitch_joint=-0.68, leg_right_ankle_pitch_joint=-0.68,
)
SQUAT = _vec(**_LEGS_SQUAT)
GRASP = _vec(**_LEGS_SQUAT, **_ARMS_HOLD)
HOLD = _vec(**_ARMS_HOLD)
PRESENT = _vec(
    arm_left_shoulder_pitch_joint=-0.55, arm_right_shoulder_pitch_joint=0.55,
    arm_left_shoulder_roll_joint=-0.26, arm_right_shoulder_roll_joint=0.26,
    arm_left_elbow_pitch_joint=0.90, arm_right_elbow_pitch_joint=-0.90,
)


# (target, duration_s, grasp?) — grasp latches on the first True and stays.
SCHEDULE = [
    (STAND, 1.0, False),
    (SQUAT, 1.8, False),
    (GRASP, 0.8, False),
    (GRASP, 0.3, True),
    (HOLD, 2.0, True),
    (PRESENT, 1.6, True),
    (PRESENT, 1.2, True),
]


def build_model(upstream: Path, cache_dir: Path) -> mujoco.MjModel:
    """Patched humanoid scene plus a free bin, written next to the include."""
    scene = prepare_mjcf(upstream, cache_dir, "humanoid")
    extra_body = """
    <geom name="pedestal" type="cylinder" size="0.04 0.14" pos="0.09 0 0.14"
          rgba="0.22 0.26 0.32 1" contype="0" conaffinity="0"/>
    <body name="crate" pos="0.09 0 0.33">
      <freejoint name="crate_free"/>
      <geom name="crate_g" type="box" size="0.05 0.18 0.05" mass="0.12"
            rgba="0.95 0.48 0.12 1" friction="1.2 0.4 0.1"/>
    </body>
"""
    text = scene.read_text()
    if "<body name=\"crate\"" not in text:
        if "</worldbody>" not in text:
            raise RuntimeError(f"cannot splice crate into {scene}")
        text = text.replace("</worldbody>", extra_body + "  </worldbody>", 1)
    # Must live next to berkeley_humanoid_lite.xml — a relative <include>.
    world = scene.parent / "pick_scene.xml"
    world.write_text(text)
    return mujoco.MjModel.from_xml_path(str(world))


def _set_joints(model, data, q):
    for i, name in enumerate(JOINTS):
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[j]] = q[i]


def _ids(model):
    return {
        "lh": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "arm_left_hand_link"),
        "rh": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "arm_right_hand_link"),
        "la": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "leg_left_ankle_roll"),
        "ra": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "leg_right_ankle_roll"),
        "crate": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "crate"),
        "crate_j": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "crate_free"),
    }


def _lock_base(data):
    """Hold the underactuated freejoint upright; z is handled by planting feet."""
    data.qpos[0] = 0.0
    data.qpos[1] = 0.0
    data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    data.qvel[0:6] = 0.0


def _plant_feet(model, data, ids, stand_ankle_z):
    mujoco.mj_forward(model, data)
    z = 0.5 * (data.xpos[ids["la"]][2] + data.xpos[ids["ra"]][2])
    data.qpos[2] -= (z - stand_ankle_z)


def _place_crate(model, data, ids, mid: np.ndarray, quat=(1.0, 0.0, 0.0, 0.0)):
    adr = model.jnt_qposadr[ids["crate_j"]]
    dadr = model.jnt_dofadr[ids["crate_j"]]
    data.qpos[adr:adr + 3] = mid
    data.qpos[adr + 3:adr + 7] = quat
    data.qvel[dadr:dadr + 6] = 0.0


def _crate_between_hands(model, data, ids) -> np.ndarray:
    L = data.xpos[ids["lh"]]
    R = data.xpos[ids["rh"]]
    mid = 0.5 * (L + R)
    # Sit the bin a finger-width below the wrists so the palms meet the sides.
    mid[2] -= 0.02
    return mid


def _lerp(a, b, t):
    t = float(np.clip(t, 0.0, 1.0))
    s = t * t * (3.0 - 2.0 * t)
    return a + s * (b - a)


def _pose(model, data, ids, q, stand_ankle_z):
    _set_joints(model, data, q)
    _lock_base(data)
    _plant_feet(model, data, ids, stand_ankle_z)
    _lock_base(data)
    mujoco.mj_forward(model, data)


def run(upstream: Path, cache_dir: Path, out: Path | None, gif: Path | None) -> None:
    model = build_model(upstream, cache_dir)
    model.opt.timestep = DT
    data = mujoco.MjData(model)
    ids = _ids(model)

    mujoco.mj_resetData(model, data)
    _set_joints(model, data, STAND)
    mujoco.mj_forward(model, data)
    stand_ankle_z = 0.5 * (data.xpos[ids["la"]][2] + data.xpos[ids["ra"]][2])
    _pose(model, data, ids, STAND, stand_ankle_z)

    print("stand hands L", np.round(data.xpos[ids["lh"]], 3),
          "R", np.round(data.xpos[ids["rh"]], 3))
    _pose(model, data, ids, GRASP, stand_ankle_z)
    crate0 = _crate_between_hands(model, data, ids)
    print("grasp hands L", np.round(data.xpos[ids["lh"]], 3),
          "R", np.round(data.xpos[ids["rh"]], 3),
          "crate0", np.round(crate0, 3))
    _pose(model, data, ids, STAND, stand_ankle_z)
    _place_crate(model, data, ids, crate0)
    mujoco.mj_forward(model, data)
    print("crate parked", np.round(data.xpos[ids["crate"]], 3))

    rec = None
    if out is not None:
        rec = EpisodeRecorder(
            model, out, fps=1.0 / POLICY_DT,
            width=960, height=540,
            caption="22 DoF  ·  squat, pick, stand   (scripted, not a policy)",
            track_body="base",
        )
        rec.camera.distance = 2.35
        rec.camera.elevation = -14.0
        rec.camera.azimuth = 145.0

    grasped = False
    q_prev = STAND.copy()
    t_abs = 0.0
    crate_z0 = float(data.xpos[ids["crate"]][2])

    for target, dur, want_grasp in SCHEDULE:
        n = max(1, int(round(dur / POLICY_DT)))
        for k in range(n):
            q_des = _lerp(q_prev, target, (k + 1) / n)
            _pose(model, data, ids, q_des, stand_ankle_z)
            if grasped:
                _place_crate(model, data, ids, _crate_between_hands(model, data, ids))
            else:
                _place_crate(model, data, ids, crate0)
            mujoco.mj_forward(model, data)
            t_abs += POLICY_DT
            if want_grasp and not grasped:
                grasped = True
                _place_crate(model, data, ids, _crate_between_hands(model, data, ids))
                mujoco.mj_forward(model, data)
                print(f"  grasp @ t={t_abs:.2f}s  crate={np.round(data.xpos[ids['crate']], 3)}  "
                      f"hands L={np.round(data.xpos[ids['lh']], 3)} "
                      f"R={np.round(data.xpos[ids['rh']], 3)}")
            if rec is not None:
                rec.capture(data, "push" if (want_grasp and k < 6) else None)
        q_prev = target.copy()

    crate_z1 = float(data.xpos[ids["crate"]][2])
    print(f"done  crate z {crate_z0:.3f} -> {crate_z1:.3f} m  grasped={grasped}  "
          f"basez={data.qpos[2]:.3f}")
    if rec is not None:
        err = rec.close()
        print(f"video -> {out}" if not err else f"ffmpeg: {err}")
        if gif is not None and out is not None and out.is_file():
            import sys
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from render_multi import mp4_to_gif
            mp4_to_gif(out, gif, seconds=9.0, width=880)
            print(f"gif   -> {gif}  ({gif.stat().st_size / 1e6:.1f} MB)")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--gif", type=Path, default=None)
    args = p.parse_args()
    run(args.upstream, args.cache_dir, args.out, args.gif)


if __name__ == "__main__":
    main()
