"""Atlas-aware coordinate epochs and the verified-relocalization resume gate.

Synthetic native sequences mirror the three failure types observed in the
processing-schedule v1 bounded-latest stereo episodes (seeds 750000/750001):
same-map relocalization (6/12), active-map reset reusing id 0 (1/12) and a new
Atlas map (5/12). Timestamps follow the retained native timelines.
"""
import math

import numpy as np
import pytest

from bhl_robust.research.native_orb import validate_atlas_telemetry, validate_response
from bhl_robust.research.navigation_sensor_controls import NativeCoordinateEpochTracker
from bhl_robust.research.stereo_relocalization import (
    AtlasCoordinateEpochTracker, ResumeBound, VerifiedRelocalizationGate, classify_losses, integrate_commands,
    planar_pose, tilt_change,
)

STEP = .04


def atlas(map_id=0, init=0, kfs=12, big=0, count=1, reset=False, total=0, changed=False, consistent=True,
          imu=False, atlas_big=None):
    return {"schema": "bhl-orb-atlas-telemetry-v1", "system_map_changed": changed, "atlas_map_count": count,
            "atlas_last_big_change_idx": big if atlas_big is None else atlas_big,
            "current_map_id": map_id, "current_map_init_kf_id": init, "current_map_max_kf_id": init + max(kfs, 0),
            "current_map_keyframes": kfs, "current_map_big_change_idx": big, "current_map_imu_initialized": imu,
            "current_map_inertial_ba1": False, "current_map_inertial_ba2": False, "snapshot_consistent": consistent,
            "active_map_reset": reset, "active_map_resets_total": total,
            "read_only_getters_after_synchronous_track": True}


def frame(t, state=2, legacy=0, telemetry=True, **kw):
    value = {"schema": "bhl-orb-native-frame-v1", "timestamp_s": t, "tracking_state": state, "tracked": state == 2,
             "map_id": legacy if state == 2 or legacy is not None else None,
             "T_W_C": np.eye(4).tolist() if state == 2 else None, "compute_seconds": .01}
    if telemetry:
        value["atlas"] = atlas(**kw)
    return value


def tracked_prefix(end=11.0, start=2.2, period=.2):
    return [frame(round(t, 2)) for t in np.arange(start, end + 1e-9, period)]


# Retained v1 timelines (bounded_latest, state/map changes):
#   straight-s750000-camera_10hz: OK map0, RL(map0) 14.2, RL(null) 14.3, OK map0 14.5 ("Relocalized!!")
#   occluders-s750000-nominal:     OK map0, RL(map0) 11.2, RL(null) 11.4, LOST 14.2, OK map0 14.4 (active map reset)
#   straight-s750000-nominal:      OK map0, RL(map0) 21.8, RL(null) 22.0, NO_IMAGES 25.0, OK map1 25.2 (new map)
def same_map_relocalization():
    return tracked_prefix(14.0) + [frame(14.2, 3), frame(14.3, 3, None), frame(14.5), frame(14.6), frame(14.7)]


def active_map_reset_same_id():
    frames = tracked_prefix(11.0, period=.2)
    frames += [frame(11.2, 3, kfs=9), frame(11.4, 3, None, kfs=9), frame(14.2, 4, None, kfs=9),
               frame(14.4, 2, 0, init=37, kfs=1, reset=True, total=1), frame(14.6, 2, 0, init=37, kfs=1, total=1)]
    return frames


def new_atlas_map():
    frames = tracked_prefix(21.6)
    frames += [frame(21.8, 3), frame(22.0, 3, None), frame(25.0, 0, None, map_id=1, init=40, kfs=0, count=2),
               frame(25.2, 2, 1, map_id=1, init=40, kfs=1, count=2), frame(25.4, 2, 1, map_id=1, init=40, kfs=2, count=2)]
    return frames


def epochs(frames, tracker=None):
    tracker = tracker or AtlasCoordinateEpochTracker()
    return [tracker.observe(f) for f in frames], tracker


def at(frames, results, stamp):
    return results[[f["timestamp_s"] for f in frames].index(stamp)]


def test_legacy_tracker_reproduces_v1_false_invalidation_for_every_archetype():
    for sequence, invalidated_at in ((same_map_relocalization(), 14.3), (active_map_reset_same_id(), 11.4),
                                     (new_atlas_map(), 22.0)):
        results, tracker = epochs(sequence, NativeCoordinateEpochTracker())
        events = tracker.receipt()["events"]
        assert [e["timestamp_s"] for e in events] == [invalidated_at]
        assert events[0]["reasons"] == ["native_map_identity_unavailable"]


def test_same_map_relocalization_is_not_an_invalidation():
    frames = same_map_relocalization()
    results, tracker = epochs(frames)
    assert not any(r["continuity_invalidated"] for r in results)
    assert {r["epoch_id"] for r in results} == {0}
    assert tracker.receipt()["coordinate_invalidations"] == 0


def test_active_map_reset_reusing_id_zero_invalidates_when_the_reset_is_visible():
    frames = active_map_reset_same_id()
    results, tracker = epochs(frames)
    # Retained map during RECENTLY_LOST/LOST is not itself an invalidation.
    assert not any(at(frames, results, t)["continuity_invalidated"] for t in (11.2, 11.4, 14.2))
    reset = at(frames, results, 14.4)
    assert reset["invalidation_reasons"] == ["atlas_active_map_reset"]
    assert reset["atlas_map_id"] == 0 and reset["epoch_id"] == 1  # same native id, new coordinate epoch
    assert tracker.receipt()["coordinate_invalidations"] == 1


@pytest.mark.parametrize("variant", ["init_keyframe_only", "flag_only", "emptied_before_reinit"])
def test_each_same_id_reset_signal_alone_fails_closed(variant):
    frames = tracked_prefix(3.0)
    if variant == "init_keyframe_only":
        frames.append(frame(3.2, init=50, kfs=1))
    elif variant == "flag_only":
        frames.append(frame(3.2, reset=True, total=1))
    else:
        frames.append(frame(3.2, 1, None, kfs=0, reset=True, total=1))
    result = epochs(frames)[0][-1]
    assert "atlas_active_map_reset" in result["invalidation_reasons"]
    if variant == "emptied_before_reinit":
        assert "native_returned_to_initialization" in result["invalidation_reasons"]
        assert result["epoch_id"] is None


def test_new_atlas_map_invalidates_at_id_change_and_starts_new_epoch():
    frames = new_atlas_map()
    results, _ = epochs(frames)
    assert not at(frames, results, 22.0)["continuity_invalidated"]
    change = at(frames, results, 25.0)
    assert set(change["invalidation_reasons"]) == {"atlas_map_id_changed", "atlas_map_count_changed",
                                                   "native_returned_to_initialization"}
    assert change["epoch_id"] is None
    assert at(frames, results, 25.2)["epoch_id"] == 1 and not at(frames, results, 25.2)["continuity_invalidated"]


@pytest.mark.parametrize("signal,reason", [
    (dict(big=1), "atlas_big_change"), (dict(changed=True), "atlas_big_change"),
    (dict(atlas_big=1), "atlas_big_change"), (dict(count=2), "atlas_map_count_changed"),
    (dict(consistent=False), "atlas_snapshot_inconsistent"), (dict(imu=True), "atlas_inertial_stage_changed"),
])
def test_big_changes_and_other_atlas_signals_invalidate_even_while_lost(signal, reason):
    frames = tracked_prefix(3.0) + [frame(3.2, 3, None, **signal)]
    result = epochs(frames)[0][-1]
    assert reason in result["invalidation_reasons"]


def test_missing_telemetry_and_map_point_conflict_fail_closed():
    missing = epochs(tracked_prefix(3.0) + [frame(3.2, telemetry=False)])[0][-1]
    assert missing["invalidation_reasons"] == ["atlas_telemetry_missing"] and missing["epoch_id"] is None
    conflict = epochs(tracked_prefix(3.0) + [frame(3.2, legacy=1)])[0][-1]
    assert conflict["invalidation_reasons"] == ["map_point_identity_conflict"] and conflict["epoch_id"] is None


def test_no_epoch_before_an_initialized_map_and_order_is_enforced():
    tracker = AtlasCoordinateEpochTracker()
    assert tracker.observe(frame(.1, 0, None, kfs=0))["epoch_id"] is None
    assert tracker.observe(frame(.2, 1, None, kfs=0))["epoch_id"] is None
    assert not tracker.observe(frame(.3, 2, telemetry=False))["continuity_invalidated"]
    assert tracker.observe(frame(.4))["epoch_id"] == 0
    with pytest.raises(ValueError, match="ordered"):
        tracker.observe(frame(.4))


def test_atlas_block_structure_is_validated_and_round_trips_through_response_validation():
    assert validate_response(frame(1.), 1.)["atlas"]["current_map_id"] == 0
    for broken in (dict(current_map_id=-1), dict(snapshot_consistent=1), dict(atlas_map_count=0),
                   dict(schema="other"), dict(active_map_reset=True, active_map_resets_total=0),
                   dict(read_only_getters_after_synchronous_track=False)):
        value = atlas()
        value.update(broken)
        with pytest.raises(ValueError):
            validate_atlas_telemetry(value)


def test_command_integration_matches_closed_form_and_clips_to_interval():
    commands = [(round(10. + k * STEP, 2), .3, 0., 0.) for k in range(50)]  # 2 s at 0.3 m/s
    straight = integrate_commands(commands, 10.5, 11.5, step_s=STEP)
    assert math.isclose(straight["x"], .3, abs_tol=1e-9) and math.isclose(straight["commanded_path_m"], .3, abs_tol=1e-9)
    partial = integrate_commands(commands, 10.02, 10.06, step_s=STEP)
    assert math.isclose(partial["x"], .3 * .04, abs_tol=1e-12)
    turning = integrate_commands([(0., 0., 0., .45), (STEP, 0., 0., .45)], 0., 1., step_s=STEP)
    assert math.isclose(turning["yaw"], .45 * 2 * STEP) and math.isclose(turning["commanded_rotation_rad"], .036)
    heading = integrate_commands([(0., .3, 0., 0.)], 0., 1., step_s=STEP, start=(1., 2., math.pi / 2))
    assert math.isclose(heading["x"], 1., abs_tol=1e-12) and math.isclose(heading["y"], 2. + .3 * STEP)
    with pytest.raises(ValueError):
        integrate_commands([(1., 0, 0, 0), (1., 0, 0, 0)], 0., 2., step_s=STEP)


def pose(x=0., y=0., yaw=0., z=.6, pitch=0.):
    value = np.eye(4)
    c, s = math.cos(yaw), math.sin(yaw)
    rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    cp, sp = math.cos(pitch), math.sin(pitch)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    value[:3, :3] = rz @ ry
    value[:3, 3] = [x, y, z]
    return value


def run_gate(frames, poses, commands=(), bound=None):
    tracker, gate = AtlasCoordinateEpochTracker(), VerifiedRelocalizationGate(bound, step_s=STEP)
    out = []
    for f in frames:
        epoch = tracker.observe(f)
        out.append(gate.observe(timestamp_s=f["timestamp_s"], tracked=f["tracked"],
                                body_pose=poses.get(f["timestamp_s"], pose()) if f["tracked"] else None,
                                epoch=epoch, commands=list(commands)))
    return out, gate


def walking_poses(frames, speed=.3, start=10., x0=.1):
    return {f["timestamp_s"]: pose(x=x0 + speed * max(0., min(f["timestamp_s"], 14.2) - start)) for f in frames}


def walking_commands(until=14.3, start=10.):
    """Applied commands: route at 0.3 m/s from 10 s; zero once the loss arrives."""
    return [(round(start + k * STEP, 2), .3 if start + k * STEP < until else 0., 0., 0.) for k in range(int(6 / STEP))]


def test_gate_resumes_same_map_relocalization_after_two_consistent_frames():
    frames = same_map_relocalization()
    out, gate = run_gate(frames, walking_poses(frames), walking_commands())
    by_time = dict(zip([f["timestamp_s"] for f in frames], out))
    assert by_time[14.0]["accepted"] and by_time[14.0]["gate_reason"] == "TRACKING"
    assert [by_time[t]["gate_reason"] for t in (14.2, 14.3)] == ["TRACKING_LOSS_STOP"] * 2
    assert by_time[14.5]["gate_reason"] == "CONFIRMING_RELOCALIZATION" and not by_time[14.5]["accepted"]
    assert by_time[14.6]["gate_reason"] == "VERIFIED_RELOCALIZATION_RESUME" and by_time[14.6]["accepted"]
    assert by_time[14.7]["accepted"] and by_time[14.7]["map_reset_id"] == 0
    recovery = gate.receipt()["recoveries"][0]
    assert recovery["loss_start_s"] == 14.2 and math.isclose(recovery["duration_s"], .4)
    # Commanded motion while lost (held command until the loss arrives) is credited.
    assert recovery["checks"][0]["commanded_path_m"] > .05


def test_gate_permanently_stops_on_reset_and_new_map_archetypes():
    for frames, stop_time in ((active_map_reset_same_id(), 14.4), (new_atlas_map(), 25.0)):
        out, gate = run_gate(frames, {}, [])
        times = [f["timestamp_s"] for f in frames]
        stop = out[times.index(stop_time)]
        assert stop["gate_reason"] == "COORDINATE_INVALIDATION" and stop["map_reset_id"] == 1
        assert not any(o["accepted"] for o in out[times.index(stop_time):])
        assert gate.receipt()["permanent_stop"]["timestamp_s"] == stop_time


def test_gate_stops_when_big_change_occurs_during_loss_even_if_id_is_unchanged():
    frames = tracked_prefix(3.0) + [frame(3.2, 3), frame(3.4, 2, big=1), frame(3.6, 2, big=1)]
    out, _ = run_gate(frames, {}, [])
    assert out[-2]["gate_reason"] == "COORDINATE_INVALIDATION" and not out[-1]["accepted"]


def test_gate_rejects_relocalization_inconsistent_with_commanded_motion():
    frames = same_map_relocalization()
    poses = walking_poses(frames)
    poses[14.5] = poses[14.5] @ pose(x=-.4, z=0.)  # recovered 0.4 m behind the dead-reckoned pose
    out, gate = run_gate(frames, poses, walking_commands())
    times = [f["timestamp_s"] for f in frames]
    stop = out[times.index(14.5)]
    assert stop["gate_reason"] == "RELOCALIZATION_INCONSISTENT_WITH_COMMANDED_MOTION"
    assert stop["residual"]["translation_residual_m"] > stop["residual"]["allowed_translation_m"]
    assert not any(o["accepted"] for o in out[times.index(14.5):])


@pytest.mark.parametrize("perturbation", [pose(yaw=math.radians(25), z=0.), pose(z=.15), pose(pitch=math.radians(15), z=0.)])
def test_gate_rejects_yaw_height_or_tilt_inconsistency(perturbation):
    frames = same_map_relocalization()
    poses = walking_poses(frames)
    poses[14.5] = poses[14.5] @ perturbation
    out, _ = run_gate(frames, poses, walking_commands())
    assert out[[f["timestamp_s"] for f in frames].index(14.5)]["gate_state"] == "PERMANENT_STOP"


def test_interrupted_confirmation_waits_instead_of_stopping():
    frames = tracked_prefix(3.0) + [frame(3.2, 3), frame(3.4), frame(3.6, 3, None), frame(3.8), frame(4.0)]
    out, gate = run_gate(frames, {}, [])
    assert [o["gate_reason"] for o in out[-5:]] == ["TRACKING_LOSS_STOP", "CONFIRMING_RELOCALIZATION", "TRACKING_LOSS_STOP",
                                                     "CONFIRMING_RELOCALIZATION", "VERIFIED_RELOCALIZATION_RESUME"]
    assert gate.receipt()["recoveries"][0]["loss_start_s"] == 3.2


def test_invalidation_before_registration_does_not_stop():
    frames = [frame(1.0, 1, None, kfs=0), frame(1.2, 2, init=3, kfs=1), frame(1.4, 0, None, map_id=1, kfs=0, count=2),
              frame(1.6, 2, 1, map_id=1, kfs=1, count=2), frame(1.8, 2, 1, map_id=1, kfs=2, count=2)]
    tracker, gate = AtlasCoordinateEpochTracker(), VerifiedRelocalizationGate(step_s=STEP)
    results = [gate.observe(timestamp_s=f["timestamp_s"], tracked=f["tracked"], body_pose=pose() if f["tracked"] else None,
                            epoch=tracker.observe(f), commands=[]) for f in frames]
    assert results[1]["gate_reason"] == "REGISTERED_INITIAL_EPOCH"
    assert results[2]["gate_reason"] == "COORDINATE_INVALIDATION"  # registered epoch 0 then lost
    tracker, gate = AtlasCoordinateEpochTracker(), VerifiedRelocalizationGate(step_s=STEP)
    late = [frames[0], frames[2], frames[3], frames[4]]
    results = [gate.observe(timestamp_s=f["timestamp_s"], tracked=f["tracked"], body_pose=pose() if f["tracked"] else None,
                            epoch=tracker.observe(f), commands=[]) for f in late]
    assert results[2]["gate_reason"] == "REGISTERED_INITIAL_EPOCH" and results[3]["accepted"]


def test_bound_validation_and_geometry_helpers():
    with pytest.raises(ValueError):
        ResumeBound(confirmation_frames=0)
    with pytest.raises(ValueError):
        ResumeBound(base_translation_m=float("nan"))
    x, y, yaw = planar_pose(pose(1., 2., .5))
    assert (x, y) == (1., 2.) and math.isclose(yaw, .5)
    assert math.isclose(tilt_change(pose(), pose(pitch=.2)), .2)
    assert ResumeBound().confirmation_frames == 2


def packets_from(frames, accepted):
    tracker = AtlasCoordinateEpochTracker()
    return [{"timestamp_s": f["timestamp_s"], "tracked": a, "native_ORB_tracking_state": f["tracking_state"],
             "atlas_epoch": tracker.observe(f)} for f, a in zip(frames, accepted)]


def test_loss_classification_reports_mechanism_and_both_recovery_times():
    frames = same_map_relocalization()
    out, _ = run_gate(frames, walking_poses(frames), walking_commands())
    summary = classify_losses(packets_from(frames, [o["accepted"] for o in out]), 40.)
    assert summary["loss_mechanisms"]["same_map_relocalization"] == 1
    loss = summary["losses"][0]
    assert math.isclose(loss["native_relocalization_s"], .3) and math.isclose(loss["navigation_recovery_s"], .4)
    # Control arm (legacy logic) never resumes: the same native recovery is right-censored.
    legacy = NativeCoordinateEpochTracker()
    control = []
    stopped = False
    for f in frames:
        e = legacy.observe(f)
        stopped |= e["continuity_invalidated"]
        control.append(bool(f["tracked"] and not stopped))
    censored = classify_losses(packets_from(frames, control), 40.)
    assert censored["navigation_recoveries"] == 0 and censored["right_censored_navigation_loss_s"] == [40. - 14.2]
    assert censored["loss_mechanisms"]["same_map_relocalization"] == 1
    for frames, mechanism in ((active_map_reset_same_id(), "active_map_reset"), (new_atlas_map(), "new_atlas_map")):
        out, _ = run_gate(frames, {}, [])
        summary = classify_losses(packets_from(frames, [o["accepted"] for o in out]), 40.)
        assert summary["loss_mechanisms"][mechanism] == 1 and summary["navigation_recoveries"] == 0
