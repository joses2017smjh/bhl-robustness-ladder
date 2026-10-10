"""Inference-boundary, uncertainty and matched-controller regression checks."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest
from bhl_robust.research.terrain_methods import (ARMS, ControlSettings, MapSettings,
    UncertainElevationMap, ground_map_metrics, path_command)


def native(x=0., y=0., yaw=0., time=1., tracked=True, reset=0):
    c, s = np.cos(yaw), np.sin(yaw)
    t = np.array([[c, -s, 0., x], [s, c, 0., y], [0., 0., 1., 0.], [0., 0., 0., 1.]])
    return {"tracked": tracked, "map_reset_id": reset, "timestamp_s": time, "T_W_I": t.tolist()}


def command(arm, packet=None, points=None, now=1.1):
    if points is None:
        points = np.tile([.4, .7, .2], (25, 1))
    return path_command(packet if packet is not None else native(), np.eye(4), np.eye(4), points, now, arm)


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("packet,now,reason", [
    (native(tracked=False), 1.1, "native_untracked_stop"),
    (native(reset=1), 1.1, "native_unregistered_or_reset_stop"),
    (native(), 1.5, "native_stale_stop"),
    (native(time=2.), 1., "native_stale_stop"),
    (native(x=5.), 1.1, "estimated_goal_stop")])
def test_shared_stops(arm, packet, now, reason):
    value, diagnostic, _ = command(arm, packet, now=now)
    assert np.array_equal(value, np.zeros(3))
    assert diagnostic["reason"] == reason


def test_heading_corrects_lateral_error_baseline_does_not():
    packet = native(y=.3)
    blind, _, _ = command("baseline", packet)
    corrected, _, _ = command("heading", packet)
    assert blind[2] == 0
    assert corrected[2] < 0
    assert corrected[0] == blind[0]


def test_heading_error_is_corrected_without_truth():
    corrected, _, _ = command("heading", native(yaw=.4))
    assert corrected[2] < 0
    assert abs(corrected[2]) <= ControlSettings().maximum_yaw_rate_rad_s


def test_regulation_reduces_speed_under_curvature_and_wall_proximity():
    heading, _, _ = command("heading", native(y=.2))
    curved, _, _ = command("regulated", native(y=.2))
    near_wall, _, _ = command("regulated", points=np.tile([.4, .23, .2], (25, 1)))
    assert 0 < curved[0] < heading[0]
    assert 0 < near_wall[0] < heading[0]


def test_frontal_hazard_stops_regulated_arm():
    points = np.tile([.2, 0., .25], (25, 1))
    regulated, diagnostic, _ = command("regulated", points=points)
    baseline, _, _ = command("baseline", points=points)
    assert not regulated.any()
    assert diagnostic["reason"] == "observed_obstacle_stop"
    assert baseline[0] > 0


def map_settings():
    return MapSettings(x_min_m=0., x_max_m=.4, y_min_m=0., y_max_m=.3,
                       resolution_m=.1, minimum_scan_points=2)


def ground_points(z=0.):
    return np.array([[x, y, z] for x in (.025, .05, .125, .15, .225, .25) for y in (.025, .05, .125, .15)])


def test_wall_column_rejected_unknown_is_not_free():
    points = np.concatenate((ground_points(), np.array([[.35, .05, z] for z in np.linspace(0, .5, 10)])))
    mapping = UncertainElevationMap(map_settings())
    receipt = mapping.update(points, np.eye(4), 1.)
    snapshot = mapping.snapshot(1.1)
    assert receipt["rejected_wall_cells"] == 1
    assert not snapshot["known"][0, 3]
    assert snapshot["hazard"][0, 3]
    assert snapshot["known"][0, 0]
    assert snapshot["height_m"][0, 0] == pytest.approx(0.)


def test_age_increases_uncertainty_then_invalidates_support():
    mapping = UncertainElevationMap(map_settings())
    mapping.update(ground_points(), np.eye(4), 1.)
    early, aged = mapping.snapshot(1.), mapping.snapshot(2.)
    known = early["known"]
    assert np.all(aged["variance_m2"][known] > early["variance_m2"][known])
    assert not mapping.snapshot(3.01)["known"].any()
    assert mapping.snapshot(3.01)["hazard"].all()


def test_pose_variance_and_measurement_variance_are_propagated():
    low = UncertainElevationMap(map_settings())
    high = UncertainElevationMap(map_settings())
    low.update(ground_points(), np.eye(4), 1., pose_variance_m2=1e-5, point_variance_m2=1e-5)
    high.update(ground_points(), np.eye(4), 1., pose_variance_m2=.01, point_variance_m2=.01)
    a, b = low.snapshot(1.), high.snapshot(1.)
    assert np.all(b["variance_m2"][a["known"]] > a["variance_m2"][a["known"]])
    for t in range(2, 20):
        high.update(ground_points(), np.eye(4), float(t), pose_variance_m2=.01)
    assert np.all(high.snapshot(19.)["variance_m2"][a["known"]] >= .01)


def test_point_diagonal_covariance_rotates_into_world_vertical():
    mapping = UncertainElevationMap(map_settings())
    covariance = np.tile([1e-6, 1e-6, .02], (len(ground_points()), 1))
    mapping.update(ground_points(), np.eye(4), 1., point_variance_m2=covariance)
    value = mapping.snapshot(1.)
    assert np.all(value["variance_m2"][value["known"]] >= .02)


@pytest.mark.parametrize("bad", [-1., float("nan"), [0., 1.]])
def test_invalid_measurement_variance_rejected(bad):
    with pytest.raises(ValueError):
        UncertainElevationMap(map_settings()).update(ground_points(), np.eye(4), 1., point_variance_m2=bad)


def test_repeated_and_future_snapshot_timestamps_rejected():
    mapping = UncertainElevationMap(map_settings())
    mapping.update(ground_points(), np.eye(4), 1.)
    with pytest.raises(ValueError):
        mapping.update(ground_points(), np.eye(4), 1.)
    with pytest.raises(ValueError):
        mapping.snapshot(.9)


def test_ground_metrics_exclude_walls_even_if_wall_error_is_huge():
    mapping = UncertainElevationMap(map_settings())
    mapping.update(ground_points(), np.eye(4), 1.)
    snapshot = mapping.snapshot(1.)
    truth = np.zeros_like(snapshot["height_m"])
    ground = snapshot["known"].copy()
    ground[0, 0] = False
    truth[0, 0] = 10000
    result = ground_map_metrics(snapshot, truth, ground)
    assert result["ground_height_rmse_m"] == 0
    assert result["covered_ground_cells"] == int(ground.sum())
    assert result["ground_coverage"] == 1.


def test_no_observed_cells_means_unavailable_error_not_zero():
    mapping = UncertainElevationMap(map_settings())
    snapshot = mapping.snapshot(1.)
    truth = np.zeros_like(snapshot["height_m"])
    result = ground_map_metrics(snapshot, truth, np.ones_like(truth, bool))
    assert result["ground_coverage"] == 0.
    assert result["ground_height_rmse_m"] is None


def test_unknown_hazard_support_excluded_from_recall_but_counted():
    mapping = UncertainElevationMap(map_settings())
    snapshot = mapping.snapshot(1.)
    shape = snapshot["height_m"].shape
    result = ground_map_metrics(snapshot, np.zeros(shape), np.ones(shape, bool), truth_hazard=np.ones(shape, bool))
    assert result["ground_hazard_recall"] is None
    assert result["unknown_ground_hazard_cells"] == np.prod(shape)


def test_paired_campaign_summary_preserves_negative_counts():
    path = Path(__file__).resolve().parents[1]/"scripts/bench/terrain_methods_campaign.py"
    spec = importlib.util.spec_from_file_location("test_terrain_methods_campaign", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = [{"actor": "s0", "terrain": "flat", "seed": 410000, "arm": arm,
             "success": arm == "heading", "fell": False, "collision": arm == "baseline",
             "nonfinite": False, "lateral_path_rmse_m": .2} for arm in ARMS]
    result = module.summarize(rows, "development")
    assert result["paired_vs_baseline"]["heading"]["goal_gain_count"] == 1
    assert result["paired_vs_baseline"]["heading"]["contact_reduction_count"] == 1
    assert not result["confirmation_executed"]
    with pytest.raises(ValueError):
        module.summarize(rows+rows[:1], "development")


def test_ground_ray_reference_ignores_wall_tops_and_restores_groups():
    mujoco = pytest.importorskip("mujoco")
    path = Path(__file__).resolve().parents[1]/"scripts/bench/terrain_methods_campaign.py"
    spec = importlib.util.spec_from_file_location("test_terrain_ground_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom name="terrain_base" type="box" size="1 1 .05" pos="0 0 -.05" group="0"/>
      <geom name="wall_fake_tall" type="box" size=".04 .04 .5" pos=".05 .05 .5" group="0"/>
    </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    groups = model.geom_group.copy()
    truth, ground, hazard = module.evaluator_ground_grid(model, data, map_settings())
    assert ground.all()
    assert np.allclose(truth, 0.)
    assert not hazard.any()
    assert np.array_equal(model.geom_group, groups)


def test_complete_collector_refuses_missing_campaign_cells(tmp_path):
    path = Path(__file__).resolve().parents[1]/"scripts/bench/terrain_methods_campaign.py"
    spec = importlib.util.spec_from_file_location("test_terrain_collect", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import json
    protocol = tmp_path/"protocol.json"
    protocol.write_text(json.dumps({"terrain_methods": {"actors": [
        {"name": f"s{i}", "checkpoint_sha256": str(i)} for i in range(3)]}}))
    with pytest.raises(ValueError, match="0/270"):
        module.collect_campaign(protocol, [tmp_path], tmp_path/"must-not-exist.json")
    assert not (tmp_path/"must-not-exist.json").exists()


def test_new_vertical_obstacle_invalidates_previously_known_ground():
    mapping = UncertainElevationMap(map_settings())
    mapping.update(ground_points(), np.eye(4), 1.)
    assert mapping.snapshot(1.)["known"][0, 0]
    points = np.concatenate((ground_points(), np.array([[.05, .05, z] for z in np.linspace(0, .5, 10)])))
    mapping.update(points, np.eye(4), 1.2)
    snapshot = mapping.snapshot(1.2)
    assert snapshot["blocked"][0, 0]
    assert not snapshot["known"][0, 0]
    assert snapshot["hazard"][0, 0]
