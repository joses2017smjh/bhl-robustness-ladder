"""Geometry and evidence-leak negative controls for sensor research routes."""
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from bhl_robust.research.sensor_geometry import (Pinhole, FusionConfig, TerrainConfig,
    project_lidar, transform_points, rigid_transform, stressed_extrinsic,
    fuse_depth, depth_metrics, risk_coverage, retained_indices, sector_minima,
    elevation_map, terrain_metrics)
spec = importlib.util.spec_from_file_location("sensor_geometry_research", ROOT / "scripts/bench/sensor_geometry_research.py")
research = importlib.util.module_from_spec(spec)
spec.loader.exec_module(research)


def test_projection_uses_optical_z_and_keeps_unknown_nan():
    result = project_lidar(np.array([[1., 0., 2.]]), np.eye(4), Pinhole(3, 3, 2, 2, 1, 1))
    assert result["depth_z_m"][1, 2] == 2.
    assert np.isnan(result["depth_z_m"][0, 0])
    assert result["occupied_pixels"] == 1


def test_zbuffer_near_surface_and_frustum_rejections():
    points = np.array([[0, 0, 3], [0, 0, 1], [0, 0, -1],
                       [np.nan, 0, 1], [10000000000., 0, 1], [0, 0, 31]])
    result = project_lidar(points, np.eye(4), Pinhole(3, 3, 2, 2, 1, 1))
    assert result["depth_z_m"][1, 1] == 1
    assert result["point_count"][1, 1] == 2
    assert result["projected_points"] == 2


@pytest.mark.parametrize("kind", ["reflection", "scale", "lastrow", "nan"])
def test_invalid_transforms_are_rejected(kind):
    transform = np.eye(4)
    if kind == "reflection":
        transform[0, 0] = -1
    elif kind == "scale":
        transform[0, 0] = 2
    elif kind == "lastrow":
        transform[3, 0] = .1
    else:
        transform[0, 3] = np.nan
    with pytest.raises(ValueError):
        rigid_transform(transform)


def test_transform_direction_and_real_extrinsic_stress():
    transform = np.eye(4)
    transform[0, 3] = 1
    assert np.allclose(transform_points([[0, 0, 2]], transform), [[1, 0, 2]])
    changed = stressed_extrinsic(np.eye(4), rotation_y_deg=90, translation_x_m=.1)
    assert np.allclose(transform_points([[0, 0, 2]], changed), [[2.1, 0, 0]], atol=1e-10)


def test_gate_visibility_disagreement_and_confidence_rejection():
    stereo = np.array([[2., 5., 2., 3., np.nan, np.nan]])
    lidar = np.array([[2.1, 2., 8., np.nan, 4., np.nan]])
    confidence = np.array([[1., 1., 1., .2, 0., 0.]])
    result = fuse_depth(stereo, lidar, confidence)
    gate = result["arms"]["confidence_gated"][0]
    assert 2 < gate[0] < 2.1
    assert np.isnan(gate[1])  # Conflicting near lidar is not blindly trusted.
    assert gate[2] == 2      # Confident visible stereo occludes far lidar.
    assert np.isnan(gate[3])
    assert gate[4] == 4
    assert np.isnan(gate[5]) # Unknown remains unknown for every method.
    for arm in result["arms"].values():
        assert np.isnan(arm[0, 5])
    assert result["diagnostics"]["conflict_unknown_pixels"] == 1
    assert result["diagnostics"]["lidar_occluded_pixels"] == 1


def test_reject_all_cannot_hide_missed_obstacles():
    result = depth_metrics([[np.nan, np.nan]], [[2., 4.]], obstacle_mask=np.ones((1, 2), dtype=bool))
    assert result["false_free_space_rate"] == 0
    assert result["obstacle_recall"] == 0
    assert result["near_missed_pixels"] == 1
    assert result["near_unknown_pixels"] == 1
    assert result["coverage_on_gt"] == 0
    assert result["rmse_m"] is None


def test_false_free_space_and_false_obstacle_both_reported():
    result = depth_metrics([[4., 2.]], [[2., 4.]], obstacle_mask=np.ones((1, 2), dtype=bool))
    assert result["false_free_space_pixels"] == 1
    assert result["false_obstacle_pixels"] == 1
    assert result["obstacle_precision"] == 0
    assert result["rmse_m"] == 2
    assert result["obstacle_recall"] == 0


def test_risk_coverage_preserves_unknown_denominator():
    curve = risk_coverage([[2., 4.]], [[.2, .9]], [[2., 4.]], obstacle_mask=np.ones((1, 2), dtype=bool))
    assert curve[0]["coverage_on_gt"] == 1
    assert curve[-1]["coverage_on_gt"] == .5
    assert curve[-1]["near_unknown_pixels"] == 1


def test_invalid_confidence_is_not_accepted_as_certainty():
    with pytest.raises(ValueError, match="confidence"):
        fuse_depth([[2.]], [[2.]], [[np.nan]])
    with pytest.raises(ValueError, match="confidence"):
        fuse_depth([[2.]], [[2.]], [[1.1]])


def test_floor_excluded_from_obstacle_recall_and_unlabelled_proxy_explicit():
    result = depth_metrics([[1., 4.]], [[1., 2.]], obstacle_mask=np.array([[False, True]]))
    assert result["near_gt_pixels"] == 1
    assert result["near_missed_pixels"] == 1
    assert result["false_free_space_rate"] == 1
    assert result["predicted_near_pixels"] == 0
    proxy = depth_metrics([[1., 4.]], [[1., 2.]])
    assert proxy["obstacle_recall"] is None
    assert proxy["false_free_space_rate"] is None
    assert proxy["obstacle_metric_scope"] == "near_surface_proxy"
    assert proxy["near_surface_proxy_recall"] == .5


def test_nested_dropout_masks_reuse_exact_points():
    full = retained_indices(501, 1., 42)
    half = retained_indices(501, .5, 42)
    low = retained_indices(501, .2, 42)
    assert len(full) == 501 and len(half) == 250 and len(low) == 100
    assert set(low).issubset(half) and set(half).issubset(full)
    assert np.array_equal(low, retained_indices(501, .2, 42))


def test_nondivisible_sector_pool_keeps_last_ray_and_unknowns():
    angles = np.linspace(-np.pi, np.pi, 501, endpoint=False)
    ranges = np.full(501, 10.)
    ranges[-1] = .2
    result = sector_minima(ranges, angles, 36)
    assert result[-1] == .2  # Old 500//36 pooling would discard this ray.
    assert np.isnan(sector_minima([np.nan], [0.], 36)).all()


def plane_fixture(a=.2, b=.1):
    config = TerrainConfig(x_min_m=-1, x_max_m=1, y_min_m=-1, y_max_m=1,
                           resolution_m=.25)
    x, y = config.axes()
    gx, gy = np.meshgrid(x, y)
    points = np.column_stack((gx.ravel(), gy.ravel(), (a * gx + b * gy).ravel()))
    return config, points, a * gx + b * gy


def test_plane_height_slope_and_residual_reconstruction():
    config, points, truth = plane_fixture()
    terrain = elevation_map(points, config)
    assert np.allclose(terrain["height_m"], truth)
    assert terrain["known"].all()
    assert np.allclose(terrain["slope_deg"], math.degrees(math.atan(math.hypot(.2, .1))))
    assert np.max(terrain["roughness_m"]) < 1e-10
    metrics = terrain_metrics(terrain, truth, np.zeros(truth.shape, dtype=bool))
    assert metrics["height_rmse_m"] == 0
    assert metrics["hazard_recall"] is None
    assert metrics["traversal_outcomes"] is None


def test_steep_known_plane_hazard_detection():
    config, points, truth = plane_fixture(a=1, b=0)
    terrain = elevation_map(points, config)
    assert terrain["hazard"].all()
    metrics = terrain_metrics(terrain, truth, np.ones(truth.shape, dtype=bool))
    assert metrics["hazard_recall"] == 1


def test_sparse_or_collinear_terrain_is_unknown_not_flat_safe():
    config = TerrainConfig(x_min_m=-1, x_max_m=1, y_min_m=-1, y_max_m=1, resolution_m=.25)
    terrain = elevation_map([[0, 0, 0]], config)
    assert np.count_nonzero(np.isfinite(terrain["height_m"])) == 1
    assert not terrain["known"].any()
    assert not terrain["hazard"].any()
    with pytest.raises(ValueError, match="3D lidar"):
        elevation_map([[0, 0, 0]], config, lidar_dimension=2)


def test_namespaces_prevent_truth_symlink_and_relative_escape(tmp_path):
    (tmp_path / "inference").mkdir()
    (tmp_path / "evaluator").mkdir()
    truth = tmp_path / "evaluator" / "truth.npy"
    np.save(truth, np.ones((1, 1)))
    alias = tmp_path / "inference" / "alias.npy"
    alias.symlink_to(truth)
    with pytest.raises(ValueError, match="escapes|symbolic"):
        research.namespaced_path(tmp_path, "inference/alias.npy", "inference")
    with pytest.raises(ValueError, match="relative"):
        research.namespaced_path(tmp_path, "../truth.npy", "inference")
    with pytest.raises(ValueError, match="truth"):
        research.prediction_path(tmp_path, "evaluator/truth.npy", tmp_path)


def test_terrain_labels_require_matching_frame_and_outside_is_unknown():
    config, points, _ = plane_fixture()
    terrain = elevation_map(points, config)
    truth = {"terrain_x_m": np.array([-.3, .3]), "terrain_y_m": np.array([-.3, .3]),
             "terrain_height_m": np.zeros((2, 2)), "terrain_grid_frame": np.array("world")}
    with pytest.raises(ValueError, match="local z-up"):
        research.interpolation_labels(truth, terrain)
    truth["terrain_grid_frame"] = np.array("M_local")
    labels, hazard = research.interpolation_labels(truth, terrain)
    assert np.isnan(labels[0]).all() and np.isnan(labels[:, 0]).all()
    assert hazard is None


def dataset_fixture(tmp_path):
    dataset, predictions = tmp_path / "dataset", tmp_path / "stereo"
    (dataset / "inference").mkdir(parents=True)
    (dataset / "evaluator").mkdir()
    (predictions / "predictions").mkdir(parents=True)
    sequences, hashes, pred_records = [], {}, []
    for index, split in enumerate(("development", "validation", "test")):
        frame = {"frame_id": 0, "timestamp_s": 0., "left": f"inference/{index}-left.png",
                 "right": f"inference/{index}-right.png", "lidar": f"inference/{index}-lidar.npz",
                 "imu": f"inference/{index}-imu.npz", "truth": f"evaluator/{index}-truth.npz"}
        for side in ("left", "right"):
            (dataset / frame[side]).write_bytes(b"test-fixture")
        np.savez(dataset / frame["lidar"], points_xyz_m=np.array([[0., 0., 2.]]), point_time_s=np.array([0.]))
        np.savez(dataset / frame["imu"], gyro_rad_s=np.zeros(3), accel_m_s2=np.zeros(3))
        np.savez(dataset / frame["truth"], depth_z_m=np.full((3, 3), 2.),
                 depth_truth_source=np.array("known_scene_geometry"),
                 obstacle_mask=np.ones((3, 3), dtype=bool),
                 obstacle_truth_source=np.array("independent_scene_geom_segmentation"),
                 terrain_grid_frame=np.array("M_local"), terrain_x_m=np.array([-1., 1.]),
                 terrain_y_m=np.array([-1., 1.]), terrain_height_m=np.zeros((2, 2)),
                 terrain_hazard=np.zeros((2, 2), dtype=bool))
        for field in ("left", "right", "lidar", "imu", "truth"):
            hashes[frame[field]] = research.sha256(dataset / frame[field])
        for field, array in (("depth", np.full((3, 3), 4.)), ("valid", np.ones((3, 3), dtype=bool)),
                             ("confidence", np.ones((3, 3)))):
            np.save(predictions / f"predictions/{index}-{field}.npy", array)
        pred_records.append({"sequence_id": f"s{index}", "frame_id": 0, "method": "sgbm",
                             **{field: f"predictions/{index}-{field}.npy" for field in ("depth", "valid", "confidence")}})
        sequences.append({"id": f"s{index}", "scene_id": f"scene{index}", "split": split,
                          "origin": "simulation", "frames": [frame]})
    manifest = {"origin": "simulation", "data_kind": "mujoco_rendered_rgb_timed_ray_lidar",
                "lidar_dimension": 3, "file_sha256": hashes, "sequences": sequences,
                "calibration": {"image_width": 3, "image_height": 3, "rectified": True,
                    "fx_px": 2., "fy_px": 2., "cx_left_px": 1., "cy_px": 1.,
                    "T_C_L": np.eye(4).tolist(), "T_M_L": np.eye(4).tolist()}}
    (dataset / "manifest.json").write_text(json.dumps(manifest))
    (predictions / "predictions.json").write_text(json.dumps({"schema": "bhl-stereo-predictions-v1",
                                                            "predictions": pred_records}))
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({"sensor_geometry": {"terrain": {"x_min_m": -.5, "x_max_m": .5,
        "y_min_m": -.5, "y_max_m": .5, "resolution_m": .5}}}))
    return dataset, predictions / "predictions.json", protocol


def test_replay_reports_independent_safety_denominators_and_execution_pass(tmp_path):
    dataset, predictions, protocol = dataset_fixture(tmp_path)
    result = research.run(protocol, dataset, predictions, tmp_path / "output")
    assert result["status"] == "PASS" and result["frames"] == 3
    assert result["terrain"]["status"] == "PASS"
    assert result["terrain"]["traversal_outcomes"] is None
    nominal_stereo = next(row for row in result["fusion"]["summary"] if row["split"] == "test" and
        row["stress"] == "nominal" and row["retention"] == 1 and row["arm"] == "stereo")
    assert nominal_stereo["metrics"]["false_free_space_rate"] == 1
    assert nominal_stereo["metrics"]["obstacle_recall"] == 0
    assert json.loads((tmp_path / "output/campaign_result.json").read_text())["status"] == "PASS"


def test_corrupt_input_checksum_refuses_replay(tmp_path):
    dataset, predictions, protocol = dataset_fixture(tmp_path)
    (dataset / "inference/0-left.png").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        research.run(protocol, dataset, predictions, tmp_path / "output")


def test_scene_split_leakage_refuses_replay(tmp_path):
    dataset, predictions, protocol = dataset_fixture(tmp_path)
    manifest = json.loads((dataset / "manifest.json").read_text())
    manifest["sequences"][2]["scene_id"] = "scene0"
    (dataset / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="scene leakage"):
        research.run(protocol, dataset, predictions, tmp_path / "output")


def test_missing_dataset_exits_incomplete_without_inventing_results(tmp_path):
    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}")
    output = tmp_path / "output"
    code = research.main(["--protocol", str(protocol), "--dataset", str(tmp_path / "missing"),
                          "--output", str(output)])
    assert code == 2
    assert json.loads((output / "campaign_result.json").read_text())["status"] == "INCOMPLETE"
