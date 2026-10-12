"""Scheduling campaign integration without native binaries or a physics GPU."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from bhl_robust.research.navigation_sensor_controls import AvailableImu, exposure_image
from bhl_robust.research.processing_schedule import LatestJoinedQueue, decoded_frame_identity


@pytest.fixture
def campaign():
    filename = Path(__file__).parents[1] / "scripts/bench/processing_schedule_campaign.py"
    spec = importlib.util.spec_from_file_location("schedule_campaign_integration_test", filename)
    module = importlib.util.module_from_spec(spec)
    previous = sys.path.copy()
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = previous
    return module


def test_recorder_has_bounded_real_backpressure_and_joins_evidence(campaign):
    recorder = campaign.BoundedRecorder(capacity=1)
    active, release, second_done = threading.Event(), threading.Event(), threading.Event()
    writes, errors = [], []
    def first():
        active.set()
        assert release.wait(3.)
        writes.append("first")
    def submit_second():
        try:
            recorder.submit(writes.append, "second")
            second_done.set()
        except BaseException as error:
            errors.append(error)
    recorder.submit(first)
    worker = threading.Thread(target=submit_second, daemon=True)
    try:
        assert active.wait(1.)
        worker.start()
        assert not second_done.wait(.05)
    finally:
        release.set()
        worker.join(3.)
        recorder.close()
    assert not worker.is_alive() and not errors
    assert writes == ["first", "second"]
    receipt = recorder.receipt()
    assert receipt["joined"] and receipt["pending"] == 0
    assert receipt["maximum_pending"] == 1 and receipt["tasks"] == 2
    assert receipt["backpressure_wall_s"] > 0
    with pytest.raises(ValueError, match="closed"):
        recorder.submit(lambda: None)


def test_recorder_worker_failure_is_visible_to_caller(campaign):
    recorder = campaign.BoundedRecorder(capacity=1)
    def fail():
        raise OSError("evidence device failed")
    recorder.submit(fail)
    with pytest.raises(OSError, match="evidence device"):
        recorder.close()


@pytest.mark.parametrize("capacity", [0, -1, True, 1.5])
def test_recorder_rejects_invalid_capacity_before_starting_worker(campaign, capacity):
    with pytest.raises(ValueError, match="capacity"):
        campaign.BoundedRecorder(capacity=capacity)


def fake_sensors(campaign, directory, pipeline, *, exposure=True):
    sensor = campaign.ScheduledSensors.__new__(campaign.ScheduledSensors)
    sensor.pipeline = pipeline
    sensor.output = directory
    directory.mkdir()
    sensor.inference_output = directory / "selected-inference"
    sensor.inference_output.mkdir()
    sensor.recorder = campaign.BoundedRecorder(capacity=1) if pipeline == "bounded_latest" else None
    sensor.wall_origin = time.monotonic()
    sensor.stage_costs = dict(simulator_ray_imu_s=0., simulator_camera_render_s=0.,
        camera_synthesis_s=0., selected_png_ipc_s=0., evidence_worker_s=0.)
    sensor.images, sensor.requests, sensor.events, sensor.recorded = [], [], [], []
    sensor.selected_images = {}
    sensor.latest_queue = LatestJoinedQueue()
    sensor.last_queue_metadata = None
    sensor.previous_dropped = 0
    sensor.spec = {"exposure_gain": .5}
    sensor.exposure_samples = 3 if exposure else 1
    sensor.exposure_s = .04 if exposure else 0.
    sensor.physics_dt = .005
    sensor.sample_timestamps = [.96, .98, 1.] if exposure else [1.]
    values = [0, 60, 150] if exposure else [90]
    sensor.sample_pixels = [[np.full((4, 6, 3), value + side, np.uint8) for value in values] for side in (0, 1)]
    sensor.runner = SimpleNamespace(d=SimpleNamespace(time=1.2))
    sensor.imu_fault = AvailableImu()
    for stamp in np.arange(.9, 1.305, .005):
        sensor.imu_fault.append(float(stamp), [0., 0., 0.], [0., 0., 9.81])
    sensor.unbilled, sensor.total_sensor_wall = 5., 5.
    return sensor


def enqueue(sensor, *, event_s=1., lidar_capture_s=1., lidar_id=0):
    cloud = dict(frame_id=lidar_id, frame_timestamp_s=lidar_capture_s)
    while len(sensor.recorded) <= lidar_id:
        sensor.recorded.append(cloud)
    payload = dict(scan=cloud, stereo=sensor.latest_camera, event_timestamp_s=event_s)
    sensor.latest_queue.offer(payload, event_s=event_s,
        camera_id=sensor.latest_camera["frame_id"], camera_capture_s=sensor.latest_camera["timestamp_s"],
        lidar_id=lidar_id, lidar_capture_s=lidar_capture_s)
    if sensor.pipeline == "legacy_cumulative":
        sensor.events.append(1)


@pytest.mark.parametrize("pipeline", ["legacy_cumulative", "bounded_latest"])
def test_camera_exposure_pixels_evidence_and_causal_imu_are_preserved(campaign, tmp_path, monkeypatch, pipeline):
    sensor = fake_sensors(campaign, tmp_path / pipeline, pipeline)
    expected = tuple(exposure_image(side, sensor.sample_timestamps, acquisition_s=1.,
        exposure_s=.045, gain=.5) for side in sensor.sample_pixels)
    try:
        sensor._finish_camera(1.)
        enqueue(sensor)
        scan, imu, dropped = sensor.newest()
        assert scan["stereo"]["timestamp_s"] == 1.
        assert dropped == 0
        assert len(imu["timestamp_s"]) == 21
        assert imu["timestamp_s"][-1] == pytest.approx(1.)
        assert sensor.imu_fault.pending[0][0] > 1.
        decoded = tuple(np.asarray(Image.open(scan["stereo"][name]).convert("RGB")) for name in ("left", "right"))
        for actual, original in zip(decoded, expected):
            np.testing.assert_array_equal(actual, original)
        assert all(not array.flags.writeable for array in sensor.latest_camera["_pixels"])
        if pipeline == "bounded_latest":
            assert scan["additional_sensor_compute_seconds"] == sensor.stage_costs["selected_png_ipc_s"]
            assert scan["additional_sensor_compute_seconds"] < 5.
        else:
            assert scan["additional_sensor_compute_seconds"] == 5.
        monkeypatch.setattr(campaign.BASE.ControlledSensors, "save", lambda self, path: Path(path).write_bytes(b"stub sensor serialization"))
        sensor.save(tmp_path / "inference.npz")
        assert sensor.images[0]["decoded_identity"] == decoded_frame_identity(*expected)
        assert sensor.schedule_receipt["decoded_selected_images_verified"] == [0]
        assert sensor.schedule_receipt["queue"]["conservation_pass"]
        assert sensor.schedule_receipt["recorder"]["joined"]
        assert len(list(sensor.output.glob("*-exposure*.png"))) == 6
    finally:
        if sensor.recorder:
            sensor.recorder.close()


def test_selected_inference_corruption_prevents_episode_evidence_pass(campaign, tmp_path, monkeypatch):
    sensor = fake_sensors(campaign, tmp_path / "candidate", "bounded_latest")
    calls = []
    try:
        sensor._finish_camera(1.)
        enqueue(sensor)
        scan, _, _ = sensor.newest()
        Image.fromarray(np.full((4, 6, 3), 255, np.uint8)).save(scan["stereo"]["left"])
        monkeypatch.setattr(campaign.BASE.ControlledSensors, "save", lambda *args: calls.append(args))
        with pytest.raises(ValueError, match="pixels differ"):
            sensor.save(tmp_path / "inference.npz")
        assert not calls
        assert not (sensor.output / "processing_schedule.json").exists()
    finally:
        sensor.recorder.close()


def test_candidate_reports_superseded_joined_event_without_losing_imu(campaign, tmp_path):
    sensor = fake_sensors(campaign, tmp_path / "candidate", "bounded_latest", exposure=False)
    try:
        sensor._finish_camera(1.)
        enqueue(sensor)
        enqueue(sensor, event_s=1.1, lidar_capture_s=1.1, lidar_id=1)
        _, imu, dropped = sensor.newest()
        assert dropped == 1
        assert len(imu["timestamp_s"]) == 41
        assert sensor.latest_queue.receipt()["streams"]["camera"]["superseded_before_selection"] == 0
        assert sensor.latest_queue.receipt()["streams"]["lidar"]["superseded_before_selection"] == 1
    finally:
        sensor.recorder.close()


def nominal_rows(campaign):
    return [dict(pipeline=p, method=a, condition="nominal", native_tracked_frames=1,
        full_horizon=True, fell=False, collision=False, nonfinite=False)
        for p in campaign.PIPELINES for a in campaign.ARMS]


@pytest.mark.parametrize("failure", ["untracked", "fell", "collision", "nonfinite", "short_horizon", "duplicate", "missing"])
def test_nominal_readiness_rejects_each_failure_without_promoting_degraded_tracking(campaign, failure):
    rows = nominal_rows(campaign)
    affected = rows[0].copy()
    key = campaign.gate_key(affected["pipeline"], affected["method"])
    if failure == "untracked":
        rows[0]["native_tracked_frames"] = 0
    elif failure == "short_horizon":
        rows[0]["full_horizon"] = False
    elif failure == "duplicate":
        rows.append(affected.copy())
    elif failure == "missing":
        rows.pop(0)
    else:
        rows[0][failure] = True
    rows.append(dict(affected, condition="camera_10hz", native_tracked_frames=1000))
    readiness = campaign.nominal_arm_readiness(rows)
    assert readiness[key] is False
    assert sum(readiness.values()) == 3


def smoke(campaign):
    return dict(schema=campaign.SCHEMA, phase="smoke", status="PASS", execution_gate=True,
        completed_cells=campaign.declared_protocol()["smoke"]["cells"], episodes=12,
        source_sha256={"code": "source-hash"}, input_sha256={"actor": "actor-hash"},
        problems=[], unrun_cells=[], arm_development_ready=campaign.nominal_arm_readiness(nominal_rows(campaign)))


@pytest.mark.parametrize("mutation", ["sources", "inputs", "missing_cell", "reordered", "nonboolean_gate", "undeclared_gate", "problem"])
def test_smoke_validation_prevents_reuse_of_wrong_source_or_partial_cohort(campaign, mutation):
    receipt = smoke(campaign)
    sources, inputs = copy.deepcopy(receipt["source_sha256"]), copy.deepcopy(receipt["input_sha256"])
    if mutation == "sources":
        sources["code"] = "changed"
    elif mutation == "inputs":
        inputs["actor"] = "changed"
    elif mutation == "missing_cell":
        receipt["completed_cells"].pop()
    elif mutation == "reordered":
        receipt["completed_cells"].reverse()
    elif mutation == "nonboolean_gate":
        receipt["arm_development_ready"][next(iter(receipt["arm_development_ready"]))] = 1
    elif mutation == "undeclared_gate":
        receipt["arm_development_ready"]["unexpected"] = True
    else:
        receipt["problems"].append("runtime error")
    with pytest.raises(ValueError):
        campaign.validate_smoke(receipt, sources=sources, inputs=inputs)


def test_predeclared_order_keeps_matched_pairs_together_and_counterbalances(campaign):
    protocol = campaign.declared_protocol()
    full, smoke_cells = protocol["development"]["cells"], protocol["smoke"]["cells"]
    assert len(full) == 48 and len(smoke_cells) == 12
    assert {c["seed"] for c in full}.isdisjoint(c["seed"] for c in smoke_cells)
    first = []
    identities = set()
    for left, right in zip(full[::2], full[1::2]):
        key = lambda row: tuple(row[k] for k in ("case", "seed", "condition", "arm"))
        assert key(left) == key(right)
        assert {left["pipeline"], right["pipeline"]} == set(campaign.PIPELINES)
        assert key(left) not in identities
        identities.add(key(left))
        first.append(left["pipeline"])
    assert all(a != b for a, b in zip(first, first[1:]))
    assert {p: first.count(p) for p in campaign.PIPELINES} == {p: 12 for p in campaign.PIPELINES}


def test_paired_summary_retains_negative_goal_outcome_and_incomplete_pair(campaign):
    base = dict(case="straight", seed=750000, condition="nominal", method="stereo", full_horizon=True)
    rows = [dict(base, pipeline="legacy_cumulative", success=True, stale_navigation_fraction=.8),
        dict(base, pipeline="bounded_latest", success=False, stale_navigation_fraction=.2),
        dict(base, case="occluders", pipeline="bounded_latest", success=True, stale_navigation_fraction=.1)]
    result = campaign.paired_summary(rows)
    assert result["complete_pairs"] == 1 and len(result["incomplete_pairs"]) == 1
    assert result["clean_goal_count_difference"] == -1
    assert result["mean_paired_stale_fraction_difference"] == pytest.approx(-.6)
    assert result["pairs"][0]["candidate_clean_goal"] is False
    with pytest.raises(ValueError, match="duplicate"):
        campaign.paired_summary(rows + [rows[0]])


def test_pair_ending_before_navigation_retains_negative_goal_evidence(campaign):
    base = dict(case="straight", seed=750000, condition="nominal", method="stereo",
        full_horizon=False, success=False, stale_navigation_fraction=None)
    rows = [dict(base, pipeline=p) for p in campaign.PIPELINES]
    result = campaign.paired_summary(rows)
    assert result["complete_pairs"] == 1
    assert result["clean_goal_count_difference"] == 0
    assert result["mean_paired_stale_fraction_difference"] is None
    assert result["pairs"][0]["both_full_horizon"] is False


def original_profile_fixture(campaign, directory):
    directory.mkdir()
    sensor = directory / "sensors"
    sensor.mkdir()
    rows = []
    for index in range(40):
        paths = {}
        for side, offset in (("left", 0), ("right", 1)):
            path = sensor / f"{index:05d}-{side}.png"
            pixels = np.full((3, 5, 3), index + offset, np.uint8)
            Image.fromarray(pixels).save(path)
            paths[side] = str(path)
            paths[side + "_sha256"] = campaign.sha256(path)
        rows.append(dict(paths, timestamp_s=2. + index * .2))
    (sensor / "acquisition.json").write_text(json.dumps(dict(images=rows)))
    source = directory / "inference_sensors.npz"
    np.savez(source, imu_timestamp_s=np.arange(2., 10., .005), scan_tail_s=np.arange(2.2, 10., .2),
        points_xyz_m=np.ones((40, 3)), gyro_rad_s=np.zeros((1600, 3)))
    (directory / "episode.json").write_text(json.dumps(dict(inference_sensors_sha256=campaign.sha256(source))))
    return directory


def test_identical_retained_input_profile_checks_both_recording_and_inference_pixels(campaign, tmp_path):
    episode = original_profile_fixture(campaign, tmp_path / "original")
    result = campaign.profile_retained_sequence(episode, tmp_path / "profile")
    assert result["status"] == "PASS" and result["frames"] == 40
    assert result["input_equivalence"]["frames"] == 40
    assert result["input_equivalence"]["status"] == "PASS"
    assert len(result["selected_source_image_sha256"]) == 80
    assert result["original_sensors_sha256"] == campaign.sha256(episode / "inference_sensors.npz")
    for pipeline in campaign.PIPELINES:
        timing = result["timings"][pipeline]
        assert len(timing["selected_transport_and_charged_evidence_s"]) == 40
        assert all(value >= 0 for value in timing["selected_transport_and_charged_evidence_s"])
        assert timing["recorder"]["joined"]
    candidate = result["timings"]["bounded_latest"]["recorder"]
    assert candidate["maximum_pending"] <= candidate["capacity"]
    assert candidate["pending"] == 0


@pytest.mark.parametrize("changed", ["image", "sensor_npz"])
def test_retained_input_corruption_cannot_produce_passing_profile(campaign, tmp_path, changed):
    episode = original_profile_fixture(campaign, tmp_path / "original")
    if changed == "image":
        Image.fromarray(np.full((3, 5, 3), 255, np.uint8)).save(episode / "sensors/00017-left.png")
    else:
        with (episode / "inference_sensors.npz").open("ab") as stream:
            stream.write(b"corruption")
    output = tmp_path / "profile"
    with pytest.raises(ValueError, match="hash mismatch"):
        campaign.profile_retained_sequence(episode, output)
    assert not (output / "campaign_result.json").exists()
