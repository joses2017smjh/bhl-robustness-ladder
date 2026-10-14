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


def test_early_safety_failures_remain_negative_without_fake_stale_fraction(campaign):
    rows = [dict(method="stereo", pipeline=pipeline, success=False, fell=True,
                 collision=False, stale_navigation_fraction=None)
            for pipeline in campaign.PIPELINES]
    groups = campaign.aggregate_groups(rows)
    for pipeline in campaign.PIPELINES:
        group = groups[pipeline + "/stereo"]
        assert group["episodes"] == 1 and group["falls"] == 1 and group["clean_goals"] == 0
        assert group["mean_stale_fraction"] is None
    json.dumps(groups, allow_nan=False)


# --- order-confirmation-20261014: counterbalanced design, warm-ups and seeds ---

import hashlib
import shutil
import subprocess
from collections import Counter
from functools import lru_cache

from scipy.stats import binom, binomtest

REPO = Path(__file__).parents[1]
V1_PROTOCOL = REPO / "results/processing-schedule-20261012/scientific-protocol.json"


def dispatch_module():
    spec = importlib.util.spec_from_file_location("schedule_dispatch_for_campaign_test", REPO / "scripts/bench/processing_schedule_dispatch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v1_protocol_is_reproduced_byte_for_byte(campaign, tmp_path):
    for protocol in (campaign.declared_protocol(), campaign.declared_protocol(campaign.V1_DESIGN)):
        target = tmp_path / f"{len(list(tmp_path.iterdir()))}.json"
        campaign.NAV.write_json(target, protocol)
        assert target.read_bytes() == V1_PROTOCOL.read_bytes()
    assert hashlib.sha256(V1_PROTOCOL.read_bytes()).hexdigest() == "b41d516b7810fa851deea4c160c1efa2eff0d75e39037bb7bb24f265d2e5d34d"
    assert campaign.protocol_design(json.loads(V1_PROTOCOL.read_text())) == campaign.V1_DESIGN
    with pytest.raises(ValueError, match="unknown"):
        campaign.declared_protocol("order-confirmation-draft")


def order_protocol(campaign):
    return campaign.declared_protocol(campaign.ORDER_DESIGN)


def test_order_design_counterbalances_first_position_within_every_arm_and_stratum(campaign):
    protocol = order_protocol(campaign)
    cells = protocol["development"]["cells"]
    assert protocol["arms"] == ["lidar_inertial"] and len(cells) == 96
    pairs = list(zip(cells[::2], cells[1::2]))
    key = lambda c: (c["case"], c["seed"], c["condition"], c["arm"])
    assert len({key(a) for a, _ in pairs}) == 48
    for index, (first, second) in enumerate(pairs):
        assert key(first) == key(second) and first["pair_index"] == second["pair_index"] == index
        assert {first["pipeline"], second["pipeline"]} == set(campaign.PIPELINES)
        assert (first["position"], second["position"]) == (1, 2)
        assert first["first_pipeline"] == second["first_pipeline"] == first["pipeline"]
    leads = [a for a, _ in pairs]
    def balanced(rows):
        counts = Counter(r["pipeline"] for r in rows)
        return counts[campaign.PIPELINES[0]] == counts[campaign.PIPELINES[1]] == len(rows) / 2
    for arm in protocol["arms"]:
        assert balanced([r for r in leads if r["arm"] == arm])
        for field in ("condition", "case", "seed"):
            for value in {r[field] for r in leads}:
                assert balanced([r for r in leads if r["arm"] == arm and r[field] == value]), (field, value)
        for condition in campaign.CONDITIONS:
            for case in campaign.CASES:
                assert balanced([r for r in leads if (r["arm"], r["condition"], r["case"]) == (arm, condition, case)])
    assert protocol["counterbalance"]["first_pipeline_counts_by_arm"] == {"lidar_inertial": {p: 24 for p in campaign.PIPELINES}}
    shard_leads = []
    for shard in protocol["development"]["shards"]:
        scored = [c for c in cells if c["shard"] == shard["name"]]
        sequence = [c["pipeline"] for c in scored]
        assert [c["case"] for c in scored] == ["straight", "straight", "occluders", "occluders"]
        assert sequence[:2] == sequence[:1:-1]  # ABBA / BAAB within every shard
        assert [w["pipeline"] for w in shard["warmup_cells"]] == sequence[:2]
        shard_leads.append(sequence[0])
    assert Counter(shard_leads) == {p: 12 for p in campaign.PIPELINES}


def test_warmups_are_unscored_by_construction_and_run_first(campaign):
    protocol = order_protocol(campaign)
    cells, shards = protocol["development"]["cells"], protocol["development"]["shards"]
    scored_seeds = {c["seed"] for c in cells}
    warmups = protocol["smoke"]["warmup_cells"] + [w for s in shards for w in s["warmup_cells"]]
    assert len(warmups) == 50 == protocol["budget"]["warmup_episodes"]
    assert {w["seed"] for w in warmups} == {campaign.WARMUP_SEED} and campaign.WARMUP_SEED not in scored_seeds
    assert all(w["role"] == "warmup" and w["case"] == campaign.WARMUP_CASE for w in warmups)
    assert all(c["role"] == "scored" for c in cells) and all(c["role"] == "smoke" for c in protocol["smoke"]["cells"])
    assert not any(w in cells or w in protocol["smoke"]["cells"] for w in warmups)
    assert protocol["warmup"]["scored"] is False and protocol["warmup"]["raw_images_archived"] is False
    for shard in shards:
        assert [(w["condition"], w["arm"], w["shard"]) for w in shard["warmup_cells"]] == [(shard["condition"], shard["arm"], shard["name"])] * 2
        indices = [w["shard_execution_index"] for w in shard["warmup_cells"]] + [c["shard_execution_index"] for c in cells if c["shard"] == shard["name"]]
        assert indices == list(range(6))
    smoke = protocol["smoke"]
    assert [x["shard_execution_index"] for x in smoke["warmup_cells"] + smoke["cells"]] == list(range(8))
    assert {x["condition"] for x in smoke["warmup_cells"]} == {"nominal"} and len(smoke["cells"]) == 6
    # Readiness uses scored smoke rows only; a nominal warm-up row would break its count.
    rows = [dict(pipeline=c["pipeline"], method=c["arm"], condition=c["condition"], native_tracked_frames=1,
                 full_horizon=True, fell=False, collision=False, nonfinite=False) for c in smoke["cells"]]
    assert all(campaign.nominal_arm_readiness(rows, ("lidar_inertial",)).values())
    warm_rows = [dict(r, condition="nominal") for r in rows[:2]]
    assert not any(campaign.nominal_arm_readiness(rows + warm_rows, ("lidar_inertial",)).values())


def test_new_seeds_are_disjoint_and_appear_only_in_the_new_design_files(campaign):
    new = set(campaign.ORDER_SEEDS) | {campaign.WARMUP_SEED}
    assert len(campaign.ORDER_SEEDS) == 8 and not new & ({campaign.SMOKE_SEED} | set(campaign.DEVELOPMENT_SEEDS))
    v1 = json.loads(V1_PROTOCOL.read_text())
    assert not new & {c["seed"] for phase in ("smoke", "development") for c in v1[phase]["cells"]}
    if not (REPO / ".git").exists():
        pytest.skip("seed provenance scan needs the git checkout")
    patterns = [arg for seed in sorted(new) for arg in ("-e", str(seed))]
    proc = subprocess.run(["git", "-C", str(REPO), "grep", "--untracked", "-l", "-w", *patterns],
                          capture_output=True, text=True)
    assert proc.returncode in (0, 1), proc.stderr
    allowed = {"scripts/bench/processing_schedule_campaign.py", "scripts/bench/processing_schedule_dispatch.py",
               "tests/test_processing_schedule_campaign.py", "tests/test_processing_schedule_dispatch.py",
               "tests/test_processing_schedule_order_analysis.py", "docs/PROCESSING_SCHEDULE_ORDER_CONFIRMATION_2026-10-14.md"}
    assert all(path in allowed or path.startswith("results/processing-schedule-order-") for path in proc.stdout.split())


@lru_cache(None)
def _rejections(discordant):
    return tuple(k for k in range(discordant + 1) if binomtest(k, discordant, .5).pvalue < .05) if discordant else ()


def mcnemar_power(n, bounded_only, legacy_only):
    p_discordant = bounded_only + legacy_only
    share = bounded_only / p_discordant
    return sum(binom.pmf(d, n, p_discordant) * sum(binom.pmf(k, d, share) for k in _rejections(d)) for d in range(n + 1))


def test_predeclared_power_table_matches_exact_mcnemar(campaign):
    for name, scenario in campaign.ORDER_POWER_SCENARIOS.items():
        for n, value in scenario["power"].items():
            assert round(mcnemar_power(int(n), scenario["bounded_only"], scenario["legacy_only"]), 3) == pytest.approx(value, abs=1e-9), (name, n)


def test_order_counts_and_dispatcher_constants_derive_from_the_declared_design(campaign):
    protocol = order_protocol(campaign)
    D = dispatch_module()
    spec = D.design_spec(D.ORDER_DESIGN)
    cells, shards = protocol["development"]["cells"], protocol["development"]["shards"]
    assert D.ORDER_DESIGN == campaign.ORDER_DESIGN and D.V1_DESIGN == campaign.V1_DESIGN
    assert tuple(D.ORDER_SEEDS) == campaign.ORDER_SEEDS == tuple(protocol["development"]["seeds"])
    assert tuple(D.ORDER_ARMS) == tuple(protocol["arms"]) and tuple(D.CONDITIONS) == campaign.CONDITIONS and tuple(D.CASES) == campaign.CASES
    assert spec["development_cells"] == len(cells) == protocol["budget"]["scored_episodes"] == 2 * protocol["budget"]["pairs"]
    assert spec["shards"] == len(shards) == protocol["budget"]["development_shards"]
    assert spec["cells_per_shard"] == len(cells) // len(shards) == 4
    assert spec["smoke_cells"] == len(protocol["smoke"]["cells"]) == protocol["budget"]["smoke_episodes"]
    assert spec["smoke_warmups"] == len(protocol["smoke"]["warmup_cells"])
    assert spec["warmups_per_shard"] == len(shards[0]["warmup_cells"]) and spec["warmup_episodes"] == protocol["budget"]["warmup_episodes"]
    assert spec["cap_bytes"] == protocol["budget"]["campaign_cap_gib"] * 1024**3
    assert {k: spec["resources"][k] for k in ("partition", "constraint", "cpus", "memory_gb", "gpus", "time_limit")} == \
        {k: protocol["resources"][k] for k in ("partition", "constraint", "cpus", "memory_gb", "gpus", "time_limit")}
    assert protocol["primary_outcome"]["pairs"] == 48 and protocol["decision_rule"]["CONFIRMED"].startswith("all 48 pairs")
    jobs = D.declared_jobs(D.SHARED_ROOT / "unit-campaign", D.ORDER_DESIGN)
    shard_args = [tuple(j["args"][j["args"].index(k) + 1] for k in ("--seed", "--condition", "--arm")) for j in jobs if j["kind"] == "run"]
    assert shard_args == [(str(s["seed"]), s["condition"], s["arm"]) for s in shards]
    v1 = campaign.declared_protocol()
    assert len(v1["development"]["cells"]) == D.design_spec(D.V1_DESIGN)["development_cells"] == 48
    assert len(v1["smoke"]["cells"]) == D.design_spec(D.V1_DESIGN)["smoke_cells"] == 12


def test_shard_plan_returns_declared_execution_order_and_rejects_undeclared_shards(campaign):
    protocol = order_protocol(campaign)
    shard = protocol["development"]["shards"][4]
    warmups, cells = campaign.shard_plan(protocol, "development", seed=shard["seed"], condition=shard["condition"], arm=shard["arm"])
    assert warmups == shard["warmup_cells"] and [c["shard_execution_index"] for c in cells] == [2, 3, 4, 5]
    assert campaign.shard_plan(protocol, "smoke") == (protocol["smoke"]["warmup_cells"], protocol["smoke"]["cells"])
    for kwargs in (dict(seed=shard["seed"], condition=shard["condition"], arm=shard["arm"], case="straight"),
                   dict(seed=shard["seed"], condition=shard["condition"]),
                   dict(seed=750000, condition="nominal", arm="lidar_inertial"),
                   dict(seed=shard["seed"], condition=shard["condition"], arm="stereo")):
        with pytest.raises(ValueError):
            campaign.shard_plan(protocol, "development", **kwargs)
    with pytest.raises(ValueError):
        campaign.shard_plan(protocol, "smoke", seed=749000, condition="nominal")
    v1 = campaign.declared_protocol()
    assert campaign.shard_plan(v1, "development", seed=750001, condition="nominal", case="straight", arm="stereo")[1] == \
        [c for c in v1["development"]["cells"] if (c["seed"], c["condition"], c["case"], c["arm"]) == (750001, "nominal", "straight", "stereo")]
    with pytest.raises(ValueError, match="no declared cells"):
        campaign.shard_plan(v1, "development", seed=751000, condition="nominal")


def fake_run_episode(campaign, outcome):
    """Internally consistent minimal episodes for control-flow tests (no physics)."""
    from bhl_robust.research.processing_schedule import service_timing
    def run_episode(upstream, actor, runtime, case, seed, output, *, condition, arm, seconds, pipeline, execution=None):
        output.mkdir(parents=True, exist_ok=False)
        bounded = pipeline == "bounded_latest"
        success = outcome(pipeline, seed, condition, case, seconds)
        timing = service_timing(request_s=10., capture_s=9.95, preprocessing_s=.01, native_client_s=.02, policy_dt=.04)
        request = dict(request_s=10., camera_id=0, lidar_id=0, camera_capture_s=9.95, lidar_capture_s=9.95,
                       preprocessing_service_s=.01, queue=dict(enqueue_s=9.95, queue_residence_s=.05))
        stream = dict(produced_unique=1, selected_unique=1, superseded_before_selection=0, pending_unselected=0, dispatches=1, reused_dispatches=0)
        evidence = dict(pipeline=pipeline, requests=[request], camera_captures_total=1, lidar_captures_total=1,
            camera_never_joined=[], lidar_never_joined=[],
            queue=dict(produced=1, selected=1, superseded=0, pending=0, maximum_waiting_events=1, streams=dict(camera=dict(stream), lidar=dict(stream))),
            recorder=dict(capacity=8, maximum_pending=1, tasks=1, joined=True, pending=0) if bounded else dict(mode="synchronous", joined=True))
        campaign.NAV.write_json(output / "sensors/processing_schedule.json", evidence)
        campaign.NAV.write_json(output / "native/stream_receipt.json", dict(stub=True))
        (output / "inference_sensors.npz").write_bytes(b"sensor arrays")
        (output / "evaluator_truth.npz").write_bytes(b"evaluator arrays")
        trace = [dict(decision_timestamp_s=t, reason="navigate" if success or t < 10 else "native_stale_stop",
                      evaluator_goal_distance_m=.1 if success and t > 10 else 3., evaluator_contact=False,
                      evaluator_tilt_rad=0., evaluator_height_m=.7) for t in (0., 10., seconds - .04)]
        stale = sum(t["reason"] == "native_stale_stop" for t in trace[1:])
        row = dict(case=case, seed=seed, condition=condition, method=arm, pipeline=pipeline, success=success,
            goal_reached_evaluator=success, fell=False, collision=False, nonfinite=False, full_horizon=True,
            elapsed_simulation_s=seconds, evaluator_initial_height_m=.7, native_frames=1, native_tracked_frames=1,
            native_imu_initialized_frames=0, trace=trace, navigation_decisions=2, stale_navigation_decisions=stale,
            stale_navigation_fraction=stale / 2, schedule_timing=[timing],
            response_deliveries=[dict(arrival_s=timing["delivery_s"], wall_seconds=timing["service_s"])],
            responses_still_pending_at_horizon=0, latency_quantization_s=.04, acquisition_schedule=evidence,
            client_wall_p95_ms=25. if bounded else 20., full_pipeline_p95_ms=60. if bounded else 900.,
            navigation_pose_age_p95_ms=120. if bounded else 1500.,
            **{field: campaign.sha256(output / name) for name, field in (("inference_sensors.npz", "inference_sensors_sha256"),
                ("evaluator_truth.npz", "evaluator_truth_sha256"), ("native/stream_receipt.json", "stream_receipt_sha256"),
                ("sensors/processing_schedule.json", "processing_schedule_sha256"))})
        if execution is not None:
            row["execution"] = dict(execution, started_utc="2026-10-14T00:00:00+00:00", finished_utc="2026-10-14T00:00:01+00:00", wall_seconds=1.)
        campaign.NAV.write_json(output / "episode.json", row)
        return row
    return run_episode


def synthetic_outcome(pipeline, seed, condition, case, seconds):
    if pipeline == "bounded_latest":
        return (seed, condition, case) != (751003, "nominal", "occluders")
    return (seed, condition, case) == (751005, "camera_10hz", "straight")


def shard_runner(campaign, tmp_path, monkeypatch, outcome=synthetic_outcome):
    monkeypatch.setattr(campaign, "run_episode", fake_run_episode(campaign, outcome))
    monkeypatch.setattr(campaign, "source_hashes", lambda: {"src/frozen.py": "fixed-source"})
    monkeypatch.setattr(campaign.STEREO, "unpack_runtime", lambda archive, checksum, work: work)
    monkeypatch.setattr(campaign.HYBRID, "unpack_lio", lambda archive, checksum, work: work)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    for name in ("deploy.yaml", "policy.onnx", "orb.tar.gz", "lio.tar.gz"):
        (inputs / name).write_bytes(name.encode())
    protocol = inputs / "scientific-protocol.json"
    campaign.NAV.write_json(protocol, order_protocol(campaign))
    counter = iter(range(10**6))
    def run(output, *extra):
        monkeypatch.setenv("H34_WORK_DIR", str(tmp_path / f"work-{next(counter)}"))
        return campaign.main(["--upstream", str(tmp_path), "--deploy", str(inputs / "deploy.yaml"),
            "--checkpoint", str(inputs / "policy.onnx"), "--orb-runtime-archive", str(inputs / "orb.tar.gz"),
            "--lio-runtime-archive", str(inputs / "lio.tar.gz"), "--orb-runtime-sha256", "orb", "--lio-runtime-sha256", "lio",
            "--protocol", str(protocol), "--output", str(output), *extra])
    return run


@pytest.fixture
def synthetic_campaign(campaign, tmp_path, monkeypatch):
    run = shard_runner(campaign, tmp_path, monkeypatch)
    assert run(tmp_path / "smoke", "--phase", "smoke") == 0
    smoke = tmp_path / "smoke/campaign_result.json"
    for shard in order_protocol(campaign)["development"]["shards"]:
        assert run(tmp_path / "collector-inputs" / shard["name"], "--phase", "development", "--seed", str(shard["seed"]),
                   "--condition", shard["condition"], "--arm", shard["arm"], "--qualified-smoke", str(smoke)) == 0
    return tmp_path, smoke


def test_shard_runs_warmups_first_retains_compact_receipts_and_never_scores_them(campaign, synthetic_campaign):
    root, smoke_path = synthetic_campaign
    smoke = json.loads(smoke_path.read_text())
    protocol = order_protocol(campaign)
    assert smoke["status"] == "PASS" and smoke["design"] == campaign.ORDER_DESIGN and smoke["episodes"] == 6
    assert smoke["warmup_cells"] == protocol["smoke"]["warmup_cells"] and smoke["warmup_episodes"] == 2
    assert smoke["arm_development_ready"] == {"legacy_cumulative/lidar_inertial": True, "bounded_latest/lidar_inertial": True}
    shard = protocol["development"]["shards"][0]
    result = json.loads((root / "collector-inputs" / shard["name"] / "campaign_result.json").read_text())
    assert result["status"] == "PASS" and result["episodes"] == 4 and result["expected_declared_cells"] == 4
    assert result["completed_cells"] == [c for c in protocol["development"]["cells"] if c["shard"] == shard["name"]]
    assert result["warmup_cells"] == shard["warmup_cells"] and result["total_declared_development_cells"] == 96
    assert not any(c["role"] == "warmup" or c["seed"] == campaign.WARMUP_SEED for c in result["completed_cells"])
    for relative in result["warmup_episode_files"]:
        directory = (root / "collector-inputs" / shard["name"] / relative).parent
        assert sorted(str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()) == sorted(campaign.WARMUP_RETAINED)
        assert campaign.sha256(directory / "episode.json") == result["warmup_episode_sha256"][relative]
    executions = [json.loads((root / "collector-inputs" / shard["name"] / f).read_text())["execution"]
                  for f in result["warmup_episode_files"] + result["episode_files"]]
    assert [e["process_episode_index"] for e in executions] == list(range(6))
    assert [e["role"] for e in executions] == ["warmup"] * 2 + ["scored"] * 4
    D = dispatch_module()
    D.check_stage_result(5, dict(kind="run"), result, D.ORDER_DESIGN)
    D.smoke_gate(smoke, D.ORDER_DESIGN)


def test_counterbalanced_collection_verifies_order_and_reports_warmups_separately(campaign, synthetic_campaign):
    root, smoke = synthetic_campaign
    collection = campaign.collect_campaign(root / "collector-inputs", smoke, root / "collection.json")
    assert collection["design"] == campaign.ORDER_DESIGN and collection["declared_cells"] == collection["measured_cells"] == 96
    assert collection["paired"]["complete_pairs"] == 48 and collection["paired"]["clean_goal_count_difference"] == 46
    assert collection["order_audit"]["within_arm_counterbalanced"] is True
    assert collection["order_audit"]["by_arm"]["lidar_inertial"]["first_pipeline_counts"] == {p: 24 for p in campaign.PIPELINES}
    assert collection["warmups"]["episodes"] == 48 and collection["warmups"]["scored"] is False
    assert collection["by_pipeline_arm"]["bounded_latest/lidar_inertial"]["episodes"] == 48
    assert all(p["seed"] != campaign.WARMUP_SEED for p in collection["paired"]["pairs"])
    declared = campaign.declared_pair_orders(order_protocol(campaign))
    assert all(tuple(p["execution_order"]) == declared[(p["case"], p["seed"], p["condition"], p["method"])] for p in collection["paired"]["pairs"])
    D = dispatch_module()
    D.check_stage_result(26, dict(kind="evaluate"), collection, D.ORDER_DESIGN)
    # End to end: the frozen analysis consumes the strict collector output.
    spec = importlib.util.spec_from_file_location("schedule_order_analysis_e2e", REPO / "scripts/bench/processing_schedule_order_analysis.py")
    analysis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analysis)
    result = analysis.analyse(json.loads((root / "collection.json").read_text()))
    assert result["verdict"]["status"] == "CONFIRMED" and result["verdict"]["checks"]["gain_in_legacy_first_pairs"]
    lidar = result["arms"]["lidar_inertial"]
    assert lidar["goals"]["mcnemar"]["bounded_only"] == 46 and lidar["order"]["sequence_pairs"] == {"legacy_first": 24, "bounded_first": 24}
    assert lidar["continuous"]["navigation_pose_age_p95_ms"]["difference"]["mean"] == pytest.approx(-1380.)
    assert result["warmups"]["episodes"] == 48 and result["warmups"]["scored"] is False


def rewrite_episode(shard_dir, relative, change, *, field="episode_sha256"):
    path = shard_dir / relative
    row = json.loads(path.read_text())
    change(row)
    path.write_text(json.dumps(row, indent=2) + "\n")
    result_path = shard_dir / "campaign_result.json"
    result = json.loads(result_path.read_text())
    result[field][relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    result_path.write_text(json.dumps(result, indent=2) + "\n")


@pytest.mark.parametrize("tamper,message", [("swapped_execution", "scored execution record"), ("warmup_scored", "warm-up execution record"),
    ("missing_warmup", "warm-ups did not run"), ("warmup_hash", "differs from result hash"), ("reordered_cells", "declared execution order")])
def test_counterbalanced_collection_rejects_order_or_warmup_violations(campaign, synthetic_campaign, tamper, message):
    root, smoke = synthetic_campaign
    shard = order_protocol(campaign)["development"]["shards"][7]
    shard_dir = root / "collector-inputs" / shard["name"]
    result = json.loads((shard_dir / "campaign_result.json").read_text())
    if tamper == "swapped_execution":
        rewrite_episode(shard_dir, result["episode_files"][0], lambda row: row["execution"].update(process_episode_index=3))
    elif tamper == "warmup_scored":
        rewrite_episode(shard_dir, result["warmup_episode_files"][0], lambda row: row["execution"].update(role="scored"),
                        field="warmup_episode_sha256")
    elif tamper == "missing_warmup":
        result["warmup_cells"].pop()
        (shard_dir / "campaign_result.json").write_text(json.dumps(result))
    elif tamper == "warmup_hash":
        with (shard_dir / result["warmup_episode_files"][1]).open("a") as stream:
            stream.write(" ")
    else:
        result["completed_cells"][:2] = result["completed_cells"][1::-1]
        result["episode_files"][:2] = result["episode_files"][1::-1]
        (shard_dir / "campaign_result.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match=message):
        campaign.collect_campaign(root / "collector-inputs", smoke, root / "collection.json")
