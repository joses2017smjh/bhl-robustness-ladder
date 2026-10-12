"""Causal queue accounting and one-charge timing for the new schedule treatment."""
import copy

import numpy as np
import pytest

from bhl_robust.research.navigation_sensor_controls import AvailableImu
from bhl_robust.research.processing_schedule import (
    LatestJoinedQueue, compare_replay_inputs, decoded_frame_identity, service_timing,
)


def offer(queue, camera, lidar, *, event=None, payload=None):
    queue.offer(payload, event_s=max(camera, lidar) if event is None else event,
        camera_id=f"camera-{camera}", camera_capture_s=camera,
        lidar_id=f"lidar-{lidar}", lidar_capture_s=lidar)


def test_latest_joined_slot_preserves_original_sensor_times_and_payload():
    queue = LatestJoinedQueue()
    offer(queue, 1., 1.)
    newest = {"imu_not_owned_by_slot": "external", "pixels": np.ones((2, 2, 3), np.uint8)}
    offer(queue, 1.1, 1., payload=newest)
    selected = queue.pop(1.2)
    assert selected["payload"] is newest
    assert selected["queue"]["camera_capture_s"] == 1.1
    assert selected["queue"]["lidar_capture_s"] == 1.
    assert selected["queue"]["event_s"] == 1.1
    assert selected["queue"]["queue_residence_s"] == pytest.approx(.1)
    assert queue.pop(1.2) is None
    receipt = queue.receipt()
    assert (receipt["produced"], receipt["selected"], receipt["superseded"], receipt["pending"]) == (2, 1, 1, 0)
    assert receipt["streams"]["camera"]["superseded_before_selection"] == 1
    assert receipt["streams"]["lidar"]["superseded_before_selection"] == 0
    assert receipt["conservation_pass"]
    assert all(value["conservation_pass"] for value in receipt["streams"].values())


def test_repeated_stream_frames_do_not_become_false_drops_or_unique_updates():
    queue = LatestJoinedQueue()
    offer(queue, 1., 1.)
    queue.pop(1.)
    offer(queue, 1.1, 1.)
    offer(queue, 1.2, 1.2)
    queue.pop(1.2)
    offer(queue, 1.3, 1.2)
    queue.pop(1.3)
    receipt = queue.receipt()
    camera, lidar = receipt["streams"]["camera"], receipt["streams"]["lidar"]
    assert (camera["produced_unique"], camera["selected_unique"], camera["superseded_before_selection"]) == (4, 3, 1)
    assert (lidar["produced_unique"], lidar["selected_unique"], lidar["superseded_before_selection"]) == (2, 2, 0)
    assert lidar["dispatches"] == 3 and lidar["reused_dispatches"] == 1
    assert camera["reused_dispatches"] == 0


def test_pending_horizon_accounting_keeps_unselected_frames_visible():
    queue = LatestJoinedQueue()
    offer(queue, 1., 1.)
    queue.pop(1.)
    offer(queue, 1.1, 1.)
    receipt = queue.receipt()
    assert receipt["pending"] == 1 and receipt["selected"] == 1
    assert receipt["streams"]["camera"]["pending_unselected"] == 1
    assert receipt["streams"]["lidar"]["pending_unselected"] == 0
    assert all(value["conservation_pass"] for value in receipt["streams"].values())


def test_selection_waits_for_causal_event_availability():
    queue = LatestJoinedQueue()
    offer(queue, 1., .9)
    assert queue.pop(.99) is None
    assert queue.receipt()["pending"] == 1
    assert queue.pop(1.) is not None
    with pytest.raises(ValueError, match="selection clock"):
        queue.pop(.999)


@pytest.mark.parametrize("changes", [
    {"event_s": 1.}, {"event_s": float("nan")}, {"camera_capture_s": 1.3},
    {"camera_id": "camera-1.0", "camera_capture_s": 1.1},
    {"camera_id": "new-id", "camera_capture_s": .9},
    {"lidar_id": True}, {"lidar_id": float("inf")}, {"lidar_id": ""},
])
def test_bad_offer_is_rejected_without_corrupting_queue_or_accounting(changes):
    queue = LatestJoinedQueue()
    offer(queue, 1., 1.)
    before = copy.deepcopy(queue.receipt())
    args = dict(event_s=1.2, camera_id="camera-1.2", camera_capture_s=1.2,
        lidar_id="lidar-1.0", lidar_capture_s=1.)
    args.update(changes)
    with pytest.raises(ValueError):
        queue.offer("invalid", **args)
    assert queue.receipt() == before
    assert queue.pop(1.2)["queue"]["camera_capture_s"] == 1.


def test_sensor_identity_cannot_return_after_supersession():
    queue = LatestJoinedQueue()
    offer(queue, 1., 1.)
    offer(queue, 1.1, 1.)
    with pytest.raises(ValueError, match="regressing"):
        offer(queue, 1., 1.2)


def test_skipping_lidar_frames_does_not_remove_intervening_imu():
    queue, imu = LatestJoinedQueue(), AvailableImu()
    for stamp in np.arange(1., 1.305, .005):
        imu.append(float(stamp), [0, 0, 0], [0, 0, 9.81])
    offer(queue, 1.1, 1.1)
    offer(queue, 1.2, 1.2)
    selected = queue.pop(1.25)
    batch = imu.consume(selected["queue"]["lidar_capture_s"])
    assert len(batch["timestamp_s"]) == 41
    assert np.max(np.diff(batch["timestamp_s"])) == pytest.approx(.005)
    assert batch["timestamp_s"][-1] == pytest.approx(1.2)
    assert imu.pending and imu.pending[0][0] > 1.2
    assert queue.receipt()["streams"]["lidar"]["superseded_before_selection"] == 1


def test_timing_charges_preprocessing_and_native_once_without_recharging_wait():
    result = service_timing(request_s=2., capture_s=1.7,
        preprocessing_s=.02, native_client_s=.041, policy_dt=.04)
    assert result["service_s"] == pytest.approx(.061)
    assert result["ready_s"] == result["estimator_start_s"] == pytest.approx(2.02)
    assert result["estimator_end_s"] == pytest.approx(2.061)
    assert result["delivery_ticks"] == 2
    assert result["delivery_s"] == pytest.approx(2.08)
    assert result["capture_age_at_request_s"] == pytest.approx(.3)
    assert result["capture_age_at_delivery_s"] == pytest.approx(.38)
    assert result["delivery_quantization_s"] == pytest.approx(.019)
    # Service below350ms cannot make a pose fresh when capture/queue age is high.
    assert result["service_s"] < .35 < result["capture_age_at_delivery_s"]


def test_zero_work_still_respects_existing_one_step_delivery():
    result = service_timing(request_s=2., capture_s=2.,
        preprocessing_s=0., native_client_s=0., policy_dt=.04)
    assert result["delivery_ticks"] == 1
    assert result["delivery_s"] == pytest.approx(2.04)


@pytest.mark.parametrize("changes", [
    {"request_s": float("nan")}, {"capture_s": 3.}, {"preprocessing_s": -.001},
    {"native_client_s": float("inf")}, {"policy_dt": 0.}, {"native_client_s": True},
    {"native_client_s": 1e308, "policy_dt": 1e-308},
    {"request_s": 1e308, "native_client_s": 1e308, "policy_dt": 1.},
])
def test_timing_rejects_noncausal_or_invalid_durations(changes):
    args = dict(request_s=2., capture_s=1.9, preprocessing_s=.01, native_client_s=.02, policy_dt=.04)
    args.update(changes)
    with pytest.raises(ValueError):
        service_timing(**args)


def profile_rows():
    pixels = np.zeros((2, 4, 3), np.uint8)
    return [{"timestamp_s": 1., "decoded_identity": decoded_frame_identity(pixels, pixels)},
        {"timestamp_s": 1.2, "decoded_identity": decoded_frame_identity(pixels + 1, pixels + 2)}]


def test_decoded_identity_is_independent_of_array_storage_layout():
    pixels = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
    assert decoded_frame_identity(pixels, pixels) == decoded_frame_identity(np.asfortranarray(pixels), pixels.copy())
    reshaped = pixels.reshape(4, 2, 3)
    assert decoded_frame_identity(pixels, pixels) != decoded_frame_identity(reshaped, reshaped)


def test_paired_profile_verifies_full_offered_sequence_and_returns_reproducible_digest():
    first = profile_rows()
    second = copy.deepcopy(first)
    second[0]["selected_by_this_pipeline"] = False
    receipt = compare_replay_inputs(first, second)
    assert receipt["status"] == "PASS" and receipt["frames"] == 2
    assert len(receipt["ordered_inputs_sha256"]) == 64
    assert receipt == compare_replay_inputs(second, first)


@pytest.mark.parametrize("change", ["timestamp", "pixels", "missing", "reorder", "duplicate", "malformed"])
def test_paired_profile_rejects_data_drift_or_incomplete_order(change):
    first, second = profile_rows(), profile_rows()
    if change == "timestamp":
        second[1]["timestamp_s"] = 1.21
    elif change == "pixels":
        second[1]["decoded_identity"]["left"]["sha256"] = "f" * 64
    elif change == "missing":
        second.pop()
    elif change == "reorder":
        second.reverse()
    elif change == "duplicate":
        second[1]["timestamp_s"] = 1.
    else:
        first[0]["decoded_identity"]["left"]["shape"] = [0, 4]
        second = copy.deepcopy(first)
    with pytest.raises(ValueError):
        compare_replay_inputs(first, second)


def test_empty_profiles_cannot_qualify():
    with pytest.raises(ValueError, match="nonempty"):
        compare_replay_inputs([], [])


@pytest.mark.parametrize("pixels", [np.zeros((2, 3), float), np.zeros((0, 3), np.uint8), np.zeros((2, 3, 2), np.uint8)])
def test_decoded_identity_rejects_unmeasured_or_invalid_pixel_shapes(pixels):
    with pytest.raises(ValueError, match="decoded pixels"):
        decoded_frame_identity(pixels, pixels)
