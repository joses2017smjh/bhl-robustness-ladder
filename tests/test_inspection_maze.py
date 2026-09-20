import numpy as np

from bhl_robust.eval.inspection_maze import InspectionMaze, wall_segments


def test_ordered_stations_required_before_exit():
    task = InspectionMaze()
    for i in range(20):
        task.update([4.8, 1.6], True, .04, i*.04)
    assert task.stage == 0
    assert not task.station_times
    assert task.completed_at is None


def test_dwell_is_continuous_and_duplicate_call_does_not_advance():
    task = InspectionMaze()
    task.update([1, 0], True, .4, 0)
    task.update([1, 0], True, .4, 0)
    assert task.held_s == .4
    task.update([.2, 0], True, .04, .04)
    assert task.held_s == 0
    task.update([1, 0], True, .6, .64)
    assert task.stage == 1


def test_both_turns_and_inspections_then_exit():
    task = InspectionMaze()
    for i, (x, y, _, _) in enumerate(task.waypoints):
        task.update([x, y], True, .6, (i+1)*.6)
    assert task.stage == 5
    assert [row["station"] for row in task.station_times] == ["inspect_A", "inspect_B"]
    assert task.completed_at is not None


def test_dead_end_and_fall_are_failures():
    task = InspectionMaze()
    task.update([1.6, -1.3], True, .04, 0)
    assert task.failure == "dead_end_entered"
    task = InspectionMaze()
    task.update([1, 0], False, .6, 0)
    assert task.failure == "fall"
    assert not task.station_times


def test_route_segments_have_body_clearance_from_walls():
    task = InspectionMaze()
    points = [np.zeros(2)] + [np.array(w[:2]) for w in task.waypoints]
    for start, end in zip(points[:-1], points[1:]):
        for point in np.linspace(start, end, 30):
            for center, half in wall_segments():
                gap = np.maximum(np.abs(point-center)-half, 0)
                assert np.linalg.norm(gap) >= .7
