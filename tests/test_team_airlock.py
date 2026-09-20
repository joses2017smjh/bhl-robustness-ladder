"""Mission gates must require simultaneous roles and the actual shared route."""

import numpy as np

from bhl_robust.eval.team_airlock import TeamAirlock, velocity_command


def test_missing_member_cannot_release_door():
    task = TeamAirlock(3)
    positions = task.stations.copy()
    positions[-1] = task.starts[-1]
    for i in range(100):
        task.update(positions, np.ones(3, bool), .04, i * .04)
    assert not task.door_open
    assert task.visited.sum() == 2


def test_dwell_is_continuous_and_all_members_upright():
    task = TeamAirlock(2)
    for i in range(14):
        task.update(task.stations, [True, True], .04, i * .04)
    task.update(task.stations, [True, False], .04, .56)
    assert task.joint_dwell == 0
    for i in range(15):
        task.update(task.stations, [True, True], .04, .6 + i * .04)
    assert task.door_open
    assert task.completed_at is None


def test_exit_without_crossing_does_not_count():
    task = TeamAirlock(2)
    task.door_open = True
    for i in range(100):
        task.update(task.exits, [True, True], .04, i * .04)
    assert task.completed_at is None


def test_member_release_requires_clearance_and_complete_mission():
    task = TeamAirlock(3)
    pos = task.stations.copy()
    task.update(pos, [True] * 3, .6, .6)
    assert task.door_open
    assert np.allclose(task.targets()[1], task.stations[1])
    time = .6
    for i in range(3):
        for waypoint in task.crossing:
            pos[i] = waypoint
            time += .04
            task.update(pos, [True] * 3, .04, time)
        assert task.active == i
        pos[i] = task.exits[i]
        time += .04
        task.update(pos, [True] * 3, .04, time)
        assert task.active == i + 1
    task.update(pos, [True] * 3, .6, time + .6)
    assert task.completed_at is not None


def test_velocity_command_frame_and_speed_limit():
    cmd = velocity_command([0, 0], np.pi / 2, [4, 0], .28)
    assert np.allclose(cmd[:2], [0, -.28])
    assert abs(cmd[2]) <= .35
    assert np.allclose(velocity_command([0, 0], 0, [0, 0]), 0)
