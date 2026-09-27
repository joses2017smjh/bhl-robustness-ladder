import math
import numpy as np

from bhl_robust.eval import random_maze as rm


def test_generate_is_connected_and_seeded():
    a = rm.generate(5, 5, seed=3)
    b = rm.generate(5, 5, seed=3)
    assert a.open_edges == b.open_edges
    assert rm.generate(5, 5, seed=4).open_edges != a.open_edges
    sol = a.solution()
    assert sol and sol[0] == a.start and sol[-1] == a.goal
    # a perfect maze has n*m-1 passages; extra openings add loops
    assert len(a.open_edges) == 5 * 5 - 1 + a.extra_openings
    assert rm.generate(5, 5, seed=3, extra_openings=0).solution()


def test_walls_dedupe_and_xml_names():
    mz = rm.generate(4, 4, seed=0)
    segs = mz.wall_segments()
    assert len(segs) == len(set(segs))
    xml = rm.world_xml(mz)
    assert xml.count("wall_maze_") == len(segs) and "goal_marker" in xml and "start_marker" in xml


def test_occupancy_marks_free_then_wall():
    g = rm.OccupancyGrid((-1, 3, -1, 3), res=0.1, margin=0.5)
    angles = np.array([0.0])
    for _ in range(5):
        g.update(0.0, 0.0, 0.0, angles, np.array([1.0]), 12.0)
    p = g.prob()
    i_hit, j_hit = g.to_cell(1.0, 0.0)
    i_mid, j_mid = g.to_cell(0.5, 0.0)
    assert p[i_hit, j_hit] > 0.65 and p[i_mid, j_mid] < 0.35
    # a miss at max range marks nothing occupied
    g2 = rm.OccupancyGrid((-1, 3, -1, 3), res=0.1, margin=0.5)
    g2.update(0.0, 0.0, 0.0, angles, np.array([12.0]), 12.0)
    assert not g2.occupied().any()


def test_planner_routes_around_a_known_wall_and_through_unknown():
    g = rm.OccupancyGrid((0, 4, 0, 4), res=0.1, margin=0.2)
    planner = rm.Planner(g, inflate_m=0.2)
    path = planner.plan((0.5, 0.5), (3.5, 0.5))
    assert path and path[-1] == (3.5, 0.5)
    # standing in the goal cell still yields the exact goal as the waypoint
    near = planner.plan((3.47, 0.52), (3.5, 0.5))
    assert near and near[-1] == (3.5, 0.5)
    # a wall across x = 2 from y = -0.2 .. 2.0 forces a detour around its top end
    for y in np.arange(-0.2, 2.0, 0.1):
        i, j = g.to_cell(2.0, y)
        g.l[i, j] = g.L_MAX
    path2 = planner.plan((0.5, 0.5), (3.5, 0.5))
    assert path2 and max(p[1] for p in path2) > 2.0


def test_controller_turns_then_walks():
    c = rm.TurnWalkController()
    cmd, state, err = c.command((0, 0), 0.0, (0.0, 2.0), False)     # target at +90 deg
    assert state == "turn" and cmd[0] == 0 and cmd[2] > 0 and cmd[1] == 0
    cmd, state, _ = c.command((0, 0), math.pi / 2 - 0.05, (0.0, 2.0), False)
    assert state == "walk" and cmd[0] > 0.2 and cmd[1] == 0
    cmd, state, _ = c.command((0, 1.9), math.pi / 2, (0.0, 2.0), False)
    assert state == "arrived" and not cmd.any()
