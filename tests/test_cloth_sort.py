"""Smoke tests for the cloth-sort redesign. No Isaac Sim, no GPU."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from bhl_robust.cloth.controller import assert_limits_cover_pinch, is_plan_valid, sample_trajectory
from bhl_robust.cloth.cost import guard_or_raise, report_cost
from bhl_robust.cloth.env import KinematicClothSortEnv, make_env
from bhl_robust.cloth.garments import BASKET_IDS, GARMENTS, GARMENT_BY_NAME, SEMANTIC_CLASSES, one_garment
from bhl_robust.cloth.kinematics import relative_up_z
from bhl_robust.cloth.layout import assert_layout, basket_aabb, basket_center, garment_spawn_range, table_aabb
from bhl_robust.cloth.mesh import grid_counts, isaac_grid_counts, plane_faces, plane_vertices
from bhl_robust.cloth.observations import ObservationSpec, pack_oracle
from bhl_robust.cloth.rewards import compute_reward
from bhl_robust.cloth.robot import EXPECTED_ARMS, EXPECTED_JOINTS, assert_articulation
from bhl_robust.cloth.scripted import pick_unsorted, scripted_action, scripted_for
from bhl_robust.cloth.success import deformable_outcome, rigid_outcome
from bhl_robust.cloth.sweep import ACTION_DIM, decode_action, encode_physical, plan_sweep
from bhl_robust.limb_partition import JOINTS_22


class ArticulationTests(unittest.TestCase):
    def test_stock_humanoid(self):
        assert_articulation()
        self.assertEqual(EXPECTED_JOINTS, 22)
        self.assertEqual(EXPECTED_ARMS, 2)
        self.assertEqual(len(JOINTS_22), 22)

    def test_rejects_extra_arm(self):
        with self.assertRaises(AssertionError):
            assert_articulation(n_arms=3)

    def test_rejects_gripper_joints(self):
        names = list(JOINTS_22) + ["arm_left_gripper_joint"]
        with self.assertRaises(AssertionError):
            assert_articulation(names)

    def test_pinch_inside_walls(self):
        assert_limits_cover_pinch()


class CatalogTests(unittest.TestCase):
    def test_five_garments_three_baskets(self):
        self.assertEqual(len(GARMENTS), 5)
        self.assertEqual(len(BASKET_IDS), 3)
        self.assertEqual(len(SEMANTIC_CLASSES), 3)
        classes = {g.semantic_class for g in GARMENTS}
        self.assertEqual(classes, set(SEMANTIC_CLASSES))

    def test_class_maps_to_basket(self):
        for g in GARMENTS:
            self.assertIn(g.target_basket, BASKET_IDS)


class LayoutTests(unittest.TestCase):
    def test_layout_invariants(self):
        assert_layout()
        table = table_aabb()
        self.assertGreater(table.high[2], table.low[2])
        for bid in BASKET_IDS:
            b = basket_aabb(bid)
            self.assertLess(b.high[2], table.high[2])


class SuccessTests(unittest.TestCase):
    def test_rigid_com_in_correct_basket(self):
        spec = one_garment()[0]
        com = basket_center(spec.target_basket)
        out = rigid_outcome(com, spec)
        self.assertTrue(out.correct)
        self.assertFalse(out.wrong)

    def test_wrong_basket_is_not_success(self):
        spec = one_garment()[0]
        other = next(b for b in BASKET_IDS if b != spec.target_basket)
        out = rigid_outcome(basket_center(other), spec)
        self.assertFalse(out.correct)
        self.assertTrue(out.wrong)

    def test_deformable_vertex_fraction(self):
        spec = one_garment()[0]
        c = basket_center(spec.target_basket)
        inside = np.repeat(c[None, :], 8, axis=0)
        outside = inside + np.array([2.0, 0.0, 0.0])
        verts = np.vstack([inside, outside])  # 50%
        miss = deformable_outcome(verts, spec, threshold=0.60)
        self.assertFalse(miss.correct)
        hit = deformable_outcome(inside, spec, threshold=0.60)
        self.assertTrue(hit.correct)


class SweepTests(unittest.TestCase):
    def test_encode_decode_roundtrip(self):
        raw = encode_physical(0.05, -0.04, 0.3, 0.4, 0.3)
        self.assertEqual(raw.size, ACTION_DIM)
        d = decode_action(raw)
        self.assertAlmostEqual(d["dx_start"], 0.05, places=5)
        self.assertAlmostEqual(d["sweep_distance"], 0.4, places=5)

    def test_plan_at_the_original_garment_spawn_is_refused(self):
        """The original layout put the garment 0.74 m from a 0.29 m reach."""
        params = decode_action(encode_physical(-0.10, 0.0, np.pi, 0.35, 0.4))
        plan = plan_sweep(np.array([0.5, 0.0]), params)
        self.assertFalse(is_plan_valid(plan))
        # The joint walls alone would have passed it; reach is what refuses it.
        self.assertTrue(is_plan_valid(plan, check_reach=False))

    def test_plan_inside_the_measured_workspace_is_valid(self):
        from bhl_robust.cloth.reach import robot_to_world
        # Robot frame (+x forward, -y = the robot's right): a short sweep
        # outward across the right-hand patch. With the robot facing -x in the
        # world, robot -y is world +y, so the sweep angle is +pi/2.
        start = robot_to_world((0.10, -0.26, 0.0))[:2]
        # 8 cm stays inside the patch; 12 cm ends at robot y = -0.38, just past
        # its edge, and is correctly refused.
        params = decode_action(encode_physical(0.0, 0.0, np.pi / 2, 0.08, 0.3))
        plan = plan_sweep(start, params)
        self.assertTrue(is_plan_valid(plan))
        traj = sample_trajectory(plan, dt=0.1)
        self.assertGreater(len(traj), 3)
        self.assertTrue(all(wp.valid for wp in traj))


class RewardTests(unittest.TestCase):
    def test_components_logged_separately(self):
        rb = compute_reward(
            prev_dist=0.5, curr_dist=0.2, newly_correct=True, newly_wrong=False,
            sweep_distance=0.4, min_useful_distance=0.5, fell=False, invalid=False,
        )
        d = rb.as_dict()
        self.assertIn("progress", d)
        self.assertIn("success", d)
        self.assertGreater(d["success"], 0)
        self.assertAlmostEqual(d["total"], rb.total)


class ObservationTests(unittest.TestCase):
    def test_pack_matches_spec(self):
        spec = ObservationSpec(n_garments=1)
        obs = pack_oracle(
            joint_pos=np.zeros(22), joint_vel=np.zeros(22),
            base_quat=np.array([1, 0, 0, 0]), base_ang_vel=np.zeros(3),
            hand_pos=np.zeros(6), garment_xy=np.zeros((1, 2)),
            garment_yaw=np.zeros(1), class_index=np.array([1]),
            rel_xy=np.zeros((1, 2)), selected=0, sorted_mask=np.zeros(1),
            basket_xy=np.zeros((3, 2)), progress=0.0, spec=spec,
        )
        self.assertEqual(obs.size, spec.size)


class EnvTests(unittest.TestCase):
    def test_create_reset_step_c0(self):
        env = make_env("C0", seed=0)
        obs = env.reset(seed=0)
        self.assertEqual(obs.size, env.observation_space_n)
        act = scripted_action(env.garment_xy(0), env.selected_spec())
        obs2, rew, done, info = env.step(act)
        self.assertEqual(obs2.size, obs.size)
        self.assertIsInstance(rew, float)
        self.assertTrue(hasattr(info.reward, "total"))

    def test_seeded_reset_deterministic(self):
        a = make_env("C0", seed=7)
        b = make_env("C0", seed=7)
        oa, ob = a.reset(seed=7), b.reset(seed=7)
        np.testing.assert_allclose(oa, ob)
        np.testing.assert_allclose(a.garment_xy(0), b.garment_xy(0))

    def test_scripted_sorts_one_garment_inside_the_reach(self):
        """Under the reach model, on the redesigned layout, scripted C0 sorts.

        On the original layout this same loop could not move the garment at all:
        every plan was refused. That is pinned in ReachTests with the old
        coordinates frozen.
        """
        env = make_env("C0", seed=1)
        env.reset(seed=1)
        done, refused = False, 0
        while not done:
            act = scripted_for(env)
            _, _, done, info = env.step(act)
            refused += int(info.invalid)
        self.assertTrue(env.all_correct)
        self.assertEqual(refused, 0)

    def test_wrong_sweep_can_miss(self):
        env = make_env("C0", seed=2)
        env.reset(seed=2)
        # Sweep the opposite way.
        act = encode_physical(0.1, 0.0, 0.0, 0.2, 0.3)  # +x, away from baskets
        _, _, _, info = env.step(act)
        self.assertFalse(info.all_correct)

    def test_five_garment_reset(self):
        env = make_env("C5", seed=0)
        env.reset(seed=0)
        self.assertEqual(env.n_garments, 5)
        snap = env.snapshot()
        self.assertEqual(len(snap), 5)

    def test_scripted_sorts_five_garments_in_turn(self):
        """Five garments, one table's worth at a time: Mode B, sequential."""
        env = make_env("C5", seed=3)
        env.reset(seed=3)
        on_table = [not st.parked for st in env._states]
        self.assertEqual(on_table, [True, False, False, False, False])
        done = False
        while not done:
            act = scripted_for(env)
            _, _, done, _ = env.step(act)
        self.assertTrue(env.all_correct)

    def test_pick_unsorted(self):
        mask = np.array([True, False, False])
        pos = np.array([[0.0, 0.0], [1.0, 0.0], [0.1, 0.0]])
        self.assertEqual(pick_unsorted(mask, pos), 2)


class CostTests(unittest.TestCase):
    def test_standard_deformable_arm_is_refused(self):
        rep = report_cost(num_envs=2048, physics="deformable")
        self.assertFalse(rep.accepted)
        self.assertGreater(rep.estimated_days, 1.0)
        with self.assertRaises(RuntimeError):
            guard_or_raise(num_envs=2048, physics="deformable")

    def test_tiny_deformable_probe_passes(self):
        # 20 iterations × 8 envs × 24 steps at 182 sps is minutes, not days.
        rep = report_cost(
            num_envs=8, iterations=20, steps_per_iter=24, physics="deformable",
        )
        self.assertTrue(rep.accepted)

    def test_gc1_arithmetic(self):
        # 8000 * 2048 * 48 = 786,432,000
        rep = report_cost(num_envs=2048, iterations=8000, steps_per_iter=48)
        self.assertEqual(rep.env_steps, 786_432_000)


class MeshTests(unittest.TestCase):
    def test_grid_counts(self):
        v, t = grid_counts(8)
        self.assertEqual(v, 64)
        self.assertEqual(t, 98)
        self.assertEqual(len(plane_vertices(8)), 64)
        self.assertEqual(len(plane_faces(8)), 98)


class SpawnTests(unittest.TestCase):
    def test_sibling_index(self):
        from bhl_robust.cloth.garments import sibling_index
        sock_a, sock_b = GARMENTS[0], GARMENTS[1]
        self.assertEqual(sibling_index(sock_a), (0, 2))
        self.assertEqual(sibling_index(sock_b), (1, 2))
        jacket = GARMENT_BY_NAME["jacket"]
        self.assertEqual(sibling_index(jacket), (0, 1))

    def test_garments_start_on_the_table_and_the_rest_park_clear(self):
        from bhl_robust.cloth.layout import all_basket_aabbs, default_spawn_xy, parking_xy
        x, y = default_spawn_xy()
        lo, hi = garment_spawn_range()
        self.assertTrue(lo[0] <= x <= hi[0] and lo[1] <= y <= hi[1])
        self.assertTrue(table_aabb().contains_xy((x, y)))
        slots = {parking_xy(i) for i in range(1, 5)}
        self.assertEqual(len(slots), 4)
        for xy in slots:
            self.assertFalse(table_aabb().contains_xy(xy))
            for box in all_basket_aabbs().values():
                self.assertFalse(box.contains_xy(xy))


class AdaptTests(unittest.TestCase):
    def test_collect_fit_eval_linear(self):
        from bhl_robust.cloth.adapt import collect_scripted, evaluate_linear, fit_linear, linear_action
        x, y = collect_scripted(episodes=4, seed=0, rung="C0")
        self.assertEqual(x.ndim, 2)
        self.assertEqual(y.shape[1], 5)
        w = fit_linear(x, y)
        self.assertEqual(w.shape, (x.shape[1] + 1, 5))
        act = linear_action(x[0], w)
        self.assertEqual(act.shape, (5,))
        ev = evaluate_linear(w, episodes=4, seed=100, rung="C1")
        self.assertEqual(len(ev.episodes), 4)
        self.assertEqual(ev.policy, "bc_linear")


class SpawnRelativeTiltTests(unittest.TestCase):
    """The fall predicate's algebra, on the login node.

    ``relative_up_z`` is the same function ``cloth_sort_mdp.tilt_from_spawn``
    calls with torch tensors, so these cases test the shipped algebra rather
    than a paraphrase of it.
    """

    #: The quaternion the task actually spawns with, as the 4-tuple in the cfg.
    #: Photographed standing (21218517 / 21218627). Isaac Lab 3.0 stores it
    #: (x, y, z, w): a half-turn about z. These wxyz cases run the algebra on
    #: the same tuple read the old way; ``QuatOrderTests`` covers the real order.
    STAND_UP = np.array([[0.0, 0.0, 1.0, 0.0]])

    def test_zero_tilt_at_the_configured_spawn(self):
        up = relative_up_z(self.STAND_UP, self.STAND_UP)
        self.assertAlmostEqual(float(up[0]), 1.0, places=6)
        self.assertLess(float(np.arccos(np.clip(up, -1, 1))[0]), 1e-6)

    def test_absolute_convention_would_have_called_this_fallen(self):
        """Guards the regression that ended every episode on step one.

        Read as (w, x, y, z) the spawn tuple is a half-turn about y, R[2, 2] = -1.
        That misreading, not the asset, is what 21233866 hit.
        """
        w, x, y, z = self.STAND_UP[0]
        absolute_r22 = 1.0 - 2.0 * (x * x + y * y)
        self.assertAlmostEqual(float(absolute_r22), -1.0, places=6)
        self.assertLess(absolute_r22, 0.70)  # i.e. "fallen" at reset
        # Relative-to-spawn does not make that mistake.
        self.assertAlmostEqual(
            float(relative_up_z(self.STAND_UP, self.STAND_UP)[0]), 1.0, places=6,
        )

    def _quat_mul(self, a, b):
        aw, ax, ay, az = a
        bw, bx, by, bz = b
        return np.array([
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ])

    def test_yaw_about_world_up_is_not_tilt(self):
        """Reset jitters yaw by +-0.08 rad. Yaw must read as zero tilt."""
        for ang in (0.08, -0.08, 0.9, np.pi):
            yaw = np.array([np.cos(ang / 2), 0.0, 0.0, np.sin(ang / 2)])
            q = self._quat_mul(yaw, self.STAND_UP[0])[None, :]
            up = float(relative_up_z(q, self.STAND_UP)[0])
            self.assertAlmostEqual(up, 1.0, places=6, msg=f"yaw {ang} read as tilt")

    def test_known_tip_angles(self):
        """A world-frame roll of theta must read back as theta."""
        for theta in (0.2, 0.78, 1.2, np.pi / 2):
            roll = np.array([np.cos(theta / 2), np.sin(theta / 2), 0.0, 0.0])
            q = self._quat_mul(roll, self.STAND_UP[0])[None, :]
            up = np.clip(relative_up_z(q, self.STAND_UP), -1.0, 1.0)
            self.assertAlmostEqual(float(np.arccos(up)[0]), theta, places=5)

    def test_fall_limit_brackets_the_termination_threshold(self):
        """0.78 rad is the limit; 0.6 stands, 1.0 falls."""
        def tilt(theta):
            roll = np.array([np.cos(theta / 2), np.sin(theta / 2), 0.0, 0.0])
            q = self._quat_mul(roll, self.STAND_UP[0])[None, :]
            return float(np.arccos(np.clip(relative_up_z(q, self.STAND_UP), -1, 1))[0])
        self.assertLess(tilt(0.60), 0.78)
        self.assertGreater(tilt(1.00), 0.78)


class QuatOrderTests(unittest.TestCase):
    """Isaac Lab 3.0 stores quaternions (x, y, z, w); the fall test must read them so."""

    @staticmethod
    def _xyzw(axis: str, angle: float, then=None) -> np.ndarray:
        from scipy.spatial.transform import Rotation as Rot
        r = Rot.from_euler(axis, angle)
        if then is not None:
            r = then * r
        return r.as_quat(scalar_first=False)[None, :]

    def setUp(self):
        from scipy.spatial.transform import Rotation as Rot
        self.spawn_rot = Rot.from_euler("z", np.pi)
        self.spawn = self.spawn_rot.as_quat(scalar_first=False)[None, :]

    def test_the_spawn_tuple_is_a_half_turn_about_z_in_xyzw(self):
        np.testing.assert_allclose(np.abs(self.spawn), [[0.0, 0.0, 1.0, 0.0]], atol=1e-9)

    def test_xyzw_reading_separates_yaw_from_tilt(self):
        for label, q, want in (
            ("yaw 30", self._xyzw("z", np.radians(30), then=self.spawn_rot), 0.0),
            ("roll 60", self._xyzw("x", np.radians(60), then=self.spawn_rot), 60.0),
            ("pitch 60", self._xyzw("y", np.radians(60), then=self.spawn_rot), 60.0),
        ):
            # A body-frame rotation after the spawn half-turn: spawn * r.
            from scipy.spatial.transform import Rotation as Rot
            axis = {"yaw 30": "z", "roll 60": "x", "pitch 60": "y"}[label]
            ang = np.radians(30 if label == "yaw 30" else 60)
            q = (self.spawn_rot * Rot.from_euler(axis, ang)).as_quat(scalar_first=False)[None, :]
            got = np.degrees(np.arccos(np.clip(relative_up_z(q, self.spawn, order="xyzw"), -1, 1)))[0]
            self.assertAlmostEqual(float(got), want, places=4, msg=label)

    def test_the_old_reading_counted_yaw_and_missed_roll(self):
        """Pins the bug: (x, y, z, w) data read as (w, x, y, z)."""
        from scipy.spatial.transform import Rotation as Rot
        yaw = (self.spawn_rot * Rot.from_euler("z", np.radians(30))).as_quat(scalar_first=False)[None, :]
        roll = (self.spawn_rot * Rot.from_euler("x", np.radians(60))).as_quat(scalar_first=False)[None, :]
        tilt = lambda q: float(np.degrees(np.arccos(np.clip(relative_up_z(q, self.spawn, order="wxyz"), -1, 1)))[0])
        self.assertAlmostEqual(tilt(yaw), 30.0, places=4)
        self.assertAlmostEqual(tilt(roll), 0.0, places=4)

    def test_heading_helper_matches_scipy(self):
        from scipy.spatial.transform import Rotation as Rot
        from bhl_robust.cloth.kinematics import yaw_atan2_args
        for ang in (0.3, -1.2, 2.9):
            r = Rot.from_euler("z", ang)
            got = np.arctan2(*yaw_atan2_args(r.as_quat(scalar_first=False)[None, :], "xyzw"))[0]
            self.assertAlmostEqual(float(got), ang, places=6)
            got = np.arctan2(*yaw_atan2_args(r.as_quat(scalar_first=True)[None, :], "wxyz"))[0]
            self.assertAlmostEqual(float(got), ang, places=6)

    def test_isaac_side_reads_quaternions_through_the_probed_order(self):
        import re
        src = (_REPO / "src" / "bhl_robust" / "tasks" / "cloth_sort_mdp.py").read_text()
        self.assertIn("QUAT_ORDER = quat_order()", src)
        self.assertIn("from bhl_robust.quat_order import quat_order", src)
        self.assertIn("relative_up_z(q, q0, order=QUAT_ORDER)", src)
        self.assertIsNone(re.search(r"w, x, y, z = q\[:, 0\]", src),
                          "a hand-rolled (w, x, y, z) unpack is back in the cloth MDP")


class MetricsTests(unittest.TestCase):
    def test_invalid_rate_is_a_rate(self):
        """Six refused sweeps out of six is a rate of 1.0, not 6.0."""
        from bhl_robust.cloth.metrics import EpisodeMetrics, RunMetrics
        run = RunMetrics(physics="kinematic", policy="scripted", num_envs=1, n_deformables=0)
        for _ in range(3):
            em = EpisodeMetrics(n_garments=1)
            em.n_sweeps, em.invalid_trajectory = 6, 6
            run.episodes.append(em)
        d = run.as_dict()
        self.assertEqual(d["invalid_trajectory_rate"], 1.0)
        self.assertEqual(d["invalid_per_episode"], 6.0)
        self.assertLessEqual(d["invalid_trajectory_rate"], 1.0)


class BenchCsvTests(unittest.TestCase):
    """The results CSV must stay parseable against its own header."""

    def test_append_unions_schemas_instead_of_misaligning(self):
        import csv
        import tempfile
        from bhl_robust.cloth.metrics import append_rows_csv
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "bench.csv"
            append_rows_csv(path, [{"robot": "kin", "rung": "C0", "sps": 350.0}])
            # A second bench with different columns, as the Isaac one has.
            append_rows_csv(path, [{"robot": "bhl", "task": "Cloth-v0",
                                    "num_envs": 8, "sps": 467.2}])
            rows = list(csv.DictReader(path.open()))
            self.assertEqual(len(rows), 2)
            for r in rows:
                self.assertNotIn(None, r, "row has more fields than the header")
            # The value must still be under its own key, not a neighbour's.
            self.assertEqual(rows[0]["rung"], "C0")
            self.assertEqual(rows[0]["task"], "")
            self.assertEqual(rows[1]["task"], "Cloth-v0")
            self.assertEqual(float(rows[1]["sps"]), 467.2)

    def test_existing_rows_are_preserved(self):
        import csv
        import tempfile
        from bhl_robust.cloth.metrics import append_rows_csv
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "bench.csv"
            append_rows_csv(path, [{"a": 1}, {"a": 2}])
            append_rows_csv(path, [{"b": 3}])
            rows = list(csv.DictReader(path.open()))
            self.assertEqual([r["a"] for r in rows], ["1", "2", ""])

    def test_committed_bench_csv_is_aligned(self):
        """Guards the file itself: job 21234167 wrote rows under a foreign header."""
        import csv
        path = _REPO / "results" / "cloth_sort_bench.csv"
        if not path.exists():
            self.skipTest("no bench csv yet")
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames or []
            for i, row in enumerate(reader, start=2):
                self.assertNotIn(
                    None, row,
                    f"{path.name} line {i} has more fields than its {len(header)}-column header",
                )


class ReachTests(unittest.TestCase):
    """The measured workspace, and what it says about the original layout."""

    def setUp(self):
        from bhl_robust.cloth.reach import load_reach
        self.t = load_reach()

    def test_table_is_the_measured_one(self):
        self.assertGreater(int(self.t.mask.sum()), 1000)
        self.assertAlmostEqual(self.t.planted_root_z, -0.137, delta=0.01)
        self.assertEqual(len(self.t.joints), 5)
        self.assertTrue(all(j.startswith("arm_right_") for j in self.t.joints))

    def test_right_hand_workspace_is_on_the_right(self):
        """Guards the frame convention: the right hand never crosses the midline."""
        ix, iy, iz = np.nonzero(self.t.mask)
        self.assertTrue(bool(np.all(self.t.y[iy] < 0.0)))
        self.assertLess(float(self.t.x[ix].max()), 0.31)

    def test_robot_faces_minus_x_as_measured(self):
        from bhl_robust.cloth.layout import ROBOT_XY, ROBOT_YAW
        from bhl_robust.cloth.reach import robot_to_world, to_robot_frame
        self.assertAlmostEqual(ROBOT_YAW, np.pi, places=6)
        ahead = np.array([ROBOT_XY[0] - 0.2, ROBOT_XY[1], 0.4])
        np.testing.assert_allclose(to_robot_frame(ahead), [0.2, 0.0, 0.4], atol=1e-9)
        p = np.array([0.07, -0.21, 0.43])
        np.testing.assert_allclose(to_robot_frame(robot_to_world(p)), p, atol=1e-9)

    def test_nothing_in_the_original_layout_is_reachable(self):
        """The finding, pinned with the old coordinates frozen here.

        The original scene put the garment at (0.52, 0), the table's near edge at
        x = 0.17 and the basket centres at (0.10, +-0.34 / 0) -- world frame, robot
        at (-0.22, 0) facing -x as measured. No fingertip contact cell covers any
        of it; the redesign moved the scene, not the finding.
        """
        from bhl_robust.cloth.reach import sweep_cell_ok
        old = {
            "garment spawn": (0.52, 0.0),
            "table centre": (0.52, 0.0),
            "table near edge": (0.17, 0.0),
            "basket socks": (0.10, 0.34),
            "basket shirts": (0.10, 0.00),
            "basket jackets": (0.10, -0.34),
        }
        for label, xy in old.items():
            self.assertFalse(sweep_cell_ok(xy, "contact"),
                             f"{label} at {xy} is reachable; the old-layout finding no longer holds")


class HandColliderTests(unittest.TestCase):
    """The shipped hands collide with nothing; the cloth tasks must use the overlay."""

    ASSET = _REPO / "assets" / "cloth" / "berkeley_humanoid_lite_hand_colliders.usda"

    def test_overlay_exists_and_adds_both_hand_colliders(self):
        self.assertTrue(self.ASSET.exists(), "run scripts/cloth/add_hand_colliders.py")
        text = self.ASSET.read_text()
        self.assertIn("berkeley_humanoid_lite.usd", text, "overlay must sublayer the upstream USD")
        for side in ("left", "right"):
            self.assertIn(f'over "arm_{side}_hand_link"', text)
        self.assertEqual(text.count('def Mesh "hand_collider"'), 2)
        self.assertEqual(text.count('"PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"'), 2)
        self.assertEqual(text.count('physics:approximation = "convexHull"'), 2)
        self.assertNotIn('def Cube "hand_collider"', text, "the bounding box is back")

    def test_collider_is_under_the_physx_hull_limit_and_keeps_the_fingertip(self):
        import re
        from bhl_robust.cloth.arm_fk import load_chain
        text = self.ASSET.read_text()
        blocks = re.findall(r"point3f\[\] points = \[(.*?)\]\n", text)
        self.assertEqual(len(blocks), 2)
        for b in blocks:
            pts = np.array([[float(v) for v in t.split(",")] for t in re.findall(r"\(([^)]*)\)", b)])
            self.assertLessEqual(len(pts), 64)
            self.assertAlmostEqual(float(pts[:, 2].min()), float(load_chain().hull[:, 2].min()), places=4)

    def test_upstream_urdf_really_has_no_hand_collision(self):
        """Pins the reason the overlay exists; if upstream fixes it, revisit."""
        import re
        urdf = (_REPO / "external/Berkeley-Humanoid-Lite/source/berkeley_humanoid_lite_assets/data/robots/"
                "berkeley_humanoid/berkeley_humanoid_lite/urdf/berkeley_humanoid_lite.urdf").read_text()
        for side in ("left", "right"):
            m = re.search(rf'<link name="arm_{side}_hand_link">(.*?)</link>', urdf, re.S)
            self.assertIsNotNone(m)
            self.assertNotIn("<collision", m.group(1))

    def test_cloth_scene_spawns_the_overlay(self):
        src = (_REPO / "src" / "bhl_robust" / "tasks" / "cloth_sort_env_cfg.py").read_text()
        self.assertIn("berkeley_humanoid_lite_hand_colliders.usda", src)
        self.assertIn("usd_path=str(HAND_COLLIDER_USD)", src)


class ScheduleTests(unittest.TestCase):
    """The Isaac action term's schedule, checked where Kit cannot start."""

    from bhl_robust.cloth.schedule import MACRO_STEP_S as _MACRO
    DT = 0.005
    STEPS = int(round(_MACRO / DT))

    def test_scripted_sweep_builds_and_fits_the_macro_step(self):
        from bhl_robust.cloth.layout import default_spawn_xy
        from bhl_robust.cloth.schedule import MACRO_STEP_S, build_schedule, pinch_arm
        for name in ("sock_a", "shirt_a", "jacket"):
            spec = GARMENT_BY_NAME[name]
            g = np.array(default_spawn_xy(spec))
            s = build_schedule(g, spec, scripted_action(g, spec), self.DT, self.STEPS)
            self.assertTrue(s.valid, f"{name}: {s.reason}")
            self.assertEqual(s.q.shape, (self.STEPS, 5))
            self.assertLessEqual(s.duration, MACRO_STEP_S)
            pinch = pinch_arm(s.joints)
            np.testing.assert_allclose(s.q[0], pinch, atol=1e-9)
            np.testing.assert_allclose(s.q[-1], pinch, atol=1e-9)
            self.assertGreater(float(np.abs(s.q - pinch).max()), 0.05, f"{name}: the arm never moved")

    def test_scripted_schedule_keeps_the_hand_off_the_garment_until_the_sweep(self):
        """Replay the commanded joints through FK of the hand hull (21300604's failure)."""
        from bhl_robust.cloth.arm_fk import hand_point, hull_points, load_chain
        from bhl_robust.cloth.layout import TABLE_TOP_Z, default_spawn_xy, table_top_rect
        from bhl_robust.cloth.reach import load_contact
        from bhl_robust.cloth.schedule import Rect, _world_xyz, build_schedule
        chain, table = load_chain(), load_contact()
        lo, hi = table_top_rect()
        for name in ("sock_a", "shirt_a", "jacket"):
            spec = GARMENT_BY_NAME[name]
            g = np.array(default_spawn_xy(spec))
            s = build_schedule(g, spec, scripted_action(g, spec), self.DT, self.STEPS)
            self.assertTrue(s.valid, f"{name}: {s.reason}")
            t0, t1 = s.sweep_window
            times = np.arange(self.STEPS) * self.DT
            rect = Rect(g, 0.0, 0.5 * np.array(spec.proxy_size[:2]))
            pts = _world_xyz(hull_points(chain, s.q))
            low = pts[..., 2] <= TABLE_TOP_Z + spec.proxy_size[2] + 0.005
            gap = np.where(low, rect.distance(pts[..., :2].reshape(-1, 2)).reshape(low.shape), np.inf).min(axis=1)
            self.assertGreater(float(gap[times < t0].min()), 0.005, f"{name}: hand reaches the garment early")
            over = ((pts[..., 0] >= lo[0]) & (pts[..., 0] <= hi[0]) & (pts[..., 1] >= lo[1]) & (pts[..., 1] <= hi[1]))
            self.assertGreater(float(np.where(over, pts[..., 2], np.inf).min()), TABLE_TOP_Z - 0.002,
                               f"{name}: hand into the table")
            tip = _world_xyz(hand_point(chain, s.q, table.p_hand))[:, :2]
            u = (s.end_xy - s.start_xy) / np.linalg.norm(s.end_xy - s.start_xy)
            during = (times >= t0) & (times <= t1)
            off_line = np.abs(np.cross(u[None], tip[during] - s.start_xy[None]))
            self.assertLess(float(off_line.max()), 0.003, f"{name}: fingertip leaves the sweep line")

    def test_feedforward_is_inverse_dynamics_over_kp_and_fits_the_arm(self):
        from bhl_robust.cloth.arm_fk import inverse_dynamics, load_chain
        from bhl_robust.cloth.layout import default_spawn_xy
        from bhl_robust.cloth import schedule as S
        spec = GARMENT_BY_NAME["shirt_a"]
        g = np.array(default_spawn_xy(spec))
        s = S.build_schedule(g, spec, scripted_action(g, spec), self.DT, self.STEPS)
        self.assertTrue(s.valid, s.reason)
        np.testing.assert_allclose(s.qd, np.gradient(s.q, self.DT, axis=0), atol=1e-9)
        self.assertLessEqual(s.tau_peak, S.TORQUE_BUDGET * S.ARM_EFFORT + 1e-9)
        # Held still at the end: the offset is exactly gravity over Kp.
        tail = inverse_dynamics(load_chain(), s.q[-1:], np.zeros((1, 5)), np.zeros((1, 5)))[0]
        np.testing.assert_allclose(s.q_cmd[-1] - s.q[-1], tail / S.ARM_KP, atol=1e-6)
        # Every phase starts and ends at rest: no velocity jump for the drive to chase.
        self.assertLess(float(np.abs(s.qd[0]).max()), 1e-2)
        self.assertLess(float(np.abs(s.qd[int(s.duration / self.DT) + 2:]).max()), 1e-2)
        self.assertLessEqual(float(np.abs(s.qd).max()), S.MAX_JOINT_SPEED + 0.05)

    def test_refused_plans_hold_the_pinch_with_gravity_compensation(self):
        from bhl_robust.cloth.arm_fk import inverse_dynamics, load_chain
        from bhl_robust.cloth import schedule as S
        spec = GARMENT_BY_NAME["shirt_a"]
        s = S.build_schedule(np.array([0.52, 0.0]), spec, scripted_action(np.array([0.52, 0.0]), spec),
                             self.DT, self.STEPS)
        self.assertFalse(s.valid)
        pinch = S.pinch_arm(s.joints)
        tau = inverse_dynamics(load_chain(), pinch[None], np.zeros((1, 5)), np.zeros((1, 5)))[0]
        np.testing.assert_allclose(s.q_cmd, np.repeat((pinch + tau / S.ARM_KP)[None], self.STEPS, 0), atol=1e-9)
        np.testing.assert_allclose(s.qd, 0.0)

    def test_a_sweep_that_starts_on_the_garment_is_refused(self):
        from bhl_robust.cloth.layout import default_spawn_xy
        from bhl_robust.cloth.schedule import build_schedule
        spec = GARMENT_BY_NAME["shirt_a"]
        g = np.array(default_spawn_xy(spec))
        on_top = encode_physical(0.0, 0.0, np.pi, 0.15, 0.4)
        s = build_schedule(g, spec, on_top, self.DT, self.STEPS)
        self.assertFalse(s.valid)
        self.assertIn("garment", s.reason)

    def test_unreachable_sweep_holds_the_pinch_pose(self):
        from bhl_robust.cloth.schedule import build_schedule, pinch_arm
        spec = GARMENT_BY_NAME["shirt_a"]
        s = build_schedule(np.array([0.52, 0.0]), spec, scripted_action(np.array([0.52, 0.0]), spec),
                           self.DT, self.STEPS)
        self.assertFalse(s.valid)
        np.testing.assert_allclose(s.q, np.repeat(pinch_arm(s.joints)[None], self.STEPS, 0))

    def test_isaac_macro_step_matches_the_schedule(self):
        """The env cfg derives decimation from MACRO_STEP_S; guard that link statically."""
        src = (_REPO / "src" / "bhl_robust" / "tasks" / "cloth_sort_env_cfg.py").read_text()
        self.assertIn("int(round(MACRO_STEP_S / self.sim.dt))", src)
        self.assertNotIn("self.decimation = 8", src)
        self.assertNotIn("self.decimation = 1\n", src)


class ContactTableTests(unittest.TestCase):
    """The fingertip contact table the redesign is placed against."""

    def setUp(self):
        from bhl_robust.cloth.reach import load_contact
        self.t = load_contact()

    def test_built_at_the_measured_best_height(self):
        from bhl_robust.cloth.layout import TABLE_TOP_Z
        self.assertAlmostEqual(self.t.table_top, TABLE_TOP_Z, places=6)
        self.assertGreaterEqual(int(self.t.contact_mask.sum()), 200)
        # The arm cannot lift the hand over most of the table on the same branch;
        # anchors are the edge band that can (78 when built).
        self.assertGreaterEqual(int((self.t.contact_mask & self.t.hover_mask).sum()), 60)
        self.assertEqual(len(self.t.joints), 5)

    def test_neighbouring_cells_share_a_branch(self):
        """Adjacent cells were separate IK solves; a joint move between them left the line."""
        Q, M = self.t.contact_q, self.t.contact_mask
        worst = 0.0
        for i, j in zip(*np.nonzero(M)):
            for a, b in ((i + 1, j), (i, j + 1), (i + 1, j + 1), (i + 1, j - 1)):
                if 0 <= a < M.shape[0] and 0 <= b < M.shape[1] and M[a, b]:
                    worst = max(worst, float(np.abs(Q[i, j] - Q[a, b]).max()))
        self.assertLessEqual(worst, 0.30)

    def test_contact_and_hover_at_an_anchor_are_one_vertical_move(self):
        from bhl_robust.cloth.arm_fk import hand_point, load_chain
        chain = load_chain()
        s = np.linspace(0, 1, 11)[:, None]
        for i, j in zip(*np.nonzero(self.t.contact_mask & self.t.hover_mask)):
            qs = self.t.contact_q[i, j][None] * (1 - s) + self.t.hover_q[i, j][None] * s
            tip = hand_point(chain, qs, self.t.p_hand)
            wander = np.linalg.norm(tip[:, :2] - [self.t.x[i], self.t.y[j]], axis=1).max()
            self.assertLessEqual(float(wander), 0.0101)

    def test_contact_cells_hold_the_hand_just_above_the_table(self):
        from bhl_robust.cloth.arm_fk import hull_points, load_chain
        ix, iy = np.nonzero(self.t.contact_mask)
        low = hull_points(load_chain(), self.t.contact_q[ix, iy])[..., 2].min(axis=1)
        self.assertGreaterEqual(float(low.min()), self.t.table_top)
        self.assertLessEqual(float(low.max()), self.t.table_top + self.t.contact_clearance + 0.0015)

    def test_blended_joints_stay_on_contact_between_cells(self):
        from bhl_robust.cloth.arm_fk import hand_point, hull_points, load_chain
        chain = load_chain()
        rng = np.random.default_rng(3)
        ix, iy = np.nonzero(self.t.contact_mask)
        n = 0
        for k in rng.choice(len(ix), 60, replace=False):
            p = np.array([self.t.x[ix[k]], self.t.y[iy[k]]]) + rng.uniform(-0.009, 0.009, 2)
            q = self.t.q_at(p, "contact", strict=True)
            if q is None:
                continue
            n += 1
            tip = hand_point(chain, q[None], self.t.p_hand)[0]
            self.assertLess(float(np.linalg.norm(tip[:2] - p)), 0.0015)
            low = float(hull_points(chain, q[None])[0, :, 2].min())
            self.assertGreater(low, self.t.table_top - 0.002)
        self.assertGreater(n, 30)

    def test_contact_point_hangs_below_the_hand_link(self):
        """The fingertips, not the link origin: that distinction is the correction."""
        self.assertLess(float(self.t.p_hand[2]), -0.10)

    def test_every_cell_clears_the_thigh_and_stays_right(self):
        ix, iy = np.nonzero(self.t.contact_mask | self.t.hover_mask)
        self.assertTrue(bool(np.all(self.t.y[iy] <= -0.18 + 1e-9)))

    def test_old_table_edge_is_out_of_fingertip_reach(self):
        ix, _ = np.nonzero(self.t.contact_mask)
        self.assertLess(float(self.t.x[ix].max()), 0.39)


class ArmFkTests(unittest.TestCase):
    def test_exported_chain_matched_mujoco(self):
        f = np.load(_REPO / "assets" / "cloth" / "right_arm_chain.npz")
        self.assertLess(float(f["fk_max_error"]), 1e-6)

    def test_hull_reduction_is_tight_and_keeps_the_fingertip_face(self):
        from bhl_robust.cloth.arm_fk import hull_excess, load_chain, reduce_hull
        full = load_chain().hull
        small = reduce_hull(full, 64)
        self.assertLessEqual(len(small), 64)
        self.assertLess(hull_excess(full, small), 0.003)
        self.assertEqual(int((small[:, 2] <= full[:, 2].min() + 1e-3).sum()),
                         int((full[:, 2] <= full[:, 2].min() + 1e-3).sum()))


class ArmTrackingTests(unittest.TestCase):
    """The arm Isaac simulates, in numpy, and what a schedule needs to make it follow."""

    TRACE = _REPO / "results" / "cloth" / "isaac_c0f_v2_trace.json"

    def test_numpy_pd_arm_reproduces_what_isaac_measured(self):
        """21307211, plan 0, before the hand touched anything: same arm, same drive."""
        import json
        from bhl_robust.cloth.arm_fk import load_chain, simulate_pd
        from bhl_robust.cloth import schedule as S
        tr = json.load(open(self.TRACE))
        samples = [x for x in tr["samples"] if x["plan"] == 0 and x["t"] < 0.45]
        ks = np.array([x["k"] for x in samples])
        q_isaac = np.array([x["q"] for x in samples])
        target = np.array([x["q_target"] for x in samples])
        full = np.arange(ks[-1] + 1)
        cmd = np.stack([np.interp(full, ks, target[:, j]) for j in range(5)], axis=1)
        sim = simulate_pd(load_chain(), q_isaac[0], cmd, np.zeros_like(cmd), tr["sim_dt"],
                          S.ARM_KP, S.ARM_KD, S.ARM_EFFORT, S.ARM_ARMATURE, substeps=5)
        self.assertLess(float(np.abs(sim[ks] - q_isaac).max()), 0.005)
        # ...and that trajectory was far from its target: the finding the feedforward answers.
        self.assertGreater(float(np.abs(target - q_isaac).max()), 0.3)

    def test_feedforward_schedule_tracks_where_bare_targets_lag(self):
        from bhl_robust.cloth.arm_fk import hand_point, load_chain, simulate_pd
        from bhl_robust.cloth.layout import default_spawn_xy
        from bhl_robust.cloth.reach import load_contact
        from bhl_robust.cloth import schedule as S
        chain, table = load_chain(), load_contact()
        spec = GARMENT_BY_NAME["shirt_a"]
        g = np.array(default_spawn_xy(spec))
        dt = 0.005
        s = S.build_schedule(g, spec, scripted_action(g, spec), dt, int(round(S.MACRO_STEP_S / dt)))
        self.assertTrue(s.valid, s.reason)
        n = int(np.ceil(s.sweep_window[1] / dt)) + 1           # approach and sweep: where the shirt was lost
        gains = (S.ARM_KP, S.ARM_KD, S.ARM_EFFORT, S.ARM_ARMATURE)
        ff = simulate_pd(chain, s.q[0], s.q_cmd[:n], s.qd[:n], dt, *gains, substeps=5)
        err = np.linalg.norm(hand_point(chain, ff, table.p_hand) - hand_point(chain, s.q[:n], table.p_hand), axis=1)
        self.assertLess(float(err.max()), 0.004, "the feedforward no longer makes the arm follow its schedule")
        bare = simulate_pd(chain, s.q[0], s.q[:n], np.zeros((n, 5)), dt, *gains, substeps=5)
        lag = np.linalg.norm(hand_point(chain, bare, table.p_hand) - hand_point(chain, s.q[:n], table.p_hand), axis=1)
        self.assertGreater(float(lag.max()), 0.03)

    def test_schedule_assumes_the_upstream_arm_actuator(self):
        """ARM_* must be the "arms" group of the shipped HUMANOID_LITE_CFG, parsed from its source."""
        import ast
        from bhl_robust.cloth import schedule as S
        src = (_REPO / "external/Berkeley-Humanoid-Lite/source/berkeley_humanoid_lite_assets/"
               "berkeley_humanoid_lite_assets/robots/berkeley_humanoid_lite.py").read_text()
        tree = ast.parse(src)
        found = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "HUMANOID_LITE_CFG" for t in node.targets):
                for kw in node.value.keywords:
                    if kw.arg == "actuators":
                        for key, val in zip(kw.value.keys, kw.value.values):
                            if getattr(key, "value", None) == "arms":
                                found.append({k.arg: ast.literal_eval(k.value) for k in val.keywords
                                              if k.arg in ("stiffness", "damping", "effort_limit", "armature")})
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0], {"stiffness": S.ARM_KP, "damping": S.ARM_KD,
                                    "effort_limit": S.ARM_EFFORT, "armature": S.ARM_ARMATURE})

    def test_isaac_term_sends_the_feedforward_and_checks_the_actuator(self):
        src = (_REPO / "src" / "bhl_robust" / "tasks" / "cloth_sort_mdp.py").read_text()
        self.assertIn("torch.as_tensor(s.q_cmd", src)
        self.assertIn("torch.as_tensor(s.qd", src)
        self.assertIn("set_joint_velocity_target(vel)", src)
        self.assertIn("self._check_arm_actuator()", src)
        self.assertNotIn("torch.as_tensor(s.q,", src, "the term is sending bare position targets again")

    def test_inverse_dynamics_matched_mujoco_when_exported(self):
        f = np.load(_REPO / "assets" / "cloth" / "right_arm_chain.npz")
        self.assertLess(float(f["id_max_error"]), 1e-6)


class BalanceTests(unittest.TestCase):
    """The free-base stance controller, pinned to the MuJoCo search that found it."""

    SEARCH = _REPO / "results" / "cloth" / "stance" / "search.json"

    def test_constants_are_the_ones_the_search_stood_with(self):
        import json
        from bhl_robust.cloth import balance
        from bhl_robust.eval.coop_replay import JOINTS
        d = json.load(open(self.SEARCH))
        st = d["stance"]
        for side in ("left", "right"):
            self.assertAlmostEqual(balance.STANCE[f"leg_{side}_hip_pitch_joint"], st["hip"])
            self.assertAlmostEqual(balance.STANCE[f"leg_{side}_knee_pitch_joint"], st["knee"])
            self.assertAlmostEqual(balance.STANCE[f"leg_{side}_ankle_pitch_joint"], st["ankle"])
        ff = dict(zip(JOINTS, d["feedforward_rad"]))
        for j, v in balance.FEEDFORWARD.items():
            self.assertAlmostEqual(v, ff[j], places=4)
        stood = {(r["pitch_kp"], r["pitch_kd"], r["roll_kp"], r["roll_kd"]) for r in d["results"] if r["worst_tilt_deg"] < 30}
        self.assertIn(tuple(balance.GAINS), stood)
        self.assertGreater(balance.ROOT_Z, st["root_z"], "the root must not start the soles inside the floor")

    def test_tilt_is_read_in_the_heading_frame(self):
        from bhl_robust.cloth.balance import ankle_offsets
        th = 0.1
        # Robot facing +x (the MuJoCo frame the gains were found in), leaning forward.
        p0, r0 = ankle_offsets(np.sin(th), 0.0, np.cos(th), 0.0, 0.0, heading=0.0, gains=(1, 0, 1, 0))
        # The same lean for a robot facing -x, as Isaac spawns it: toward world -x.
        p1, r1 = ankle_offsets(-np.sin(th), 0.0, np.cos(th), 0.0, 0.0, heading=np.pi, gains=(1, 0, 1, 0))
        self.assertAlmostEqual(p0, th, places=9)
        self.assertAlmostEqual(p1, th, places=9)
        self.assertAlmostEqual(r0, 0.0, places=9)
        self.assertAlmostEqual(r1, 0.0, places=9)
        # Lean toward the robot's right: world -y facing +x, world +y facing -x.
        _, ra = ankle_offsets(0.0, -np.sin(th), np.cos(th), 0.0, 0.0, heading=0.0, gains=(1, 0, 1, 0))
        _, rb = ankle_offsets(0.0, np.sin(th), np.cos(th), 0.0, 0.0, heading=np.pi, gains=(1, 0, 1, 0))
        self.assertAlmostEqual(ra, th, places=9)
        self.assertAlmostEqual(rb, th, places=9)

    def test_up_axis_matches_scipy_in_both_orders(self):
        from scipy.spatial.transform import Rotation as Rot
        from bhl_robust.cloth.kinematics import up_axis
        rng = np.random.default_rng(0)
        for r in Rot.random(20, random_state=1):
            want = r.as_matrix()[:, 2]
            got = np.array(up_axis(r.as_quat(scalar_first=False)[None, :], "xyzw"))[:, 0]
            np.testing.assert_allclose(got, want, atol=1e-9)
            got = np.array(up_axis(r.as_quat(scalar_first=True)[None, :], "wxyz"))[:, 0]
            np.testing.assert_allclose(got, want, atol=1e-9)

    def test_isaac_term_reads_tilt_through_the_probed_order(self):
        src = (_REPO / "src" / "bhl_robust" / "tasks" / "cloth_sort_mdp.py").read_text()
        self.assertIn("up_axis(q, QUAT_ORDER)", src)
        self.assertIn("yaw_atan2_args(q, QUAT_ORDER)", src)
        self.assertIn("root_ang_vel_w", src)


class StanceLayoutTests(unittest.TestCase):
    """The layout's heights follow the stance; the solved arm configurations do not change."""

    def test_heights_rise_with_the_settled_root(self):
        from bhl_robust.cloth import balance, layout
        from bhl_robust.cloth.arm_fk import load_chain
        from bhl_robust.cloth.reach import load_contact
        f = np.load(_REPO / "assets" / "cloth" / "right_hand_tip_table.npz")
        self.assertAlmostEqual(layout.SOLVE_ROOT_Z, float(f["planted_root_z"]), places=12)
        self.assertAlmostEqual(layout.STANDING_ROOT_Z, balance.SETTLED_ROOT_Z)
        self.assertAlmostEqual(layout.TABLE_TOP_Z - layout.STANDING_ROOT_Z, 0.30 - layout.SOLVE_ROOT_Z, places=9)
        self.assertAlmostEqual(load_contact().table_top, layout.TABLE_TOP_Z, places=9)
        self.assertAlmostEqual(load_chain().root_z, layout.STANDING_ROOT_Z, places=9)
        self.assertEqual(layout.ROBOT_ROOT_Z, balance.ROOT_Z)

    def test_fingertip_cells_are_the_same_distance_above_the_table(self):
        """A cell's hull, relative to the table top, is where it was solved."""
        from bhl_robust.cloth.arm_fk import hull_points, load_chain
        from bhl_robust.cloth.reach import load_contact
        t, chain = load_contact(), load_chain()
        ix, iy = np.nonzero(t.contact_mask)
        low = hull_points(chain, t.contact_q[ix, iy])[..., 2].min(axis=1) - t.table_top
        self.assertGreaterEqual(float(low.min()), -1e-9)
        self.assertLessEqual(float(low.max()), t.contact_clearance + 0.0015)

    def test_every_rigid_cloth_scene_stands_in_the_stance(self):
        src = (_REPO / "src" / "bhl_robust" / "tasks" / "cloth_sort_env_cfg.py").read_text()
        body = src[src.index("class ClothSortRigidEnvCfg"):src.index("class ClothSortRigidFiveEnvCfg")]
        self.assertIn("balance.STANCE", body)
        self.assertIn("self.actions.sweep.balance = ", body)


class IsaacConfigWiringTests(unittest.TestCase):
    """Static guards over the Isaac task modules.

    These parse the source instead of importing it: the modules need a live
    SimulationApp, so a login-node test cannot construct them. That is exactly
    why 21228029 got as far as a GPU node before failing. The class of bug it
    hit is statically visible, so it is checked here where it costs no queue
    time.
    """

    _ISAAC_MODULES = ("cloth_sort_mdp.py", "cloth_sort_env_cfg.py")

    def _trees(self):
        import ast
        for name in self._ISAAC_MODULES:
            path = _REPO / "src" / "bhl_robust" / "tasks" / name
            yield name, ast.parse(path.read_text()), path.read_text()

    def test_configclass_never_defaults_class_type_to_none(self):
        """``configclass`` freezes field defaults when it builds the dataclass.

        A cfg that declares ``class_type: type = None`` and patches the class
        attribute afterwards leaves every *instance* holding None, and the
        action manager calls it: ``TypeError: 'NoneType' object is not
        callable``.
        """
        import ast
        for name, tree, _ in self._trees():
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                decorated = any(
                    (isinstance(d, ast.Name) and d.id == "configclass")
                    or (isinstance(d, ast.Attribute) and d.attr == "configclass")
                    for d in node.decorator_list
                )
                if not decorated:
                    continue
                for stmt in node.body:
                    if not isinstance(stmt, ast.AnnAssign):
                        continue
                    if not (isinstance(stmt.target, ast.Name) and stmt.target.id == "class_type"):
                        continue
                    self.assertIsNotNone(
                        stmt.value,
                        f"{name}:{node.name}.class_type is declared with no default",
                    )
                    self.assertNotIsInstance(
                        stmt.value, ast.Constant,
                        f"{name}:{node.name}.class_type defaults to a constant "
                        f"({ast.dump(stmt.value)}); it must default to the term class itself",
                    )

    def test_no_post_hoc_class_type_patching(self):
        """Module-level ``SomeCfg.class_type = SomeTerm`` does not take effect."""
        import ast
        for name, tree, _ in self._trees():
            for node in tree.body:
                if not isinstance(node, ast.Assign):
                    continue
                for tgt in node.targets:
                    if isinstance(tgt, ast.Attribute) and tgt.attr == "class_type":
                        self.fail(
                            f"{name}: module-level assignment to .class_type at line "
                            f"{node.lineno}. configclass has already frozen the field "
                            f"default, so instances keep the old value."
                        )

    def test_action_term_class_precedes_its_cfg(self):
        """The term class must be defined before the cfg that defaults to it."""
        import ast
        for name, tree, _ in self._trees():
            order = {
                n.name: n.lineno for n in tree.body if isinstance(n, ast.ClassDef)
            }
            for n in tree.body:
                if not isinstance(n, ast.ClassDef):
                    continue
                for stmt in n.body:
                    if (
                        isinstance(stmt, ast.AnnAssign)
                        and isinstance(stmt.target, ast.Name)
                        and stmt.target.id == "class_type"
                        and isinstance(stmt.value, ast.Name)
                        and stmt.value.id in order
                    ):
                        self.assertLess(
                            order[stmt.value.id], n.lineno,
                            f"{name}: {n.name} defaults class_type to {stmt.value.id}, "
                            f"which is defined later in the file",
                        )

    _ISAAC_SCRIPTS = (
        "scripts/bench/cloth_sort_smoke.py",
        "scripts/bench/cloth_sort_isaac_bench.py",
        "scripts/cloth/eval_isaac.py",
    )

    def test_cloth_resolution_is_never_assigned_after_construction(self):
        """The mesh is spawned in ``__post_init__``.

        ``cfg.cloth_resolution = 10`` afterwards relabels the config without
        changing the mesh, so a bench row would say 10x10 over an 8x8 cloth.
        Resolution has to go through ``build_cfg`` and reach ``__init__``.
        """
        import ast
        for rel in self._ISAAC_SCRIPTS:
            tree = ast.parse((_REPO / rel).read_text())
            for node in ast.walk(tree):
                if not isinstance(node, ast.Assign):
                    continue
                for tgt in node.targets:
                    if isinstance(tgt, ast.Attribute) and tgt.attr == "cloth_resolution":
                        self.fail(
                            f"{rel}:{node.lineno} assigns .cloth_resolution after "
                            f"construction; the spawned mesh will not change. "
                            f"Pass it to build_cfg() instead."
                        )

    def test_isaac_scripts_construct_through_build_cfg(self):
        """One construction path, so the read-back guard cannot be bypassed."""
        import ast
        for rel in self._ISAAC_SCRIPTS:
            tree = ast.parse((_REPO / rel).read_text())
            for node in ast.walk(tree):
                # gym.spec(...).kwargs["env_cfg_entry_point"]() called directly
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Subscript)
                    and isinstance(node.func.value, ast.Attribute)
                    and node.func.value.attr == "kwargs"
                ):
                    self.fail(
                        f"{rel}:{node.lineno} calls the env cfg entry point directly. "
                        f"Use build_cfg(), which verifies the spawned cloth "
                        f"resolution against the requested one."
                    )

    def test_no_bitand_inside_an_unparenthesised_comparison(self):
        """``a <= b & c`` is a chained comparison, not a mask.

        ``&`` binds tighter than ``<=`` in Python, so the missing parentheses
        silently rewrite the predicate and torch raises
        ``"bitwise_and_cuda" not implemented for 'Float'``. 21233802 died of
        exactly this in the basket test.
        """
        import ast
        for name, tree, _ in self._trees():
            for node in ast.walk(tree):
                if not isinstance(node, ast.Compare):
                    continue
                for operand in [node.left, *node.comparators]:
                    if (
                        isinstance(operand, ast.BinOp)
                        and isinstance(operand.op, (ast.BitAnd, ast.BitOr))
                    ):
                        self.fail(
                            f"{name}:{node.lineno} has a bitwise & / | inside an "
                            f"unparenthesised comparison. Parenthesise each "
                            f"comparison: ((a) <= b) & ((c) <= d)."
                        )

    def test_isaac_and_kinematic_score_the_same_basket_volume(self):
        """One basket AABB, not two.

        The Isaac predicate used to re-derive the box from ``BASKET_INNER``
        with a half-extent for z, giving a 0.07 m ceiling against the
        kinematic 0.14 m: the same garment scored differently in the two
        engines. It must read ``layout.basket_aabb``.
        """
        import ast
        src = (_REPO / "src" / "bhl_robust" / "tasks" / "cloth_sort_mdp.py").read_text()
        self.assertIn(
            "basket_aabb", src,
            "cloth_sort_mdp must score against layout.basket_aabb",
        )
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == "BASKET_INNER":
                self.fail(
                    "cloth_sort_mdp re-derives the basket box from BASKET_INNER; "
                    "use layout.basket_aabb so the two engines cannot drift."
                )

    def test_basket_aabb_contains_its_own_centre(self):
        for bid in BASKET_IDS:
            box = basket_aabb(bid)
            c = basket_center(bid)
            self.assertTrue(bool(np.all(c >= box.low)) and bool(np.all(c <= box.high)))
            # z spans the full inner height, not half of it.
            self.assertAlmostEqual(float(box.high[2] - box.low[2]), 0.14, places=6)

    def test_cloth_furniture_is_collider_only_and_the_rest_is_untouched(self):
        """Cloth scenery must not be a physics body; maze/coop scenery must stay one.

        Newton turns a prim with ``rigid_props`` into a Newton physics body and
        ``AssetBaseCfg`` refuses it (21233868). The cloth table and baskets are
        therefore collider-only -- in the rigid scene too, so C0/C1 and C2/C3
        are not scoring different table physics. Everything else keeps
        ``rigid_props``, because published numbers were measured on it.
        """
        import ast
        src = (_REPO / "src" / "bhl_robust" / "tasks" / "furniture.py").read_text()
        tree = ast.parse(src)
        funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}

        def collider_only_calls(fn):
            out = []
            for node in ast.walk(fn):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                        and node.func.id == "_box":
                    out.append(any(
                        kw.arg == "collider_only"
                        and isinstance(kw.value, ast.Constant) and kw.value.value is True
                        for kw in node.keywords
                    ))
            return out

        for name in ("cloth_table", "sorting_basket"):
            self.assertIn(name, funcs)
            flags = collider_only_calls(funcs[name])
            self.assertTrue(flags, f"{name} spawns no boxes?")
            self.assertTrue(
                all(flags),
                f"{name} spawns a box that is still a rigid body; Newton will "
                f"reject it via FrameView",
            )

        for name in ("corridor_walls", "plinth", "shelf", "net"):
            if name not in funcs:
                continue
            self.assertFalse(
                any(collider_only_calls(funcs[name])),
                f"{name} became collider-only; that changes the physics the "
                f"maze/coop numbers were measured on",
            )

    def test_every_reported_episode_metric_is_actually_assigned(self):
        """A metric that is never written reports its default forever.

        ``eval_isaac.py`` published ``success_rate`` and ``fall_rate`` while
        assigning neither: ``em.fell`` was never set, and the ``success``
        lookup raised into a bare ``except`` because ``_term_dones`` is a
        tensor, not a dict. Both were structurally pinned to 0.0.
        """
        import ast
        from bhl_robust.cloth.metrics import EpisodeMetrics
        src = (_REPO / "scripts" / "cloth" / "eval_isaac.py").read_text()
        tree = ast.parse(src)
        assigned = set()
        for node in ast.walk(tree):
            # `em.x = ...` and `em.x += ...` both write the field; a counter is
            # almost always the second kind.
            targets = node.targets if isinstance(node, ast.Assign) else (
                [node.target] if isinstance(node, ast.AugAssign) else [])
            for tgt in targets:
                if isinstance(tgt, ast.Attribute) and isinstance(tgt.value, ast.Name) \
                        and tgt.value.id == "em":
                    assigned.add(tgt.attr)
        # Fields the run-level summary derives its rates from.
        for field in ("success", "fell", "invalid_trajectory"):
            self.assertIn(field, EpisodeMetrics.__dataclass_fields__)
            self.assertIn(
                field, assigned,
                f"eval_isaac.py reports a rate over EpisodeMetrics.{field} but "
                f"never assigns it; it can only ever report the default",
            )

    def test_no_bare_except_around_the_measurement(self):
        """``except Exception: pass`` around a metric read hides a wrong number."""
        import ast
        tree = ast.parse((_REPO / "scripts" / "cloth" / "eval_isaac.py").read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler):
                continue
            if all(isinstance(b, ast.Pass) for b in node.body):
                self.fail(
                    f"eval_isaac.py:{node.lineno} swallows an exception with a "
                    f"bare pass. That is how success_rate stayed 0.0."
                )

    def test_isaac_resolution_counts_cells_not_vertices(self):
        """An 8x8 Isaac cloth is 81 vertices, not 64.

        ``MeshRectangleCfg(resolution=(n, n))`` counts cells. Job 21233960
        logged "Particles per body: 81" and "Registered UsdGeom.Mesh: 81
        vertices" for resolution 8. Reporting 64 would put a wrong vertex
        count in the throughput table.
        """
        self.assertEqual(isaac_grid_counts(8)[0], 81)
        self.assertEqual(isaac_grid_counts(10)[0], 121)
        self.assertEqual(isaac_grid_counts(16)[0], 289)
        # grid_counts keeps vertices-per-side semantics for the OBJ writer.
        self.assertEqual(grid_counts(8)[0], 64)
        for n in (8, 10, 12, 16):
            self.assertEqual(isaac_grid_counts(n)[0], (n + 1) ** 2)

    def test_isaac_scripts_report_isaac_vertex_counts(self):
        """The Isaac benches must not label a cloth with the OBJ vertex count."""
        import ast
        for rel in ("scripts/bench/cloth_sort_smoke.py",
                    "scripts/bench/cloth_sort_isaac_bench.py"):
            tree = ast.parse((_REPO / rel).read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                        and node.func.id == "grid_counts":
                    self.fail(
                        f"{rel}:{node.lineno} uses grid_counts for an Isaac cloth; "
                        f"resolution counts cells, so use isaac_grid_counts"
                    )

    def test_registered_cloth_ids_match_the_documented_ladder(self):
        """The four gym ids named in docs/CLOTH_SORT.md are the ones registered."""
        import ast
        src = (_REPO / "src" / "bhl_robust" / "tasks" / "__init__.py").read_text()
        doc = (_REPO / "docs" / "CLOTH_SORT.md").read_text()
        ids = [
            "ClothSort-BHL-Rigid-Oracle-v0",
            "ClothSort-BHL-RigidFive-Oracle-v0",
            "ClothSort-BHL-Deformable-Oracle-v0",
            "ClothSort-BHL-ActiveCloth-Oracle-v0",
            "ClothSort-BHL-RigidFixedBase-Oracle-v0",
            "ClothSort-BHL-DeformableFixedBase-Oracle-v0",
        ]
        for tid in ids:
            self.assertIn(tid, src, f"{tid} is documented but not registered")
            self.assertIn(tid, doc, f"{tid} is registered but not documented")


if __name__ == "__main__":
    unittest.main()
