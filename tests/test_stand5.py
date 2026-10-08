"""CubeToShelfStand5 terms, rules and launchers, without Isaac Lab.

`stand5_mdp` is loaded from its file path, as the launchers load it (the
`bhl_robust.tasks` package registers gym ids and needs a running simulator); it
loads `stand4_mdp` / `stand_mdp` the same way. Synthetic quaternions are written
(w, x, y, z), stored in the stack's layout with `quat_order.reorder` and read back
with `quat_order.unpack_wxyz` (the 2976f36 helpers): (x, y, z, w) as on v60 unless
a test says otherwise.

The fake-tree tests run the REAL training launcher (its `source _env.sh`
line points at a stub and cache paths stay inside the fixture) in its real, non-smoke mode on a throw-away copy of the
files it reads; they take ~3.5 min and are skipped when STAND5_SKIP_SLOW=1.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
from bhl_robust import quat_order as qo  # noqa: E402  (import-safe off-simulator)

S5_PATH = REPO / "src/bhl_robust/tasks/stand5_mdp.py"
S4_PATH = REPO / "src/bhl_robust/tasks/stand4_mdp.py"
STAND_MDP = REPO / "src/bhl_robust/tasks/stand_mdp.py"
COOP_MDP = REPO / "src/bhl_robust/tasks/coop_lift_mdp.py"
TASK_V2_CFG = REPO / "src/bhl_robust/tasks/task_v2_env_cfg.py"
TASKS_INIT = REPO / "src/bhl_robust/tasks/__init__.py"
LEDGER = REPO / "SLURM_JOBS.md"
TRAIN3 = REPO / "slurm/repo20260923/gpu_v2_stand3_train.sbatch"
TRAIN4 = REPO / "slurm/repo20260923/gpu_v2_stand4_train.sbatch"
TRAIN5 = REPO / "slurm/repo20260923/gpu_v2_stand5_train.sbatch"
SMOKE4 = REPO / "slurm/repo20260923/gpu_v2_stand4_smoke.sbatch"
SMOKE5 = REPO / "slurm/repo20260923/gpu_v2_stand5_smoke.sbatch"
INNER5 = REPO / "slurm/inner/inner_v2_stand5_smoke.sh"
HOSTPY = Path(sys.executable)

#: HEAD 3a67bc2 (the branch commit this workstream started from): byte length and
#: sha256 of the files Stand5 must leave as they are.
BASE = "3a67bc2"
TASK_V2_HEAD = (39538, "c2210bb02d4c8ba867aafce3a546b02f91eb19355c372bcee760559e5e6b3218")
PINNED_SHA = {
    "src/bhl_robust/tasks/stand4_mdp.py": "8fac7b6c8e9fae407769960bfdefa28bb75cf2871bcc8af790a085f480db516b",
    "src/bhl_robust/tasks/stand_mdp.py": "53081846295bb72e64ff05de726e8a18ea8e7284844c9eff43434ee57ee3aaea",
    "src/bhl_robust/tasks/coop_lift_mdp.py": "5fc2b8c8a26c311597f18498c99f0eaa791c14d67c8cea39e4e6302569daa3fd",
    "src/bhl_robust/tasks/coop_lift_env_cfg.py": "a92a2359426c734773436a98fcbe163ce1465b59c999611cb976feb6737a8b89",
    "src/bhl_robust/quat_order.py": "59d1921bc95765cb0d5633164003e1f78a15e3a4f7cf66bb231979be6b087ef2",
    "slurm/repo20260923/gpu_v2_stand4_train.sbatch": "05775c5408b99f60afafc8bd7dc42aad1db291927b2eb241bb5bfc5a234b7c32",
    "slurm/repo20260923/gpu_v2_stand4_smoke.sbatch": "d47913b9f0c13fae979cc284c55db4a0cb8653ca25c126758bc70e7f7b5b70df",
}
STAND5_END = "# --- end stand5 ---"


def _load(path=S5_PATH, name="stand5_mdp_under_test"):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s5 = _load()
s4 = s5.s4
sm = s5.sm
H = s4.CUBE_HALF
S2 = math.sqrt(2.0)

REF_DIR = REPO / s5.STAND4_REF_DIR
REF_PARAMS = REF_DIR / f"params_{s5.STAND4_REF_SMOKE_JOB}"
REF_ENV = REF_PARAMS / "env.yaml"
REF_AGENT = REF_PARAMS / "agent.yaml"
REF_LOG = REF_DIR / f"train_{s5.STAND4_REF_SMOKE_JOB}.log"
REF_SHA = REF_DIR / f"sha256_{s5.STAND4_REF_SMOKE_JOB}.txt"
OTHER_STAND4_SMOKE = (REPO / "external/Berkeley-Humanoid-Lite/logs/rsl_rl/task_v2/"
                      "2026-10-02_03-12-09_v2-cubetoshelfstand4-blind-s100-j21506555-smoke/params")
HAVE_REF = REF_ENV.is_file() and REF_AGENT.is_file() and REF_LOG.is_file() and REF_SHA.is_file()


def q_axis(axis, deg):
    """(w, x, y, z) of a rotation by `deg` about `axis`."""
    return s5.quat_axis_angle(axis, deg)


IDENT = q_axis((0, 0, 1), 0.0)


def cols(rows_wxyz, order="xyzw", dtype=torch.float64):
    """Store (w, x, y, z) rows in `order` and read them back as (w, x, y, z) columns."""
    stored = torch.tensor([qo.reorder(r, order) for r in rows_wxyz], dtype=dtype)
    return qo.unpack_wxyz(stored, order=order)


def zin(frames, bodies, order="xyzw"):
    return s5.zaxis_in_frame(cols(frames, order), cols(bodies, order))


def _norm(text: str) -> str:
    return " ".join(text.split())


def _header(path: Path) -> str:
    """The comment header (before `set -euo pipefail`), '#' stripped, normalised."""
    head = path.read_text().split("set -euo pipefail")[0]
    return _norm(" ".join(ln[1:] for ln in head.splitlines() if ln.startswith("#")))


def _kill_block(path: Path) -> list[str]:
    lines = path.read_text().splitlines()
    i = next(k for k, ln in enumerate(lines) if ln.startswith("#   kill      Stand2's rule:"))
    j = next(k for k in range(i + 1, len(lines)) if lines[k].startswith("#   complete"))
    return lines[i:j]


def _sbatch_flags(path: Path) -> dict:
    return dict(re.findall(r"^#SBATCH --([\w-]+)=?(.*)$", path.read_text(), re.M))


def _git_show(rev, path):
    try:
        return subprocess.run(["git", "-C", str(REPO), "show", f"{rev}:{path}"], capture_output=True,
                              check=True).stdout
    except Exception:                                               # noqa: BLE001
        return None


# ------------------------------------------- synthetic Stand5 dumps (from Stand4's)


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _span(lines, path):
    """(start, end) line indices of the mapping at `path` in an Isaac Lab dump."""
    lo, hi, start = 0, len(lines), None
    for depth, key in enumerate(path):
        ind = 2 * depth
        start = next((i for i in range(lo, hi) if _indent(lines[i]) == ind and (
            lines[i].strip() == f"{key}:" or lines[i].strip().startswith(f"{key}: "))), None)
        if start is None:
            raise KeyError(path)
        end = start + 1
        while end < hi:
            ln = lines[end]
            if ln.strip() and (_indent(ln) < ind or (_indent(ln) == ind and not ln.lstrip().startswith("- "))):
                break
            end += 1
        lo, hi = start + 1, end
    return start, hi


def stand5_env_yaml(stand4_env: str) -> str:
    """Stand4's dumped env.yaml with exactly Stand5's two changes, as Isaac Lab would
    dump them: object_zaxis_a/b (copies of object_pos_a/b's blocks -- same params
    robot_cfg, clip +/-100, no noise, no scale -- with Stand5's func) appended to
    both groups; lift_height's func and its two new params."""
    lines = stand4_env.splitlines()
    for g in ("critic", "policy"):                  # the later block first: indices stay valid
        _, end = _span(lines, ("observations", g))
        new = []
        for name, robot in s5.OBS_TERM_ROBOTS:
            src = "object_pos_a" if robot == "robot_a" else "object_pos_b"
            a, b = _span(lines, ("observations", g, src))
            blk = list(lines[a:b])
            blk[0] = blk[0].replace(f"{src}:", f"{name}:", 1)
            blk = [ln.replace("bhl_robust.tasks.coop_lift_mdp:object_pos_in_root", s5.FUNC_OBS5)
                   for ln in blk]
            new += blk
        lines[end:end] = new
    a, b = _span(lines, ("curriculum", "lift_height"))
    for i in range(a, b):
        lines[i] = lines[i].replace(f"func: {s5.FUNC_CURR4}", f"func: {s5.FUNC_CURR5}")
    _, pe = _span(lines, ("curriculum", "lift_height", "params"))
    lines[pe:pe] = ["      clearance: 0.02", "      tilt_max_deg: 15.0"]
    return "\n".join(lines) + "\n"


def stand5_agent_yaml(stand4_agent: str, run_name="v2-cubetoshelfstand5-blind-s100-j1-smoke") -> str:
    return re.sub(r"(?m)^run_name: .*$", f"run_name: {run_name}", stand4_agent)


def edit_block(text: str, path, old: str, new: str) -> str:
    """Replace `old` by `new` once, inside the dump's mapping at `path` only."""
    lines = text.splitlines()
    a, b = _span(lines, path)
    blk = "\n".join(lines[a:b])
    assert old in blk, (path, old)
    lines[a:b] = blk.replace(old, new, 1).split("\n")
    return "\n".join(lines) + "\n"


def drop_block(text: str, path) -> str:
    lines = text.splitlines()
    a, b = _span(lines, path)
    del lines[a:b]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------- imports


class ImportAndConstantTests(unittest.TestCase):
    def test_imports_without_isaaclab(self):
        had = {k for k in sys.modules if k == "isaaclab" or k.startswith("isaaclab.")}
        _load(name="stand5_again")
        now = {k for k in sys.modules if k == "isaaclab" or k.startswith("isaaclab.")}
        self.assertEqual(now - had, set())

    def test_reuses_stand4(self):
        self.assertEqual(s5.s4.__file__, str(S4_PATH))
        self.assertIs(s5.sm, s5.s4.sm)
        self.assertEqual(s5.STAND4_RULE, s4.PREDECLARED_RULE)
        self.assertEqual((s5.PRECEDENCE_NOTE, s5.COMPLETE_NOTE, s5.SUCCESS_NOTE, s5.TILT_NOTE),
                         (s4.PRECEDENCE_NOTE, s4.COMPLETE_NOTE, s4.SUCCESS4_NOTE, s4.TILT_NOTE))

    def test_frozen_constants(self):
        self.assertEqual((s5.CURR_CLEARANCE, s5.CURR_TILT_MAX_DEG), (0.02, 15.0))
        self.assertEqual((s5.STAND5_MAX_ITER, s5.COMPLETE_ITER, s5.RESULT_WINDOW, s5.RESULT_MIN_SUCCESS,
                          s5.STAND5_SEEDS), (8000, 7999, 200, 0.10, (0, 1)))
        self.assertEqual(s5.STAND4_OBS_WIDTH, {"policy": 194, "critic": 206})
        self.assertEqual(s5.STAND5_OBS_WIDTH, {"policy": 200, "critic": 212})
        self.assertEqual(s5.OBS_TERM_ROBOTS, (("object_zaxis_a", "robot_a"), ("object_zaxis_b", "robot_b")))
        self.assertEqual(s5.CURRICULUM5_PARAMS, {
            "term_name": "lifting_object", "step": 0.02, "min_height": 0.04, "max_height": 0.06,
            "success_rate_target": 0.35, "clearance": 0.02, "tilt_max_deg": 15.0})
        self.assertEqual((s5.SMOKE_SEED, s5.SMOKE_PROBE_ENVS, s5.SMOKE_ITERS, s5.SMOKE_ENVS),
                         (100, 16, 20, 1024))
        self.assertNotIn(s5.SMOKE_SEED, s5.STAND5_SEEDS)

    def test_design_and_rules_are_verbatim_in_the_ledger(self):
        ledger = LEDGER.read_text()
        i = ledger.index("**User approval recorded 2026-10-03 09:55**")
        item = ledger[i:]
        self.assertIn("- " + s5.DESIGN_S[0] + "\n", item)
        for line in s5.DESIGN_S[1:]:
            self.assertIn("  - " + line + "\n", item)
        self.assertEqual(s5.PREDECLARED_RULE5, s5.DESIGN_S[3])
        self.assertTrue(s5.PREDECLARED_RULE5.startswith("Stand4's rule verbatim: the kill rule at model_1000"))
        # Stand4's rule as predeclared in the ledger (2026-10-02 03:25)
        self.assertIn('"' + s4.PREDECLARED_RULE + '"', ledger)

    def test_labels(self):
        for s in ("LEARNED crew policy", "blind", "ORACLE", "now including cube orientation",
                  "object_pos_a/b", "object_zaxis_a/b", "base_lin_vel_a/b", "object_lin_vel",
                  "object_ang_vel", "MODIFIED hand colliders", "the actor and the critic both"):
            self.assertIn(s, s5.LABELS_NOTE5)
        self.assertIn("never compared with CubeToShelfStand3 or CubeToShelf", s5.TASK5_NOTE)
        self.assertEqual(s5.PASS_LABEL5, "PASS: learned roll-proof placement (Stand5)")

    def test_no_raw_quaternion_literals_and_no_overlay_literal(self):
        src = S5_PATH.read_text()
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Tuple) and len(node.elts) == 4 and all(
                    isinstance(e, ast.Constant) and isinstance(e.value, (int, float)) for e in node.elts):
                self.fail(f"numeric 4-tuple literal at line {node.lineno}")
        self.assertIn("from bhl_robust.quat_order import unpack_wxyz", src)
        self.assertNotIn("root_quat_w[:", src)
        self.assertNotIn("berkeley_humanoid_lite_hand_colliders", src)   # test_stand4's overlay check
        self.assertIn("native_quat(q) for q in quats_wxyz", INNER5.read_text())   # probe teleports


# ------------------------------------------------- (i) the orientation observation


class OrientationTermTests(unittest.TestCase):
    """The cube's own z axis in a robot's root frame, on synthetic (x, y, z, w) quaternions."""

    def assertVec(self, got, want, tol=1e-9):
        self.assertTrue(torch.allclose(got, torch.tensor(want, dtype=got.dtype), atol=tol), (got, want))

    def test_world_axis_of_every_smoke_case(self):
        for case, q in s5.AXIS_CASES.items():
            got = s5.body_zaxis_world(*cols([q]))[0]
            self.assertVec(got, s5.EXPECTED_WORLD_Z[case])

    def test_rolled_90_about_x_reads_along_y(self):
        ident = [IDENT] * 7
        cubes = [q_axis((1, 0, 0), 90), q_axis((1, 0, 0), -90), q_axis((0, 1, 0), 90),
                 q_axis((0, 1, 0), -90), q_axis((1, 0, 0), 180), q_axis((0, 0, 1), 37), IDENT]
        got = zin(ident, cubes)
        want = [(0, -1, 0), (0, 1, 0), (1, 0, 0), (-1, 0, 0), (0, 0, -1), (0, 0, 1), (0, 0, 1)]
        for g, w in zip(got, want):
            self.assertVec(g, w)

    def test_in_a_yawed_or_rolled_robot_frame(self):
        yaw_p, yaw_m, roll_x = q_axis((0, 0, 1), 90), q_axis((0, 0, 1), -90), q_axis((1, 0, 0), 90)
        frames = [yaw_p, yaw_p, yaw_m, roll_x, yaw_p]
        cubes = [q_axis((1, 0, 0), 90), q_axis((0, 1, 0), 90), q_axis((1, 0, 0), 90), IDENT, IDENT]
        want = [(-1, 0, 0), (0, -1, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
        for g, w in zip(zin(frames, cubes), want):
            self.assertVec(g, w)
        # a yaw of the robot never changes the last component (the tilt the clauses read)
        g = torch.Generator().manual_seed(3)
        cubes = [q_axis(tuple(torch.randn(3, generator=g).tolist()), float(torch.rand(1, generator=g)) * 180)
                 for _ in range(16)]
        yaws = [q_axis((0, 0, 1), float(a)) for a in torch.linspace(-180, 180, 16)]
        a, b = zin([IDENT] * 16, cubes), zin(yaws, cubes)
        self.assertTrue(torch.allclose(a[:, 2], b[:, 2], atol=1e-12))

    def test_against_an_independent_reference(self):
        """R_r^T R_c e_z vs rotating e_z by conj(q_r) * q_c with the vector formula."""
        g = torch.Generator().manual_seed(11)
        qr = torch.randn(512, 4, generator=g, dtype=torch.float64)
        qc = torch.randn(512, 4, generator=g, dtype=torch.float64) * 3.0     # not normalised
        got = s5.zaxis_in_frame(qr.unbind(-1), qc.unbind(-1))

        def mul(a, b):
            aw, ax, ay, az = a.unbind(-1)
            bw, bx, by, bz = b.unbind(-1)
            return torch.stack((aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                                aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw), -1)
        rn, cn = qr / qr.norm(dim=-1, keepdim=True), qc / qc.norm(dim=-1, keepdim=True)
        rel = mul(rn * torch.tensor([1.0, -1.0, -1.0, -1.0], dtype=torch.float64), cn)
        w, u = rel[:, :1], rel[:, 1:]
        v = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64).expand_as(u)
        t = 2.0 * torch.cross(u, v, dim=-1)
        want = v + w * t + torch.cross(u, t, dim=-1)
        self.assertTrue(torch.allclose(got, want, atol=1e-9))
        self.assertTrue(torch.allclose(got.norm(dim=-1), torch.ones(512, dtype=torch.float64), atol=1e-9))

    def test_the_storage_order_matters_and_the_control_cases_detect_it(self):
        stored = torch.tensor([qo.reorder(q, "xyzw") for q in s5.AXIS_CASES.values()], dtype=torch.float64)
        right = s5.body_zaxis_world(*qo.unpack_wxyz(stored, order="xyzw"))
        wrong = s5.body_zaxis_world(*qo.unpack_wxyz(stored, order="wxyz"))
        dev = {c: float((wrong[i] - right[i]).abs().max()) for i, c in enumerate(s5.AXIS_CASES)}
        self.assertEqual({c for c, d in dev.items() if d > 0.5 + 1e-6}, set(s5.ORDER_CONTROL_CASES))
        self.assertAlmostEqual(dev["x+90"], 1.0, places=9)        # read swapped: a yaw, +z stays +z
        self.assertAlmostEqual(dev["x180"], 2.0, places=9)
        self.assertAlmostEqual(dev["y+90"], 0.0, places=9)        # why y+90 is not a control

    def test_v51_wxyz_storage_gives_the_same_vectors(self):
        cubes = list(s5.AXIS_CASES.values())
        frames = [q_axis((0, 0, 1), 90)] * len(cubes)
        self.assertTrue(torch.allclose(zin(frames, cubes, "xyzw"), zin(frames, cubes, "wxyz"), atol=1e-12))

    def test_env_term_reads_both_quaternions_through_unpack(self):
        fake_coop = types.SimpleNamespace(_t=lambda v: v)
        robots = {"robot_a": q_axis((0, 0, 1), 90), "robot_b": q_axis((0, 0, 1), -90)}
        cubes = [q_axis((1, 0, 0), 90), IDENT, q_axis((1, 0, 0), 180)]

        def scene(order):
            data = lambda q: types.SimpleNamespace(data=types.SimpleNamespace(  # noqa: E731
                root_quat_w=torch.tensor([qo.reorder(x, order) for x in q], dtype=torch.float32)))
            return {"robot_a": data([robots["robot_a"]] * 3), "robot_b": data([robots["robot_b"]] * 3),
                    "object": data(cubes)}
        with mock.patch.object(s5.sm, "_coop", lambda: fake_coop), \
                mock.patch.object(qo, "_CACHED_ORDER", "xyzw"):
            env = types.SimpleNamespace(scene=scene("xyzw"))
            a = s5.object_zaxis_in_root(env, robot_cfg=types.SimpleNamespace(name="robot_a"))
            b = s5.object_zaxis_in_root(env, robot_cfg=types.SimpleNamespace(name="robot_b"))
        self.assertEqual((tuple(a.shape), a.dtype), ((3, 3), torch.float32))
        for g, w in zip(a, [(-1, 0, 0), (0, 0, 1), (0, 0, -1)]):
            self.assertVec(g.double(), w, tol=1e-6)
        for g, w in zip(b, [(1, 0, 0), (0, 0, 1), (0, 0, -1)]):
            self.assertVec(g.double(), w, tol=1e-6)
        # the same simulator state on a (w, x, y, z) stack reads the same
        with mock.patch.object(s5.sm, "_coop", lambda: fake_coop), \
                mock.patch.object(qo, "_CACHED_ORDER", "wxyz"):
            a2 = s5.object_zaxis_in_root(types.SimpleNamespace(scene=scene("wxyz")),
                                         robot_cfg=types.SimpleNamespace(name="robot_a"))
        self.assertTrue(torch.allclose(a, a2, atol=1e-6))

    def test_signature(self):
        sig = inspect.signature(s5.object_zaxis_in_root).parameters
        self.assertEqual(list(sig), ["env", "robot_cfg", "object_name"])
        self.assertIs(sig["robot_cfg"].default, inspect.Parameter.empty)
        self.assertEqual(sig["object_name"].default, "object")


# ------------------------------------------------------ (ii) the lift curriculum


class _FakeRewardManager:
    def __init__(self, minimal_height=0.04):
        self.cfg = types.SimpleNamespace(params={"minimal_height": minimal_height})
        self.set_calls = 0

    def get_term_cfg(self, name):
        assert name == "lifting_object", name
        return self.cfg

    def set_term_cfg(self, name, cfg):
        assert name == "lifting_object", name
        self.cfg = cfg
        self.set_calls += 1


def _curr_env(poses, order="xyzw", level=None, pinch_d=None):
    """Env with one cube per (p_local, q_wxyz); env origins at 0 (world = local)."""
    p = torch.tensor([pp for pp, _ in poses], dtype=torch.float32)
    q = torch.tensor([qo.reorder(qq, order) for _, qq in poses], dtype=torch.float32)
    obj = types.SimpleNamespace(data=types.SimpleNamespace(root_pos_w=p, root_quat_w=q,
                                                           root_lin_vel_w=torch.zeros_like(p)))
    env = types.SimpleNamespace(scene={"object": obj}, num_envs=len(poses), device="cpu",
                                cfg=types.SimpleNamespace(object_spawn_z=0.55, clock_lift_height=False),
                                reward_manager=_FakeRewardManager())
    if level is not None:
        env._bhl_lift_h = level
    if pinch_d is not None:
        env._bhl_pinch_d = torch.full((len(poses),), float(pinch_d))
    return env


FLAT_LIFTED = ((0.0, 0.0, sm.PLINTH_TOP + H + 0.03), IDENT)               # corner 3 cm clear
FLAT_ON_PLINTH = ((0.0, 0.0, sm.PLINTH_TOP + H), IDENT)
EDGE_ON_PLINTH = ((0.0, 0.0, sm.PLINTH_TOP + H * S2), q_axis((0, 1, 0), 45.0))   # centre +0.058


def _stand4_rule_competence(env, ids, level):
    """Stand4's (Stand3's) promotion test, restated: centre > spawn + level AND pinch < 0.20."""
    z = env.scene["object"].data.root_pos_w[:, 2]
    ok = (z > env.cfg.object_spawn_z + level) & (env._bhl_pinch_d < 0.20)
    return float(ok[ids].float().mean())


class CurriculumTests(unittest.TestCase):
    PARAMS = {k: v for k, v in s5.CURRICULUM5_PARAMS.items()}

    def setUp(self):
        self.patches = [mock.patch.object(s5.sm, "_coop", lambda: types.SimpleNamespace(_t=lambda v: v)),
                        mock.patch.object(s5.sm, "_v2", lambda: types.SimpleNamespace(
                            _obj_local=lambda env, name="object": env.scene[name].data.root_pos_w)),
                        mock.patch.object(qo, "_CACHED_ORDER", "xyzw")]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def run_term(self, env, ids=None, **kw):
        prm = dict(self.PARAMS, **kw)
        return s5.stand5_lift_height_curriculum(env, ids, **prm)

    def test_promotes_holds_and_demotes_on_the_roll_proof_share(self):
        for k, want in ((0, 0.04), (2, 0.04), (3, 0.06), (5, 0.06), (8, 0.06)):
            # k of 8 envs roll-proof: share 0 (demote, floored), .25 (hold), .375 (> .35: promote), .625, 1
            env = _curr_env([FLAT_LIFTED] * k + [FLAT_ON_PLINTH] * (8 - k), level=0.04)
            got = self.run_term(env, torch.arange(8))
            self.assertEqual(got, s5.curriculum_decision(0.04, k / 8, 0.02, 0.04, 0.06, 0.35))
            self.assertAlmostEqual(got, want)
            self.assertEqual(env.reward_manager.cfg.params["minimal_height"], got)   # written
            self.assertEqual(env._bhl_lift_h, got)
            self.assertEqual(env.reward_manager.set_calls, 1)
        # demotion from the cap below half the target (0.175), hold between
        for k, want in ((1, 0.04), (2, 0.06), (3, 0.06)):       # .125 demotes, .25 holds, .375 holds at the cap
            env = _curr_env([FLAT_LIFTED] * k + [FLAT_ON_PLINTH] * (8 - k), level=0.06)
            self.assertAlmostEqual(self.run_term(env, torch.arange(8)), want)
        # no level yet: starts at min_height, as coop's
        env = _curr_env([FLAT_LIFTED] * 4)
        self.assertAlmostEqual(self.run_term(env, torch.arange(4)), 0.06)

    def test_clearance_boundary_2cm(self):
        base = sm.PLINTH_TOP + H
        for dz, want in ((0.0199, 0.04), (0.0201, 0.06)):
            env = _curr_env([((0.0, 0.0, base + dz), IDENT)], level=0.04)
            self.assertAlmostEqual(self.run_term(env, torch.tensor([0])), want, msg=dz)
        # the boundary moves with the parameter (it is read, not hard-coded)
        env = _curr_env([((0.0, 0.0, base + 0.0201), IDENT)], level=0.04)
        self.assertAlmostEqual(self.run_term(env, torch.tensor([0]), clearance=0.03), 0.04)

    def test_tilt_boundary_15deg(self):
        for deg, want in ((14.9, 0.06), (15.1, 0.04)):
            for axis in ((1, 0, 0), (0, 1, 0), (1, 1, 0)):
                env = _curr_env([((0.0, 0.0, 0.80), q_axis(axis, deg))], level=0.04)
                self.assertAlmostEqual(self.run_term(env, torch.tensor([0])), want, msg=(deg, axis))
        env = _curr_env([((0.0, 0.0, 0.80), q_axis((1, 0, 0), 14.9))], level=0.04)
        self.assertAlmostEqual(self.run_term(env, torch.tensor([0]), tilt_max_deg=10.0), 0.04)
        # a pure yaw is not a tilt
        env = _curr_env([((0.0, 0.0, 0.80), q_axis((0, 0, 1), 60.0))], level=0.04)
        self.assertAlmostEqual(self.run_term(env, torch.tensor([0])), 0.06)

    def test_supports_under_the_footprint(self):
        seated = sm.DECK_TOP + H
        cases = [((sm.DECK_CENTER, 0.0, seated + 0.03), 0.06),   # 3 cm over the deck top
                 ((sm.DECK_CENTER, 0.0, seated + 0.01), 0.04),   # 1 cm: not clear of the deck
                 ((1.0, 1.0, 0.20), 0.06)]                       # off every raised support: the floor
        for p, want in cases:
            env = _curr_env([(p, IDENT)], level=0.04)
            self.assertAlmostEqual(self.run_term(env, torch.tensor([0])), want, msg=p)

    def test_ignores_centre_height_and_pinch(self):
        # rolled onto an edge on the plinth, pinched: Stand4's test promotes, Stand5's does not
        env = _curr_env([EDGE_ON_PLINTH] * 4, level=0.04, pinch_d=0.0)
        ids = torch.arange(4)
        self.assertEqual(_stand4_rule_competence(env, ids, 0.04), 1.0)
        self.assertAlmostEqual(self.run_term(env, ids), 0.04)
        # lifted clear and flat, hands far away: Stand4's test does not promote, Stand5's does
        env = _curr_env([FLAT_LIFTED] * 4, level=0.04, pinch_d=1.0)
        self.assertEqual(_stand4_rule_competence(env, ids, 0.04), 0.0)
        self.assertAlmostEqual(self.run_term(env, ids), 0.06)
        src = inspect.getsource(s5.stand5_lift_height_curriculum)
        body = src[src.index('"""', src.index('"""') + 3) + 3:]
        self.assertNotIn("_bhl_pinch_d", body)
        self.assertNotIn("object_spawn_z", body)

    def test_env_ids_subset_int32_and_empty(self):
        env = _curr_env([FLAT_LIFTED] * 2 + [FLAT_ON_PLINTH] * 6, level=0.04)
        self.assertAlmostEqual(self.run_term(env, torch.tensor([0, 1, 2], dtype=torch.int32)), 0.06)  # 2/3
        env = _curr_env([FLAT_LIFTED] * 2 + [FLAT_ON_PLINTH] * 6, level=0.04)
        self.assertAlmostEqual(self.run_term(env, torch.tensor([2, 3, 4, 5], dtype=torch.int32)), 0.04)
        for ids in (None, torch.tensor([], dtype=torch.int32), []):       # all envs, as coop's
            env = _curr_env([FLAT_LIFTED] * 2 + [FLAT_ON_PLINTH] * 6, level=0.06)
            self.assertAlmostEqual(self.run_term(env, ids), 0.06)          # .25: hold

    def test_clock_branch_delegates_to_the_original(self):
        calls = []
        fake = types.SimpleNamespace(_t=lambda v: v, lift_height_curriculum=lambda *a, **k: (
            calls.append((a, k)), 0.123)[1])
        env = _curr_env([FLAT_LIFTED])
        env.cfg.clock_lift_height = True
        with mock.patch.object(s5.sm, "_coop", lambda: fake):
            self.assertEqual(self.run_term(env, torch.tensor([0])), 0.123)
        self.assertEqual(calls[0][1], {k: v for k, v in self.PARAMS.items()
                                       if k not in ("clearance", "tilt_max_deg")})

    def test_mirrors_coop_line_for_line(self):
        coop = COOP_MDP.read_text()
        start = coop.index("def lift_height_curriculum(")
        coop_fn = coop[start:coop.index("\ndef ", start + 10)]
        ours = inspect.getsource(s5.stand5_lift_height_curriculum)
        tail = coop_fn[coop_fn.index("    if success > success_rate_target:"):].rstrip()
        self.assertIn(tail, ours)
        self.assertIn('    height = getattr(env, "_bhl_lift_h", min_height)', ours)
        self.assertIn("    if env_ids is None or len(env_ids) == 0:\n        idx = slice(None)\n"
                      "    else:\n        idx = env_ids", ours)
        self.assertIn('if getattr(env.cfg, "clock_lift_height", False):', ours)
        sig4 = inspect.signature(ast_func_defaults(coop_fn))
        sig5 = inspect.signature(s5.stand5_lift_height_curriculum)
        for name, p in sig4.parameters.items():
            self.assertEqual(sig5.parameters[name].default, p.default, name)
        self.assertEqual(list(sig5.parameters)[-2:], ["clearance", "tilt_max_deg"])
        self.assertEqual((sig5.parameters["clearance"].default, sig5.parameters["tilt_max_deg"].default),
                         (0.02, 15.0))

    def test_decision_boundaries(self):
        d = lambda c, lvl=0.04: s5.curriculum_decision(lvl, c, 0.02, 0.04, 0.06, 0.35)  # noqa: E731
        self.assertEqual((d(0.35), d(0.3501), d(0.175), d(0.1749, 0.06), d(math.nan, 0.06)),
                         (0.04, 0.06, 0.04, 0.04, 0.06))
        self.assertEqual(d(1.0, 0.06), 0.06)                         # capped
        self.assertEqual(d(0.0, 0.04), 0.04)                         # floored


def ast_func_defaults(fn_src: str):
    """A stub with the same parameters and defaults as the function source (no body)."""
    node = ast.parse(fn_src).body[0]
    node.body = [ast.Pass()]
    node.returns = None
    for a in node.args.args:
        a.annotation = None
    mod = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    code = compile(mod, "<sig>", "exec")
    ns: dict = {}
    exec(code, ns)                                                   # noqa: S102 -- a parsed stub
    return ns[node.name]


# ------------------------------------------------------------------- rules


def _sc(last, success=0.0, time_out=0.6, value=10.0, nan_success_at=None, ep_len=400.0):
    it = range(0, last + 1)
    succ = [(i, math.nan if i == nan_success_at else success) for i in it]
    out = {sm.TAG_LEN: [(i, ep_len) for i in it], sm.TAG_TIMEOUT: [(i, time_out) for i in it],
           sm.TAG_SUCCESS: succ, sm.TAG_FALLEN: [(i, 1 - time_out) for i in it],
           sm.TAG_VALUE: [(i, value) for i in it], s4.TAG_LIFT_LEVEL: [(i, 0.04) for i in it],
           s4.TAG_GATE: [(i, 0.9) for i in it]}
    for t in s4.DIAG_TAGS:
        out[t] = [(i, 0.1) for i in it]
    return out


class RuleTests(unittest.TestCase):
    def test_kill_is_stand3s(self):
        for args in ((70.0, 0.0, 0.9), (70.0, 0.05, 0.0), (326.1, 0.02, 0.05), (150.0, 0.2, 0.09),
                     (100.0, 0.10, 0.0), (99.9, 0.5, 0.0999), (99.9, 0.0999, 0.0), (math.nan,) * 3):
            self.assertEqual(json.dumps(s5.kill_verdict(*args), sort_keys=True),
                             json.dumps(sm.kill_verdict3(*args), sort_keys=True), args)
        for sc in (_sc(1000, success=0.0, time_out=0.05), _sc(1000, success=0.5, ep_len=50.0),
                   _sc(1000, time_out=0.3, ep_len=99.0), _sc(1000, time_out=0.3)):
            self.assertEqual(json.dumps(s5.evaluate_kill(sc), sort_keys=True),
                             json.dumps(sm.evaluate_kill3(sc), sort_keys=True))
        self.assertEqual(s5.kill_verdict(99.9, 0.5, 0.0)["verdict"], "KILL")        # length
        self.assertEqual(s5.kill_verdict(100.0, 0.5, 0.0)["verdict"], "CONTINUE")
        self.assertEqual(s5.kill_verdict(400.0, 0.0999, 0.0)["verdict"], "KILL")     # non-fall
        self.assertEqual(s5.kill_verdict(400.0, 0.05, 0.05)["verdict"], "CONTINUE")
        self.assertEqual(s5.kill_verdict(50.0, 0.0, 0.10)["verdict"], "CONTINUE")    # placing: length skipped

    def test_seed_boundaries(self):
        r = s5.seed_result5
        self.assertFalse(r(7998, 0.5)["complete"] or r(7998, 0.5)["decided"])
        self.assertTrue(r(7999, 0.5)["complete"] and r(7999, 0.5)["passes"])
        self.assertFalse(r(7999, 0.0999)["passes"])
        self.assertTrue(r(7999, 0.10)["passes"])
        self.assertFalse(r(7999, math.nan)["passes"])
        self.assertTrue(r(1016, 0.9, killed=True)["decided"] and not r(1016, 0.9, killed=True)["passes"])
        b = r(7999, 0.9, bad_config=True)
        self.assertFalse(b["complete"] or b["decided"] or b["passes"] or b["killed"])
        for kw in ({"window_finite": False}, {"has_value": False}):
            self.assertFalse(r(7999, 0.2, **kw)["decided"])
        for args in ((7998, 0.5), (7999, 0.0999), (7999, 0.10), (8000, 0.3), (1016, 0.0)):
            for kw in ({}, {"killed": True}, {"bad_config": True}, {"window_finite": False}):
                self.assertEqual(r(*args, **kw), s4.seed_result4(*args, **kw))

    def test_evaluate_seed_is_stand4s(self):
        cases = [_sc(7999, success=0.12), _sc(7999, success=0.0999), _sc(7999, success=0.10),
                 _sc(7999, success=0.2, nan_success_at=7990), _sc(4468, success=0.5), _sc(7998, success=0.5)]
        for sc in cases:
            for kw in ({}, {"killed": True}, {"bad_config": True}):
                self.assertEqual(json.dumps(s5.evaluate_seed5(sc, **kw), default=str),
                                 json.dumps(s4.evaluate_seed4(sc, **kw), default=str))
        r = s5.evaluate_seed5(_sc(7999, success=0.12))
        self.assertTrue(r["complete"] and r["passes"])
        # the window mean against 0.10, with exact binary fractions (7/64 passes, 3/32 does not);
        # values before the window never count
        for val, want in ((7 / 64, True), (3 / 32, False)):
            sc = _sc(7999, success=0.5 if not want else 0.0)
            for i in range(7800, 8000):
                sc[sm.TAG_SUCCESS][i] = (i, val)
            r = s5.evaluate_seed5(sc)
            self.assertEqual(r["success"], val)
            self.assertEqual(r["passes"], want, val)

    def test_pair_verdict_is_stand4s_relabelled(self):
        R = s4.seed_result4
        states = {"pass": R(7999, 0.2), "pass_edge": R(7999, 0.10), "neg": R(7999, 0.0999),
                  "inc": R(4468, 0.9), "kil": R(1016, 0.0, killed=True), "kil_hi": R(1016, 0.9, killed=True),
                  "bad": R(7999, 0.9, bad_config=True), "nanwin": R(7999, 0.2, window_finite=False)}
        for a in states:
            for b in states:
                v4, v5 = s4.pair_verdict4([states[a], states[b]]), s5.pair_verdict5([states[a], states[b]])
                self.assertEqual(v5, v4.replace(s4.PASS_LABEL, s5.PASS_LABEL5), (a, b))
                self.assertEqual(v5.split(" ")[0], v4.split(" ")[0], (a, b))
        V = s5.pair_verdict5
        self.assertEqual(V([states["pass"], states["inc"]]), f"{s5.PASS_LABEL5} (1/2 seeds)")
        self.assertEqual(V([states["pass_edge"], states["pass"]]), f"{s5.PASS_LABEL5} (2/2 seeds)")
        self.assertTrue(V([states["neg"], states["neg"]]).startswith("NEGATIVE (0/2"))
        self.assertTrue(V([states["kil_hi"], states["neg"]]).startswith("NEGATIVE"))
        self.assertIn("1 killed", V([states["kil"], states["neg"]]))
        for x in ("inc", "bad", "nanwin"):
            self.assertTrue(V([states["neg"], states[x]]).startswith("INCOMPLETE"), x)


# ------------------------------------------------- config check and dumps


@unittest.skipUnless(HAVE_REF, "Stand4's reference smoke 21506757 is not on disk")
class ConfigAndDumpTests(unittest.TestCase):
    env4 = REF_ENV.read_text() if HAVE_REF else ""
    agent4 = REF_AGENT.read_text() if HAVE_REF else ""

    def test_synthetic_stand5_dump_passes(self):
        e5, a5 = stand5_env_yaml(self.env4), stand5_agent_yaml(self.agent4)
        r = s5.config_check5(a5, e5)
        self.assertTrue(r["ok"], r["failed"])
        self.assertTrue({"obs_policy_stand4_terms_then_zaxis_a_b", "obs_critic_stand4_terms_then_zaxis_a_b",
                         "lift_curriculum_roll_proof"} <= set(r["checks"]))
        self.assertTrue(set(s4.config_check(a5, e5)["checks"]) < set(r["checks"]))   # Stand4's + Stand5's

    def test_stand4_dump_fails_exactly_the_stand5_clauses(self):
        r = s5.config_check5(self.agent4, self.env4)
        self.assertEqual(set(r["failed"]), {"obs_policy_stand4_terms_then_zaxis_a_b",
                                            "obs_critic_stand4_terms_then_zaxis_a_b",
                                            "lift_curriculum_roll_proof"})

    def test_each_mutation_fails_its_clause(self):
        e5, a5 = stand5_env_yaml(self.env4), stand5_agent_yaml(self.agent4)
        pol, cri, cur = ("obs_policy_stand4_terms_then_zaxis_a_b", "obs_critic_stand4_terms_then_zaxis_a_b",
                         "lift_curriculum_roll_proof")
        P, C = ("observations", "policy"), ("observations", "critic")
        LH, LHP = ("curriculum", "lift_height"), ("curriculum", "lift_height", "params")
        old_func = "bhl_robust.tasks.coop_lift_mdp:object_pos_in_root"
        cases = [
            (edit_block(e5, P + ("object_zaxis_a",), s5.FUNC_OBS5, old_func), pol),
            (edit_block(e5, C + ("object_zaxis_b",), s5.FUNC_OBS5, old_func), cri),
            (drop_block(e5, C + ("object_zaxis_b",)), cri),
            (edit_block(e5, P, "    object_zaxis_a:", "    object_zaxis_q:"), pol),
            (edit_block(e5, P + ("object_zaxis_a",), "name: robot_a", "name: robot_b"), pol),
            (edit_block(e5, C + ("object_zaxis_b",), "      noise: null",
                        "      noise:\n        func: isaaclab.utils.noise.noise_model:uniform_noise"), cri),
            (edit_block(e5, C + ("object_zaxis_a",), "      clip: !!python/tuple\n      - -100.0\n      - 100.0",
                        "      clip: null"), cri),
            (edit_block(e5, P + ("object_zaxis_b",), "      scale: null", "      scale: 2.0"), pol),
            (edit_block(e5, LHP, "clearance: 0.02", "clearance: 0.03"), cur),
            (edit_block(e5, LHP, "tilt_max_deg: 15.0", "tilt_max_deg: 20.0"), cur),
            (edit_block(e5, LHP, "step: 0.02", "step: 0.04"), cur),
            (edit_block(e5, LHP, "success_rate_target: 0.35", "success_rate_target: 0.30"), cur),
            (edit_block(e5, LH, f"func: {s5.FUNC_CURR5}", f"func: {s5.FUNC_CURR4}"), cur),
            (drop_block(e5, ("curriculum", "lift_height", "params")), cur),
        ]
        for k, (text, clause) in enumerate(cases):
            self.assertNotEqual(text, e5, k)
            r = s5.config_check5(a5, text)
            self.assertIn(clause, r["failed"], (k, clause, r["failed"]))
        # the zaxis terms before `actions`: Stand4's terms must come first, in order
        lines = e5.splitlines()
        a, b = _span(lines, P + ("object_zaxis_a",))
        blk = lines[a:b]
        del lines[a:b]
        x, _ = _span(lines, P + ("actions",))
        lines[x:x] = blk
        self.assertIn(pol, s5.config_check5(a5, "\n".join(lines) + "\n")["failed"])
        # and the Stand4 clauses still bite (one example each side)
        r = s5.config_check5(a5, edit_block(e5, ("rewards", "lifting_object"), "weight: 15.0", "weight: 30.0"))
        self.assertEqual(r["failed"], ["lifting_object_stand4_x15"])
        r = s5.config_check5(a5.replace("entropy_coef: 0.001", "entropy_coef: 0.005"), e5)
        self.assertEqual(r["failed"], ["agent_entropy_coef_0.001"])

    def test_dump_diff_declared_keys_only(self):
        e5, a5 = stand5_env_yaml(self.env4), stand5_agent_yaml(self.agent4)
        d = s5.dump_diff(e5, self.env4, a5, self.agent4)
        self.assertTrue(d["ok"], d)
        self.assertEqual(d["agent"]["keys"], ["run_name"])
        self.assertTrue(all(any(k == e or k.startswith(e + ".") for e in s5.CFG_DIFF_ALLOWED5)
                            for k in d["env"]["keys"]))
        # anything else that differs is caught; a declared change that is absent is caught
        extra = edit_block(e5, ("rewards", "lifting_object"), "weight: 15.0", "weight: 30.0")
        d = s5.dump_diff(extra, self.env4, a5, self.agent4)
        self.assertFalse(d["ok"])
        self.assertEqual(d["env"]["unexpected"], ["rewards.lifting_object.weight"])
        d = s5.dump_diff(edit_block(e5, ("curriculum", "lift_height", "params"), "      clearance: 0.02\n", ""),
                         self.env4, a5, self.agent4)
        self.assertEqual(d["env"]["missing"], ["curriculum.lift_height.params.clearance"])
        d = s5.dump_diff(e5, self.env4, a5.replace("seed: 100", "seed: 0"), self.agent4)
        self.assertEqual(d["agent"]["unexpected"], ["seed"])
        d = s5.dump_diff(drop_block(e5, ("observations", "critic", "object_zaxis_b")), self.env4, a5, self.agent4)
        self.assertEqual(d["env"]["missing"], ["observations.critic.object_zaxis_b"])

    @unittest.skipUnless((OTHER_STAND4_SMOKE / "env.yaml").is_file(), "Stand4 smoke 21506555 params absent")
    def test_loader_on_two_real_stand4_dumps(self):
        """Same hashed sources (smokes 21506555 and 21506757): nothing but run_name differs."""
        e2, a2 = (OTHER_STAND4_SMOKE / "env.yaml").read_text(), (OTHER_STAND4_SMOKE / "agent.yaml").read_text()
        self.assertEqual(s5.diff_keys(s5.load_dump(e2), s5.load_dump(self.env4)), [])
        self.assertEqual(s5.diff_keys(s5.load_dump(a2), s5.load_dump(self.agent4)), ["run_name"])
        d = s5.load_dump(self.env4)
        self.assertEqual(d["observations"]["policy"]["object_pos_a"]["clip"], {"!python/tuple": [-100.0, 100.0]})
        self.assertIn("!python/object/apply:builtins.slice",
                      d["observations"]["policy"]["projected_gravity_a"]["params"]["asset_cfg"]["joint_ids"])
        self.assertEqual(s5._group_terms(d["observations"]["policy"]), list(s5.STAND4_OBS_TERMS["policy"]))
        self.assertEqual(s5._group_terms(d["observations"]["critic"]), list(s5.STAND4_OBS_TERMS["critic"]))

    def test_real_stand5_smoke_dump(self):
        """The first Stand5 smoke's dumped params (when on disk): the synthetic fixture used above
        equals it key for key, and the real dump passes config_check5 and the diff against Stand4's."""
        runs = sorted((REPO / "external/Berkeley-Humanoid-Lite/logs/rsl_rl/task_v2").glob(
            "*_v2-cubetoshelfstand5-blind-s100-j*-smoke"))
        runs = [r for r in runs if (r / "params/env.yaml").is_file() and (r / "params/agent.yaml").is_file()]
        if not runs:
            self.skipTest("no Stand5 smoke run on disk")
        e5, a5 = (runs[0] / "params/env.yaml").read_text(), (runs[0] / "params/agent.yaml").read_text()
        self.assertEqual(s5.diff_keys(s5.load_dump(stand5_env_yaml(self.env4)), s5.load_dump(e5)), [])
        self.assertEqual(s5.diff_keys(s5.load_dump(stand5_agent_yaml(self.agent4)), s5.load_dump(a5)),
                         ["run_name"])
        self.assertTrue(s5.config_check5(a5, e5)["ok"])
        self.assertTrue(s5.dump_diff(e5, self.env4, a5, self.agent4)["ok"])

    def test_loader_constructs_nothing(self):
        text = "a: !!python/object/apply:os.system\n- echo hacked\nb: !!python/name:os.system ''\n"
        d = s5.load_dump(text)
        self.assertEqual(d["a"], {"!python/object/apply:os.system": ["echo hacked"]})
        self.assertEqual(d["b"], {"!python/name:os.system": ""})


# ----------------------------------------------------------- log tables


@unittest.skipUnless(HAVE_REF, "Stand4's reference smoke 21506757 is not on disk")
class LogTableTests(unittest.TestCase):
    log4 = REF_LOG.read_text(errors="replace") if HAVE_REF else ""

    def stand5_log(self, order_ok=True, mlp=(200, 212)):
        """Stand4's reference log with Stand5's tables and network widths written in."""
        text = self.log4
        for g, w in (("policy", 194), ("critic", 206)):
            old = f"Active Observation Terms in Group: '{g}' (shape: ({w},))"
            self.assertEqual(text.count(old), 1, old)
            text = text.replace(old, old.replace(f"({w},)", f"({w + 6},)"))
        names = ("object_zaxis_a", "object_zaxis_b") if order_ok else ("object_zaxis_b", "object_zaxis_a")
        for g, last, nxt in (("policy", "actions", 13), ("critic", "object_ang_vel", 17)):
            head = text.index(f"Active Observation Terms in Group: '{g}'")
            eol = text.index("\n", text.index(f"| {last} ", head))
            rows = "".join(f"\n|     {nxt + k}    | {n:<33} |    (3,)   |" for k, n in enumerate(names))
            text = text[:eol] + rows + text[eol:]
        text = text.replace("Linear(in_features=194,", f"Linear(in_features={mlp[0]},")
        return text.replace("Linear(in_features=206,", f"Linear(in_features={mlp[1]},")

    def test_stand4_reference_tables(self):
        t = s5.obs_tables(self.log4)
        self.assertEqual(t["policy"]["shape"], (194,))
        self.assertEqual(t["critic"]["shape"], (206,))
        self.assertEqual([n for n, _ in t["policy"]["terms"]], list(s5.STAND4_OBS_TERMS["policy"]))
        self.assertEqual([n for n, _ in t["critic"]["terms"]], list(s5.STAND4_OBS_TERMS["critic"]))
        self.assertEqual(sum(d[0] for _, d in t["policy"]["terms"]), 194)
        self.assertEqual(s5.mlp_in_features(self.log4), {"actor": 194, "critic": 206})

    def test_expected_and_check(self):
        t4 = s5.obs_tables(self.log4)
        want = s5.expected_stand5_tables(t4)
        self.assertEqual(want["policy"]["shape"], (200,))
        self.assertEqual(want["critic"]["shape"], (212,))
        self.assertEqual(want["policy"]["terms"][-2:], [("object_zaxis_a", (3,)), ("object_zaxis_b", (3,))])
        log5 = self.stand5_log()
        self.assertTrue(s5.tables_check(s5.obs_tables(log5), t4, s5.mlp_in_features(log5))["ok"],
                        s5.tables_check(s5.obs_tables(log5), t4, s5.mlp_in_features(log5)))
        self.assertFalse(s5.tables_check(s5.obs_tables(self.log4), t4)["ok"])          # Stand4's own
        bad = self.stand5_log(order_ok=False)
        self.assertFalse(s5.tables_check(s5.obs_tables(bad), t4)["ok"])
        bad = self.stand5_log(mlp=(194, 212))
        self.assertFalse(s5.tables_check(s5.obs_tables(bad), t4, s5.mlp_in_features(bad))["ok"])
        self.assertFalse(s5.tables_check({}, t4)["ok"])
        self.assertFalse(s5.tables_check(s5.obs_tables(log5), {})["ok"])


# ------------------------------------------------------ smoke stage evaluators


def _obs_rec(order="xyzw", robot_yaw=(90.0, -90.0), n=16, perturb=None):
    """What a correct probe records in the obs stage (float64 kernels on xyzw storage)."""
    cases = [s5.axis_case(i) for i in range(n)]
    cube = torch.tensor([qo.reorder(s5.AXIS_CASES[c], order) for c in cases], dtype=torch.float64)
    robots = {r: torch.tensor([qo.reorder(q_axis((0, 0, 1), y), order)] * n, dtype=torch.float64)
              for r, y in zip(("robot_a", "robot_b"), robot_yaw)}
    cw = qo.unpack_wxyz(cube, order=order)
    terms = {name: s5.zaxis_in_frame(qo.unpack_wxyz(robots[r], order=order), cw)
             for name, r in s5.OBS_TERM_ROBOTS}
    rec = {"cases": cases, "valid": [True] * n, "cube_quat_native": cube.tolist(),
           "robot_quat_native": {r: q.tolist() for r, q in robots.items()},
           "term_values": {k: v.tolist() for k, v in terms.items()},
           "isaac_ref_values": {k: v.tolist() for k, v in terms.items()},
           "world_zaxis_isaac": s5.body_zaxis_world(*cw).tolist(),
           "tail": {g: torch.cat([terms["object_zaxis_a"], terms["object_zaxis_b"]], -1).tolist()
                    for g in s5.OBS_GROUPS}}
    if perturb:
        perturb(rec)
    return rec


class ObsStageTests(unittest.TestCase):
    def test_pass(self):
        r = s5.obs_stage_eval(_obs_rec(), "xyzw")
        self.assertTrue(r["ok_tail"] and r["ok_axis"], r)
        self.assertGreater(r["order_control_min_dev"], 0.5)

    def test_failures(self):
        def tail_off(rec):
            rec["tail"]["critic"][3][4] += 1e-3
        def wrong_robot(rec):                                        # robot_b's numbers under a's name
            rec["term_values"]["object_zaxis_a"] = rec["term_values"]["object_zaxis_b"]
            rec["tail"] = {g: [a + b for a, b in zip(rec["term_values"]["object_zaxis_a"],
                                                     rec["term_values"]["object_zaxis_b"])]
                           for g in s5.OBS_GROUPS}
        def isaac_disagrees(rec):                                    # env 1 is x+90: not (0, 0, 1)
            self.assertEqual(rec["cases"][1], "x+90")
            rec["isaac_ref_values"]["object_zaxis_b"][1] = [0.0, 0.0, 1.0]
        def cube_moved(rec):                                         # orientation not what was written
            rec["cube_quat_native"][1] = list(qo.reorder(q_axis((1, 0, 0), 80.0), "xyzw"))
        def one_case_missing(rec):
            rec["valid"] = [c != "x180" for c in rec["cases"]]
        r = s5.obs_stage_eval(_obs_rec(perturb=tail_off), "xyzw")
        self.assertFalse(r["ok_tail"])
        self.assertTrue(r["ok_axis"])
        for p in (wrong_robot, isaac_disagrees, cube_moved, one_case_missing):
            self.assertFalse(s5.obs_stage_eval(_obs_rec(perturb=p), "xyzw")["ok_axis"], p.__name__)
        # a wxyz stack is not the v60 stack the smoke must run on
        self.assertFalse(s5.obs_stage_eval(_obs_rec(order="wxyz"), "wxyz")["ok_axis"])
        # the right records judged in the wrong order fail
        self.assertFalse(s5.obs_stage_eval(_obs_rec(), "wxyz")["ok_axis"])
        self.assertFalse(s5.obs_stage_eval({}, "xyzw")["ok_axis"])
        r = s5.obs_stage_eval({"cases": []}, "xyzw")
        self.assertFalse(r["ok_axis"] or r["ok_tail"])


def _curr_rec(perturb=None, n=16):
    """What a correct probe records in the curriculum stage (the term run on fakes)."""
    groups = [s5.curr_group(i) for i in range(n)]
    poses = [s5.curr_pose_local(i) for i in range(n)]
    p = torch.tensor([pp for pp, _ in poses], dtype=torch.float32)
    q = torch.tensor([qo.reorder(qq, "xyzw") for _, qq in poses], dtype=torch.float32)
    w, x, y, z = qo.unpack_wxyz(q, order="xyzw")
    prm = dict(s5.CURRICULUM5_PARAMS)
    roll = s4.roll_proof_mask(p, w, x, y, z, clearance=prm["clearance"], tilt_max_deg=prm["tilt_max_deg"])
    spawn, lo, hi = 0.55, prm["min_height"], prm["max_height"]
    centre = p[:, 2] > spawn + lo
    sel_roll = torch.nonzero(roll).squeeze(-1)
    sel_centre = torch.nonzero(~roll & centre).squeeze(-1)
    plan = [("c1", sel_roll, lo, 1.0), ("c2", sel_centre, lo, 0.0), ("c3", sel_centre, hi, 0.0),
            ("c4", torch.arange(n).int(), lo, 0.0)]
    calls = []
    upd = {k: prm[k] for k in ("step", "min_height", "max_height", "success_rate_target")}
    for name, sel, pre, pinch in plan:
        ids = sel.tolist()
        c5 = float(roll[sel].float().mean())
        c4 = float(((p[sel, 2] > spawn + pre) & (pinch < 0.20)).float().mean())
        r5, r4 = s5.curriculum_decision(pre, c5, **upd), s5.curriculum_decision(pre, c4, **upd)
        calls.append({"name": name, "ids": ids, "ids_dtype": str(sel.dtype), "pre": pre, "pinch_d": pinch,
                      "r5": r5, "minimal_height_after_r5": r5, "lift_h_after_r5": r5, "r4": r4})
    rec = {"groups": groups, "valid": [True] * n, "spawn_z": spawn, "params": prm,
           "p_local": p.double().tolist(), "quat_native": q.double().tolist(),
           "z_world": p[:, 2].tolist(), "roll_ok": roll.tolist(), "calls": calls}
    if perturb:
        perturb(rec)
    return rec


class CurriculumStageTests(unittest.TestCase):
    def test_poses_are_what_the_groups_say(self):
        for i in range(8):
            (x, y, z), q = s5.curr_pose_local(i)
            pt = torch.tensor([[x, y, z]], dtype=torch.float64)
            w_, x_, y_, z_ = cols([q])
            g = s5.curr_group(i)
            roll = bool(s4.roll_proof_mask(pt, w_, x_, y_, z_)[0])
            centre = z > 0.55 + 0.04
            self.assertEqual((roll, centre), {"clear_flat": (True, True), "clear_tilt20": (False, True),
                                              "edge_plinth": (False, True),
                                              "flat_plinth": (False, False)}[g], g)
        (x, y, z), q = s5.curr_pose_local(2)                           # resting on its edge
        self.assertAlmostEqual(float(s4.lowest_corner_z(torch.tensor([[x, y, z]], dtype=torch.float64),
                                                        *cols([q]))[0]), sm.PLINTH_TOP, places=9)

    def test_pass(self):
        r = s5.curriculum_stage_eval(_curr_rec(), "xyzw")
        self.assertTrue(r["ok"], r)
        self.assertTrue(r["promotes_on_roll_proof_only"] and r["ignores_centre_and_pinch"])
        c = {x["name"]: x for x in r["calls"]}
        self.assertEqual((c["c1"]["want5"], c["c1"]["want4"]), (0.06, 0.04))
        self.assertEqual((c["c2"]["want5"], c["c2"]["want4"]), (0.04, 0.06))
        self.assertEqual((c["c3"]["want5"], c["c3"]["want4"]), (0.04, 0.06))

    def test_failures(self):
        def stand4_rule_used(rec):                                     # the term promoted like Stand4's
            for c in rec["calls"]:
                c["r5"] = c["minimal_height_after_r5"] = c["lift_h_after_r5"] = c["r4"]
        def not_written(rec):
            rec["calls"][0]["minimal_height_after_r5"] = rec["calls"][0]["pre"]
        def flags_lie(rec):
            rec["roll_ok"][1] = True
        def order_misread(rec):                                       # the tilt-20 cube read as flat
            rec["quat_native"][1] = list(qo.reorder(IDENT, "xyzw"))
            rec["roll_ok"][1] = True
        def no_int32(rec):
            rec["calls"][3]["ids_dtype"] = "torch.int64"
        def no_discriminating_env(rec):
            rec["calls"][1]["ids"] = []
            rec["calls"][2]["ids"] = []
        def stand4_function_wrong(rec):
            rec["calls"][1]["r4"] = 0.04
        for p in (stand4_rule_used, not_written, flags_lie, order_misread, no_int32,
                  no_discriminating_env, stand4_function_wrong):
            self.assertFalse(s5.curriculum_stage_eval(_curr_rec(p), "xyzw")["ok"], p.__name__)
        self.assertFalse(s5.curriculum_stage_eval({}, "xyzw")["ok"])
        # an invalid (reset) env is not judged against its group
        def reset_env(rec):
            rec["valid"][1] = False
            rec["roll_ok"][1] = True
            rec["quat_native"][1] = list(qo.reorder(IDENT, "xyzw"))
            for c in rec["calls"]:
                c["ids"] = [i for i in c["ids"] if i != 1]
        self.assertTrue(s5.curriculum_stage_eval(_curr_rec(reset_env), "xyzw")["roll_ok_matches_design"])


def _probe_ok():
    exp = {"reward_terms": ["reaching_coarse", "lifting_object", "placed"],
           "termination_terms": ["time_out", "fallen", "success"],
           "curriculum_terms": ["lift_height", "stage_lift", "upright_gate"] + list(s4.DIAG_TERMS),
           "obs_terms": {g: list(s5.STAND4_OBS_TERMS[g]) for g in s5.OBS_GROUPS},
           "lift_height_func": s5.FUNC_CURR4, "lift_height_params": dict(s5.CURRICULUM4_PARAMS)}
    new = {g: {n: {"func": s5.FUNC_OBS5, "robot": r, "noise": "None", "scale": "None",
                   "clip": [-100.0, 100.0]} for n, r in s5.OBS_TERM_ROBOTS} for g in s5.OBS_GROUPS}
    dims4 = {"policy": [[3]] * 4 + [[22]] * 4 + [[3]] * 2 + [[22]] * 2 + [[44]]}
    dims4["critic"] = dims4["policy"] + [[3]] * 4
    m = {"reward_terms": exp["reward_terms"], "termination_terms": exp["termination_terms"],
         "curriculum_terms": exp["curriculum_terms"],
         "obs_terms": {g: list(s5.STAND4_OBS_TERMS[g]) + list(s5.OBS_TERMS5) for g in s5.OBS_GROUPS},
         "obs_term_dims": {g: dims4[g] + [[3], [3]] for g in s5.OBS_GROUPS},
         "obs_group_dims": {"policy": [200], "critic": [212]},
         "lifting_object": {"func": s4.FUNC_LIFT4, "weight": 15.0},
         "placed": {"func": s4.FUNC_PLACED, "weight": 5000.0}, "success": {"func": s4.FUNC_SUCCESS4},
         "lift_height": {"func": s5.FUNC_CURR5, "params": dict(s5.CURRICULUM5_PARAMS)}, "new_obs": new}
    keys = []
    for e in s5.CFG_DIFF_ALLOWED5:
        keys += [e + ".func", e + ".clip"] if e.startswith("observations") else [e]
    return {"status": "ok", "quat_order": "xyzw", "managers": m, "stand4_expected": exp,
            "cfg_diff_keys": sorted(keys), "obs": _obs_rec(), "curriculum": _curr_rec()}


def _ref_log_text():
    rows = {g: "\n".join(f"|     {i}     | {n:<33} |    ({d},)   |"
                         for i, (n, d) in enumerate(zip(s5.STAND4_OBS_TERMS[g], ([3] * 4 + [22] * 4 + [3] * 2
                                                                                 + [22] * 2 + [44] + [3] * 4))))
            for g in s5.OBS_GROUPS}
    out = []
    for g, w in (("policy", 194), ("critic", 206)):
        out += [f"| Active Observation Terms in Group: '{g}' (shape: ({w},)) |",
                "+-----------+-----------------------------------+-----------+",
                "|   Index   | Name                              |   Shape   |",
                "+-----------+-----------------------------------+-----------+", rows[g],
                "+-----------+-----------------------------------+-----------+"]
    return "\n".join(out) + "\n"


def _run_log_text(widths=(200, 212)):
    t = _ref_log_text().replace("(194,)", f"({widths[0]},)").replace("(206,)", f"({widths[1]},)")
    for g, last, nxt in (("policy", "actions", 13), ("critic", "object_ang_vel", 17)):
        head = t.index(f"Group: '{g}'")
        i = t.index(f"| {last}", head)
        eol = t.index("\n", i)
        add = "".join(f"\n|     {nxt + k}    | {n:<33} |    (3,)   |" for k, n in enumerate(s5.OBS_TERMS5))
        t = t[:eol] + add + t[eol:]
    log = t + "".join(f"\x1b[1m   Learning iteration {i}/20   \x1b[0m\n  Curriculum/roll_ok: 0.0100\n"
                      for i in range(20))
    return (log + f"Actor Model: MLPModel(\n  (mlp): MLP(\n    (0): Linear(in_features={widths[0]}, out_features=256, "
            f"bias=True)\nCritic Model: MLPModel(\n  (mlp): MLP(\n    (0): Linear(in_features={widths[1]}, "
            "out_features=256, bias=True)\ntrain.sh: reached 20 logged iterations\n")


class SmokeVerdictTests(unittest.TestCase):
    def v(self, probe=None, tags=None, log=None, cfg=None, diff=None, ref=None, sb=None, rec=None,
          mutate=None):
        probe = _probe_ok() if probe is None else probe
        if mutate:
            mutate(probe)
        return s5.smoke_verdict5(probe, set(s5.SMOKE_TAGS) if tags is None else tags,
                                 _run_log_text() if log is None else log,
                                 {"ok": True} if cfg is None else cfg, {"ok": True} if diff is None else diff,
                                 _ref_log_text() if ref is None else ref, {"ok": True} if sb is None else sb,
                                 {"ok": True} if rec is None else rec)

    def test_pass(self):
        r = self.v()
        self.assertEqual(r["verdict"], "PASS", r)
        self.assertEqual(set(r["clauses"]), set(s5.SMOKE5_CLAUSES))

    def test_each_clause_fails(self):
        def ldel(path, key):
            return lambda p: path(p).pop(key)
        cases = {
            "A1_managers": lambda p: p["managers"]["lift_height"].update(func=s5.FUNC_CURR4),
            "A2_cfg_diff_declared_only": lambda p: p["cfg_diff_keys"].append("rewards.lift_progress.weight"),
            "A3_obs_widths": lambda p: p["managers"]["obs_group_dims"].update(policy=[194]),
            "A4_obs_tail_is_the_terms": lambda p: p["obs"]["tail"]["policy"][0].__setitem__(0, 9.0),
            "A5_axis_order_xyzw": lambda p: p["obs"]["isaac_ref_values"]["object_zaxis_a"].__setitem__(
                2, [1.0, 0.0, 0.0]),
            "A6_curriculum_roll_proof": lambda p: p["curriculum"]["calls"][0].update(r5=0.04),
        }
        for clause, mutate in cases.items():
            self.assertEqual(self.v(mutate=mutate)["verdict"], f"FAIL ({clause})", clause)
        self.assertEqual(self.v(tags=set(s5.SMOKE_TAGS) - {"Curriculum/lift_height"})["verdict"],
                         "FAIL (B1_logged_tags)")
        self.assertEqual(self.v(log=_run_log_text().replace("Learning iteration 19/20", ""))["verdict"],
                         "FAIL (B2_training_ran)")
        self.assertEqual(self.v(log=_run_log_text() + "Traceback (most recent call last):\n")["verdict"],
                         "FAIL (B3_no_failure_markers)")
        self.assertEqual(self.v(cfg={"ok": False})["verdict"], "FAIL (B4_config_check)")
        self.assertEqual(self.v(diff={"ok": False})["verdict"], "FAIL (B5_dump_diff_vs_stand4_smoke)")
        self.assertEqual(self.v(log=_run_log_text((194, 206)))["verdict"], "FAIL (B6_train_obs_tables)")
        self.assertEqual(self.v(sb={"ok": False})["verdict"], "FAIL (C1_stand4_sources_identical)")
        self.assertEqual(self.v(rec={"ok": False})["verdict"], "FAIL (C2_config_recorded)")
        # a new term without the clip, with noise, or on the wrong robot
        for k, val in (("clip", None), ("noise", "UniformNoiseCfg()"), ("robot", "robot_b")):
            r = self.v(mutate=lambda p, k=k, val=val: p["managers"]["new_obs"]["critic"]["object_zaxis_a"]
                       .update({k: val}))
            self.assertEqual(r["verdict"], "FAIL (A1_managers)", k)
        # Stand4's expected term lists must themselves be Stand4's
        r = self.v(mutate=lambda p: p["stand4_expected"]["obs_terms"]["policy"].pop())
        self.assertIn("A1_managers", r["verdict"])

    def test_incomplete(self):
        self.assertTrue(s5.smoke_verdict5(None, set(s5.SMOKE_TAGS), _run_log_text(), {"ok": True}, {"ok": True},
                                          _ref_log_text(), {"ok": True}, {"ok": True})["verdict"]
                        .startswith("INCOMPLETE"))
        for kw in ({"mutate": lambda p: p.update(status="error")},):
            self.assertTrue(self.v(**kw)["verdict"].startswith("INCOMPLETE"))
        r = s5.smoke_verdict5(_probe_ok(), set(s5.SMOKE_TAGS), None, {"ok": True}, {"ok": True},
                              _ref_log_text(), {"ok": True}, {"ok": True})
        self.assertTrue(r["verdict"].startswith("INCOMPLETE"))
        for args in ((None, {"ok": True}, _ref_log_text(), {"ok": True}, {"ok": True}),
                     ({"ok": True}, None, _ref_log_text(), {"ok": True}, {"ok": True}),
                     ({"ok": True}, {"ok": True}, None, {"ok": True}, {"ok": True}),
                     ({"ok": True}, {"ok": True}, "no tables here", {"ok": True}, {"ok": True}),
                     ({"ok": True}, {"ok": True}, _ref_log_text(), None, {"ok": True}),
                     ({"ok": True}, {"ok": True}, _ref_log_text(), {"ok": True}, None)):
            r = s5.smoke_verdict5(_probe_ok(), set(s5.SMOKE_TAGS), _run_log_text(), *args)
            self.assertTrue(r["verdict"].startswith("INCOMPLETE"), (args, r["verdict"]))


# ----------------------------------------------------- source identity (Stand4)


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


class Stand4IdentityTests(unittest.TestCase):
    def test_task_v2_cfg_is_head_plus_the_stand5_block(self):
        data = TASK_V2_CFG.read_bytes()
        n, sha = TASK_V2_HEAD
        self.assertEqual(hashlib.sha256(data[:n]).hexdigest(), sha)
        rest = data[n:].decode()
        self.assertTrue(rest.startswith(s5.STAND5_MARKER + "\n"), rest[:80])
        self.assertTrue(rest.rstrip("\n").endswith(STAND5_END))
        self.assertEqual(rest.count("# --- stand5 ---"), 1)
        self.assertEqual(s5.task_v2_stand4_prefix(data), data[:n])
        base = _git_show(BASE, "src/bhl_robust/tasks/task_v2_env_cfg.py")
        if base is not None:
            self.assertEqual(data[:n], base)

    def test_stand4_sources_untouched(self):
        for rel, sha in PINNED_SHA.items():
            self.assertEqual(_sha(REPO / rel), sha, rel)

    @unittest.skipUnless(REF_SHA.is_file(), "Stand4 smoke 21506757's sha256 list is not on disk")
    def test_stand4_sources_check_against_the_reference_smoke(self):
        r = s5.stand4_sources_check(REPO, REF_SHA.read_text())
        self.assertTrue(r["ok"], r)
        self.assertEqual(len(r["checks"]), 6)
        with tempfile.TemporaryDirectory() as td:
            t = Path(td)
            for rel in list(s5.STAND4_REF_FILES) + [s5.TASK_V2_REL]:
                (t / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(REPO / rel, t / rel)
            self.assertTrue(s5.stand4_sources_check(t, REF_SHA.read_text())["ok"])
            p = t / "src/bhl_robust/tasks/stand_mdp.py"
            p.write_bytes(p.read_bytes() + b"\n")
            self.assertEqual(s5.stand4_sources_check(t, REF_SHA.read_text())["failed"],
                             ["src/bhl_robust/tasks/stand_mdp.py"])
            shutil.copyfile(REPO / "src/bhl_robust/tasks/stand_mdp.py", p)
            q = t / s5.TASK_V2_REL                                       # an edit above the marker
            q.write_bytes(q.read_bytes().replace(b"CUBE_STAND4_VARIANTS = ", b"CUBE_STAND4_VARIANTS  = ", 1))
            self.assertFalse(s5.stand4_sources_check(t, REF_SHA.read_text())["ok"])
            shutil.copyfile(REPO / s5.TASK_V2_REL, q)
            q.write_bytes(q.read_bytes().replace(s5.STAND5_MARKER.encode(), b"\n\n# --- stand6 ---"))
            self.assertFalse(s5.stand4_sources_check(t, REF_SHA.read_text())["ok"])      # no marker
            shutil.copyfile(REPO / s5.TASK_V2_REL, q)
            # unchanged since the smoke started (tasks/__init__.py excluded)
            (t / "src/bhl_robust/tasks/__init__.py").write_text("x = 1\n")
            start = subprocess.run(["sha256sum", s5.TASK_V2_REL, "src/bhl_robust/tasks/__init__.py"], cwd=t,
                                   capture_output=True, text=True, check=True).stdout
            self.assertTrue(s5.stand4_sources_check(t, REF_SHA.read_text(), start)["ok"])
            (t / "src/bhl_robust/tasks/__init__.py").write_text("x = 2\n")
            self.assertTrue(s5.stand4_sources_check(t, REF_SHA.read_text(), start)["ok"])
            q.write_bytes(q.read_bytes() + b"# later\n")
            r = s5.stand4_sources_check(t, REF_SHA.read_text(), start)
            self.assertEqual(r["changed_since_start"], [s5.TASK_V2_REL])
            self.assertFalse(r["ok"])

    def test_registration_is_one_appended_guarded_block(self):
        src = TASKS_INIT.read_text()
        mark, end = "# --- stand5 ---", "# --- end stand5 ---"
        self.assertEqual((src.count(mark), src.count(end)), (1, 1))
        block = src[src.index(mark):src.index(end)]
        self.assertEqual(block.count("gym.register("), 1)
        self.assertIn('id="TaskV2-BHL-CubeToShelfStand5-Blind-v0"', block)
        self.assertIn('"env_cfg_entry_point": task_v2_env_cfg.CUBE_STAND5_VARIANTS["blind"]', block)
        self.assertIn('"rsl_rl_cfg_entry_point": task_v2_env_cfg._STAND3_RUNNER', block)
        self.assertIn('entry_point="isaaclab.envs:ManagerBasedRLEnv"', block)
        self.assertTrue("try:" in block and "except Exception as _exc:" in block and "NOT registered" in block)
        self.assertNotIn("CubeToShelfStand4", block)                 # test_stand4 counts that name
        self.assertNotIn("PlateCross", block)                        # test_platecross
        self.assertNotIn("# --- ", block[len(mark):])
        self.assertNotIn("Stand5", src[:src.index(mark)])
        base = _git_show(BASE, "src/bhl_robust/tasks/__init__.py")
        if base is not None:
            self.assertTrue(src.encode().startswith(base))               # appended only
        ast.parse(src)


def _stand5_try_node():
    tree = ast.parse(TASK_V2_CFG.read_text())
    tries = [n for n in tree.body if isinstance(n, ast.Try)
             and any(isinstance(b, ast.ClassDef) and b.name == "CubeToShelfStand5Cfg" for b in n.body)]
    assert len(tries) == 1, len(tries)
    return tries[0]


class Stand5CfgClassTests(unittest.TestCase):
    def setUp(self):
        self.src = TASK_V2_CFG.read_text()
        self.block = self.src[self.src.index(s5.STAND5_MARKER):]
        self.node = _stand5_try_node()
        self.cls = next(b for b in self.node.body if isinstance(b, ast.ClassDef))
        self.cls_src = ast.get_source_segment(self.src, self.cls)

    def test_the_class_is_stand4_plus_the_two_changes(self):
        self.assertEqual([ast.unparse(b) for b in self.cls.bases], ["CubeToShelfStand4Cfg"])
        self.assertEqual([ast.unparse(d) for d in self.cls.decorator_list], ["configclass"])
        fns = [b for b in self.cls.body if isinstance(b, ast.FunctionDef)]
        self.assertEqual([f.name for f in fns], ["__post_init__"])
        body = fns[0].body
        self.assertEqual(ast.unparse(body[0]), "super().__post_init__()")
        # the only writes: setattr on the two observation groups, and curriculum.lift_height
        writes = [ast.unparse(t) for n in ast.walk(fns[0]) if isinstance(n, ast.Assign) for t in n.targets]
        self.assertEqual(writes, ["lh", "self.curriculum.lift_height"])
        calls = [n for n in ast.walk(fns[0]) if isinstance(n, ast.Call) and ast.unparse(n.func) == "setattr"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(ast.unparse(calls[0]), "setattr(grp, name, ObsTerm(func=s5.object_zaxis_in_root, "
                         "params={'robot_cfg': SceneEntityCfg(robot)}, clip=(-stand.OBS_CLIP, stand.OBS_CLIP)))")
        loops = [n for n in ast.walk(fns[0]) if isinstance(n, ast.For)]
        self.assertEqual([ast.unparse(lp.iter) for lp in loops],
                         ["(self.observations.policy, self.observations.critic)", "s5.OBS_TERM_ROBOTS"])
        lh = next(n for n in ast.walk(fns[0]) if isinstance(n, ast.Assign)
                  and ast.unparse(n.targets[0]) == "self.curriculum.lift_height")
        self.assertEqual(ast.unparse(lh.value),
                         "CurrTerm(func=s5.stand5_lift_height_curriculum, params={**lh.params, "
                         "'clearance': s4.CORNER_CLEARANCE, 'tilt_max_deg': s4.LIFT_TILT_MAX_DEG})")
        code = self.cls_src.split('"""')[2]
        for banned in ("self.rewards", "self.terminations", "self.events", "self.actions", "self.scene",
                       "self.sim", "episode_length_s", ".weight", "init_state", "noise=", "scale="):
            self.assertNotIn(banned, code, banned)
        self.assertIsNone(re.search(r"\br\.", code))

    def test_variants_guard_and_markers(self):
        self.assertTrue(self.block.rstrip("\n").endswith(STAND5_END))
        tail = [ast.unparse(b) for b in self.node.body if not isinstance(b, ast.ClassDef)]
        self.assertEqual(tail, ["from bhl_robust.tasks import stand5_mdp as s5",
                                "CUBE_STAND5_VARIANTS = _variants(CubeToShelfStand5Cfg, 'CubeToShelfStand5')"])
        self.assertEqual([ast.unparse(h.type) for h in self.node.handlers], ["Exception"])
        self.assertIn("NOT defined", ast.unparse(self.node.handlers[0]))
        self.assertEqual(len(self.node.orelse) + len(self.node.finalbody), 0)
        # what test_stand_mdp reads to the end of the file
        self.assertNotIn(".weight = ", self.block)
        self.assertNotIn("r.placed = ", self.block)
        self.assertNotIn("_V2_RUNNER = ", self.block)


# ----------------------------------------------------------------- launchers


class TrainLauncherTests(unittest.TestCase):
    def test_header_carries_the_frozen_design_and_rule_verbatim(self):
        head = _header(TRAIN5)
        for line in s5.DESIGN_S:
            self.assertIn(_norm(line), head, line)
        self.assertIn(_norm(s4.PREDECLARED_RULE), head)
        for s in ("COMPLETE iff its events reach iteration 7999 (8000 iterations) AND every Loss/value, "
                  "success and time_out value in its last 200 iterations is finite AND that window holds a "
                  "Loss/value value", "neither a PASS nor a NEGATIVE", "PASS: learned roll-proof placement (Stand5)",
                  "Precedence (Stand4's): PASS if any seed passes whatever the other did, then INCOMPLETE, then "
                  "NEGATIVE", "afterok on gpu_v2_stand5_smoke.sbatch", "--export=ALL,STAND5_SMOKE_JOB=<smoke>",
                  "BYTES-CHECK", "RUNNER-CHECK", "never from exit codes", "object_zaxis_a", "194 -> 200",
                  "206 -> 212", "instead of centre > spawn + level AND pinch distance < 0.20 m", "fresh runs",
                  "LEARNED crew policy (one PPO actor drives both robots, blind)", "now including cube orientation",
                  "MODIFIED hand colliders", "stand4_mdp.roll_proof_mask", "clipped +/-100",
                  "quat_order.unpack_wxyz", "Stand4's resources",
                  # the curriculum clause as applied
                  "the lift curriculum's competence is the share of the resetting envs whose cube's lowest "
                  "corner is >= 0.02 m above every support under it AND whose tilt is <= 15 deg "
                  "(stand4_mdp.roll_proof_mask); no centre-height and no pinch clause",
                  "a cube clear and flat in the air counts without a grasp"):
            self.assertIn(s, head, s)

    def test_kill_block_is_stand3s_and_stand4s(self):
        b3, b4, b5 = _kill_block(TRAIN3), _kill_block(TRAIN4), _kill_block(TRAIN5)
        self.assertEqual(len(b3), 9)
        self.assertEqual(b5, b3)
        self.assertEqual(b5, b4)
        body = TRAIN5.read_text()
        self.assertIn("s5.evaluate_kill(sc)", body)
        self.assertIn('res["ready"] = last >= s5.sm.KILL_AT_ITER', body)
        self.assertIn("KILL_AT=1000", body)

    def test_resources_and_flags_are_stand4s(self):
        f4, f5 = _sbatch_flags(TRAIN4), _sbatch_flags(TRAIN5)
        self.assertEqual(f5.pop("job-name"), "v2-stand5-train")
        f4.pop("job-name")
        self.assertEqual(f5, f4)
        self.assertEqual((f5["array"], f5["time"], f5["mem"], f5["gres"], f5["cpus-per-task"]),
                         ("0-1", "40:00:00", "64G", "gpu:1", "8"))

    def test_body_is_stand4s_with_the_stand5_names(self):
        """Stand4's launcher body, line for line, after the declared renames; the lines that
        differ are the BYTES list, the hashed list, the stand5_mdp calls and JSON fields only."""
        import difflib

        def body(p):
            return p.read_text().split("set -euo pipefail", 1)[1]
        renames = [("STAND4_SMOKE_JOB", "STAND5_SMOKE_JOB"), ("stand4_eval", "stand5_eval"),
                   ("stand4_2026-10-01", "stand5_2026-10-03"), ("CubeToShelfStand4", "CubeToShelfStand5"),
                   ("cubetoshelfstand4", "cubetoshelfstand5"), ("v2-stand4-", "v2-stand5-"),
                   ("STAND4 ", "STAND5 "), ("STAND4:", "STAND5:"), ("gpu_v2_stand4_smoke", "gpu_v2_stand5_smoke"),
                   ("slurm/repo20260923/inner_v2_stand4_smoke", "slurm/inner/inner_v2_stand5_smoke"),
                   ('"stand4_mdp"', '"stand5_mdp"'),
                   ("/src/bhl_robust/tasks/stand4_mdp.py", "/src/bhl_robust/tasks/stand5_mdp.py"),
                   ("s4 = importlib", "s5 = importlib"), ("spec.loader.exec_module(s4)", "spec.loader.exec_module(s5)"),
                   ("Stand4", "Stand5")]
        b4 = body(TRAIN4)
        for a, b in renames:
            b4 = b4.replace(a, b)
        diff = [ln for ln in difflib.unified_diff(b4.splitlines(), body(TRAIN5).splitlines(), lineterm="", n=0)
                if ln[:1] in "+-" and not ln.startswith(("+++", "---"))]
        removed = [ln[1:] for ln in diff if ln[0] == "-"]
        added = [ln[1:] for ln in diff if ln[0] == "+"]
        ok_line = re.compile(r"^(BYTES_REQUIRED=|echo \"=== node: |\( cd \"\$REPO\" && sha256sum |      (src|assets)/|# |"
                             r".*\bs[45]\.|.*stand[45]_mdp\b|                  f, indent=1\)$|"
                             r" +open\(os\.path\.join\(run, \"params\", \"env\.yaml\"\)\)\.read\(\)\)$)")
        self.assertEqual([ln for ln in removed + added if not ok_line.match(ln)], [], "\n".join(diff))
        self.assertLessEqual(len(diff), 60, "\n".join(diff))          # 50 at writing
        # every removed Stand4 call has its Stand5 counterpart
        for a, b in (("s4.evaluate_kill(sc)", "s5.evaluate_kill(sc)"),
                     ("s4.evaluate_seed4(sc, killed=killed, bad_config=bad)",
                      "s5.evaluate_seed5(sc, killed=killed, bad_config=bad)"),
                     ("s4.config_check(", "s5.config_check5("), ("s4.pair_verdict4(seeds)", "s5.pair_verdict5(seeds)")):
            self.assertTrue(any(a in ln for ln in removed), a)
            self.assertTrue(any(b in ln for ln in added), b)

    def test_verdicts_from_json_never_exit_codes_and_no_overwrite(self):
        body = TRAIN5.read_text()
        for s in ("s5.evaluate_seed5(sc, killed=killed, bad_config=bad)", "s5.pair_verdict5(seeds)",
                  "s5.config_check5(", 'open(out, "x")', "refusing to overwrite",
                  'result_s"${SEED}"_*.json', "refusing to start", 'jget "$js" decided',
                  "runner_check_s${SEED}_${JOBTAG}.json", 'mv "$tmp" "$js"',
                  'export TASK="TaskV2-BHL-CubeToShelfStand5-Blind-v0"',
                  "MAX_ITER=${MAX_ITER:-8000} NUM_ENVS=${NUM_ENVS:-1024}",
                  "export SEED=${SLURM_ARRAY_TASK_ID:-${SEED:-0}}", 'case "$SEED" in 0|1) ;;',
                  'export RUN_NAME="v2-cubetoshelfstand5-blind-s${SEED}"',
                  "OUTDIR=$REPO/results/repo-gpu-20260923/stand5_2026-10-03",
                  "SMOKE_OUT=$REPO/results/repo-gpu-20260923/stand5_2026-10-03-smoke", ': > "$OVERRIDE_FILE"',
                  'prior=$(ls "$OUTDIR"/result_s"${SEED}"_*.json 2>/dev/null || true)'):
            self.assertIn(s, body, s)
        self.assertIn('[ "$killed" = 1 ] && exit 0', body)

    def test_bytes_check_pinned_to_the_stand5_sources(self):
        body = TRAIN5.read_text()
        for s in ("check_smoke_bytes() {", 'check_smoke_bytes "${STAND5_SMOKE_JOB:-}"', "sha256sum -c --strict -",
                  'verdict_${job}.json', 'sha256_${job}.txt', '!= "PASS"', 'rm -f "$STAMP"',
                  "BYTES-CHECK $RUN_NAME: FAIL -- refusing to start"):
            self.assertIn(s, body, s)
        self.assertLess(body.index('check_smoke_bytes "${STAND5_SMOKE_JOB:-}"'), body.index("\nv60_boot_gate\n"))
        required = re.search(r'^BYTES_REQUIRED="([^"]+)"$', body, re.M).group(1).split()
        self.assertEqual(set(required), {
            "src/bhl_robust/tasks/stand5_mdp.py", "src/bhl_robust/tasks/stand4_mdp.py",
            "src/bhl_robust/tasks/task_v2_env_cfg.py", "src/bhl_robust/tasks/stand_mdp.py",
            "src/bhl_robust/tasks/coop_lift_mdp.py", "src/bhl_robust/tasks/coop_lift_env_cfg.py",
            "src/bhl_robust/quat_order.py", "assets/cloth/berkeley_humanoid_lite_hand_colliders.usda"})
        excluded = re.search(r"grep -v -E '  \(([^)]+)\)\$'", body).group(1).replace("\\.", ".").split("|")
        self.assertEqual(set(excluded), {"src/bhl_robust/tasks/__init__.py",
                                         "slurm/repo20260923/gpu_v2_stand5_smoke.sbatch",
                                         "slurm/inner/inner_v2_stand5_smoke.sh"})
        self.assertEqual(sorted(_smoke_hashed()), sorted(required + excluded))

    def test_bash_syntax_and_heredocs_compile(self):
        for p in (TRAIN5, SMOKE5, INNER5):
            r = subprocess.run(["bash", "-n", str(p)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, (p, r.stderr))
            text = p.read_text()
            docs = re.findall(r"<<'PY'\n(.*?)\nPY\n", text, re.S)
            self.assertGreaterEqual(len(docs), 1, p)
            for i, d in enumerate(docs):
                compile(d, f"{p.name}:heredoc{i}", "exec")


def _smoke_hashed() -> list:
    smoke = SMOKE5.read_text()
    cmd = smoke[smoke.index("sha256sum src/bhl_robust/tasks/stand5_mdp.py"):]
    cmd = cmd[:cmd.index('> "$HASHES"')]
    return [w for w in cmd.replace("\\", " ").replace(")", " ").split() if "/" in w]


class SmokeLauncherTests(unittest.TestCase):
    def test_flags(self):
        f = _sbatch_flags(SMOKE5)
        self.assertIn("smoke", f["job-name"])
        hh, mm, ss = (int(v) for v in f["time"].split(":"))
        self.assertLessEqual(hh * 3600 + mm * 60 + ss, 3600)
        f4 = _sbatch_flags(SMOKE4)
        f.pop("job-name"), f4.pop("job-name")
        self.assertEqual(f, f4)                                          # Stand4's smoke resources

    def test_outputs_seed_and_exit(self):
        body = SMOKE5.read_text()
        for s in ("STAND5_OUT=$REPO/results/repo-gpu-20260923/stand5_2026-10-03-smoke",
                  'RUN_NAME="v2-cubetoshelfstand5-blind-s${SEED}-j${JOB}-smoke"', "SEED=100",
                  'case "$SEED" in 0|1)', "NUM_ENVS=1024 MAX_ITER=20", "NUM_ENVS=16 PROBE_STEPS=30",
                  'sys.exit(0 if v["verdict"] == "PASS" else 1)', "exit $rc",
                  "s5.smoke_verdict5(probe, tags, train_log, cfg, diff, ref_log, stand4_bytes, recorded)",
                  "s5.config_check5(a, e)", "s5.dump_diff(e, ref_env, a, ref_agent)", "refusing to overwrite",
                  'bhl_exec "$REPO/slurm/inner/inner_v2_stand5_smoke.sh"', "shutil.copyfile(src, dst)",
                  "start_sha_text=open(hashes_p).read()", "export TASK=TaskV2-BHL-CubeToShelfStand5-Blind-v0",
                  'res["stand4_reference"]["files_sha256"]'):
            self.assertIn(s, body, s)
        head = _header(SMOKE5)
        for clause in s5.SMOKE5_CLAUSES:
            self.assertIn(clause.split("_")[0] + " ", head, clause)
        for s in ("SCRIPTED", "Never a result", "afterok", "21506757", "194 -> 200", "206 -> 212",
                  "(x, y, z, w)", "int32", "exported/recorded config"):
            self.assertIn(s, head, s)
        self.assertEqual(s5.SMOKE_ITERS, 20)

    def test_inner_probe(self):
        body = INNER5.read_text()
        for s in ("refusing to overwrite", "s5.diff_check(res[\"cfg_diff_keys\"], s5.CFG_DIFF_ALLOWED5)",
                  "write_root_pose_to_sim_index", "u.reset_buf", "json.dump(res", "STAND5-PROBE", "timeout 1500",
                  "matrix_from_quat", 'res["stage"] = "obs"', 'res["stage"] = "curriculum"',
                  'res["stage"] = "steps"', "coop.lift_height_curriculum(u, sel, **p4)",
                  "lh.func(u, sel, **lh.params)", "sel_all = torch.nonzero(valid).squeeze(-1).int()",
                  "om._group_obs_term_cfgs", "s5.object_zaxis_in_root(u, robot_cfg=", "quat_order()",
                  "native_quat(q) for q in quats_wxyz", "s5.TASK4_ID"):
            self.assertIn(s, body, s)
        i = body.index('res["stage"] = "curriculum"')
        self.assertLess(i, body.index("env.reset()", i))
        self.assertLess(body.index("env.reset()", i), body.index("put_cube([p for p, _ in poses]", i))
        self.assertLess(body.index("json.dump(res, f"), body.rindex("env.close()"))
        self.assertLess(body.index("json.dump(res, f"), body.rindex("simulation_app.close()"))
        self.assertIn('rm.set_term_cfg("lifting_object", t_cfg)', body)      # the probe restores


# -------------------------------------------- the train launcher on a fake tree

SLOW_SKIP = os.environ.get("STAND5_SKIP_SLOW") == "1"
FAKE_FILES = ("src/bhl_robust/__init__.py", "src/bhl_robust/quat_order.py",
              "src/bhl_robust/tasks/stand5_mdp.py", "src/bhl_robust/tasks/stand4_mdp.py",
              "src/bhl_robust/tasks/stand_mdp.py", "src/bhl_robust/tasks/task_v2_env_cfg.py",
              "src/bhl_robust/tasks/__init__.py", "src/bhl_robust/tasks/coop_lift_mdp.py",
              "src/bhl_robust/tasks/coop_lift_env_cfg.py",
              "assets/cloth/berkeley_humanoid_lite_hand_colliders.usda",
              "slurm/repo20260923/gpu_v2_stand5_smoke.sbatch", "slurm/inner/inner_v2_stand5_smoke.sh")
ENV_SOURCE = "source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh"

WRITE_EVENTS = r'''
import math, sys
from torch.utils.tensorboard import SummaryWriter
run, mode = sys.argv[1], sys.argv[2]
last = 1000 if mode == "kill" else 7999
base = {"kill": (50.0, 0.0, 0.0), "pass": (300.0, 0.5, 0.12), "neg": (400.0, 0.6, 0.0)}[mode]
w = SummaryWriter(run)
tags = ["Curriculum/" + n for n in ("lift_height", "upright_gate", "pinch_dist", "cube_tilt_deg", "corner_clear",
                                    "roll_ok", "lift_paid", "released", "robot_force_max")]
for i in range(last + 1):
    L, T, S = base
    for k, v in (("Train/mean_episode_length", L), ("Episode_Termination/time_out", T),
                 ("Episode_Termination/success", S), ("Episode_Termination/fallen", 1.0 - T - S),
                 ("Loss/value", 10.0)):
        w.add_scalar(k, v, i)
    for t in tags:
        w.add_scalar(t, 0.05, i)
w.close()
'''

FAKE_TRAIN = r'''#!/bin/bash
# stub for bhl_exec "$REPO/slurm/inner/train.sh": records what the launcher exported, writes a run
set -euo pipefail
W="$FAKE_W"
echo "TASK=$TASK SEED=$SEED MAX_ITER=$MAX_ITER NUM_ENVS=$NUM_ENVS RUN_NAME=$RUN_NAME BHL_STACK=$BHL_STACK ENABLE_CAMERAS=$ENABLE_CAMERAS OVERRIDE_SIZE=$(stat -c %s "$OVERRIDE_FILE") ARGS=$*" >> "$W/train_calls_s$SEED.txt"
run="$UPSTREAM/logs/rsl_rl/task_v2/$(date +%Y-%m-%d_%H-%M-%S)_${RUN_NAME}"
mkdir -p "$run/params"
cp "$W/fixtures/agent.yaml" "$W/fixtures/env.yaml" "$run/params/"
mode_var="FAKE_MODE_S${SEED}"; mode=${!mode_var:-neg}
"$W/venv/bin/python" "$W/write_events.py" "$run" "$mode"
if [ "$mode" = kill ]; then touch "$run/model_1000.pt"; sleep 900; fi
sleep_var="FAKE_SLEEP_S${SEED}"; sleep "${!sleep_var:-125}"
exit 0
'''


def _fake_tree(root: Path, smoke_job: str, smoke_verdict="PASS") -> Path:
    """A throw-away workspace: the launcher's inputs copied, stubs for the cluster bits."""
    W = root / "ws"
    R = W / "bhl-robustness-ladder"
    for rel in FAKE_FILES:
        (R / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / rel, R / rel)
    (R / "slurm/inner").mkdir(parents=True, exist_ok=True)
    (R / "slurm/inner/v60_boot_gate.sh").write_text('v60_boot_gate() { echo "v60_boot_gate: stub"; }\n')
    (R / "slurm/_env.sh").write_text(
        f'WORKSPACE={W}\nREPO=$WORKSPACE/bhl-robustness-ladder\nUPSTREAM=$REPO/external/Berkeley-Humanoid-Lite\n'
        f'PY={HOSTPY}\nBHL_FORWARD_VARS="TASK RUN_NAME SEED NUM_ENVS MAX_ITER OVERRIDE_FILE"\n'
        'setup_node_cache() { export TMPDIR=$WORKSPACE/tmp; mkdir -p "$TMPDIR"; }\n'
        'bhl_exec() { FAKE_W="$WORKSPACE" UPSTREAM="$UPSTREAM" REPO="$REPO" bash "$WORKSPACE/fake_train.sh" "$@"; }\n')
    text = TRAIN5.read_text()
    replacements = {
        ENV_SOURCE: f"source {R}/slurm/_env.sh",
        "export OV_CACHE=/scratch/$USER/ov-cache-${JOBTAG}":
            'export OV_CACHE="$WORKSPACE/cache/ov-cache-${JOBTAG}"',
        "export CUDA_CACHE_PATH=/scratch/$USER/nv-computecache-${JOBTAG}":
            'export CUDA_CACHE_PATH="$WORKSPACE/cache/nv-computecache-${JOBTAG}"',
    }
    for original, replacement in replacements.items():
        assert text.count(original) == 1
        text = text.replace(original, replacement)
    (R / "slurm/repo20260923/gpu_v2_stand5_train.sbatch").write_text(text)
    (W / "venv").symlink_to(HOSTPY.parent.parent)
    (W / "fake_train.sh").write_text(FAKE_TRAIN)
    (W / "write_events.py").write_text(WRITE_EVENTS)
    (W / "fixtures").mkdir()
    (W / "fixtures/env.yaml").write_text(stand5_env_yaml(REF_ENV.read_text()))
    (W / "fixtures/agent.yaml").write_text(stand5_agent_yaml(REF_AGENT.read_text()))
    sm_dir = R / "results/repo-gpu-20260923/stand5_2026-10-03-smoke"
    sm_dir.mkdir(parents=True)
    hashes = subprocess.run(["sha256sum"] + _smoke_hashed(), cwd=R, capture_output=True, text=True, check=True)
    (sm_dir / f"sha256_{smoke_job}.txt").write_text(hashes.stdout)
    (sm_dir / f"verdict_{smoke_job}.json").write_text(json.dumps({"verdict": smoke_verdict}))
    return W


def _launch(W: Path, array_job: str, task: str, smoke_job: str | None, **extra):
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(W), "USER": os.environ.get("USER", ""),
           "LANG": "C.UTF-8", "SLURM_JOB_ID": f"{array_job}{task}", "SLURM_ARRAY_JOB_ID": array_job,
           "SLURM_ARRAY_TASK_ID": task}
    if smoke_job is not None:
        env["STAND5_SMOKE_JOB"] = smoke_job
    env.update(extra)
    log = open(W / f"launcher_{array_job}_{task}.log", "w")
    p = subprocess.Popen(["bash", str(W / "bhl-robustness-ladder/slurm/repo20260923/gpu_v2_stand5_train.sbatch")],
                         env=env, stdout=log, stderr=subprocess.STDOUT, cwd=W)
    return p, W / f"launcher_{array_job}_{task}.log"


@unittest.skipUnless(HAVE_REF and HOSTPY.is_file(), "Stand4 reference dump or test interpreter missing")
class FakeTreeLauncherTests(unittest.TestCase):
    """The real launcher, real mode (no MAX_ITER / NUM_ENVS / SEED overrides), on a fake tree."""

    def setUp(self):
        self.td = tempfile.TemporaryDirectory(prefix="stand5-fake-")
        self.root = Path(self.td.name)

    def tearDown(self):
        self.td.cleanup()

    def out_dir(self, W):
        return W / "bhl-robustness-ladder/results/repo-gpu-20260923/stand5_2026-10-03"

    def run_fast(self, W, task, smoke, aj):
        p, log = _launch(W, aj, task, smoke)
        rc = p.wait(timeout=120)
        return rc, log.read_text()

    def test_refusals_before_anything_trains(self):
        W = _fake_tree(self.root, "777")
        out = self.out_dir(W)
        logroot = W / "bhl-robustness-ladder/external/Berkeley-Humanoid-Lite/logs/rsl_rl/task_v2"
        # no smoke job id
        rc, log = self.run_fast(W, "0", None, "9101")
        self.assertEqual(rc, 1, log)
        self.assertIn("BYTES-CHECK v2-cubetoshelfstand5-blind-s0: FAIL -- refusing to start", log)
        self.assertIn("STAND5_SMOKE_JOB is not set", (out / "bytes_check_s0_9101_0.txt").read_text())
        # a non-PASS smoke
        (W / "bhl-robustness-ladder/results/repo-gpu-20260923/stand5_2026-10-03-smoke/verdict_778.json").write_text(
            json.dumps({"verdict": "FAIL (A5_axis_order_xyzw)"}))
        shutil.copyfile(W / "bhl-robustness-ladder/results/repo-gpu-20260923/stand5_2026-10-03-smoke/sha256_777.txt",
                        W / "bhl-robustness-ladder/results/repo-gpu-20260923/stand5_2026-10-03-smoke/sha256_778.txt")
        rc, log = self.run_fast(W, "1", "778", "9102")
        self.assertEqual(rc, 1, log)
        self.assertIn("verdict is not PASS", log)
        # bytes changed after the smoke
        p = W / "bhl-robustness-ladder/src/bhl_robust/tasks/stand5_mdp.py"
        orig = p.read_bytes()
        p.write_bytes(orig + b"# edited after the smoke\n")
        rc, log = self.run_fast(W, "0", "777", "9103")
        self.assertEqual(rc, 1, log)
        self.assertIn("src/bhl_robust/tasks/stand5_mdp.py: FAILED", log)
        p.write_bytes(orig)
        # tasks/__init__.py may differ (shared, excluded) -- this one goes on to train, so stop at the seed check
        # instead: a seed that is not predeclared
        rc, log = self.run_fast(W, "2", "777", "9104")
        self.assertEqual(rc, 1, log)
        self.assertIn("seed 2 is not a predeclared Stand5 seed", log)
        # a result for the seed already exists
        out.mkdir(parents=True, exist_ok=True)
        (out / "result_s1_123.json").write_text("{}")
        rc, log = self.run_fast(W, "1", "777", "9105")
        self.assertEqual(rc, 1, log)
        self.assertIn("refusing to start -- a result for seed 1 exists", log)
        # a run dir with the run name already exists
        logroot.mkdir(parents=True, exist_ok=True)
        (logroot / "2026-01-01_00-00-00_v2-cubetoshelfstand5-blind-s0").mkdir()
        rc, log = self.run_fast(W, "0", "777", "9106")
        self.assertEqual(rc, 1, log)
        self.assertIn("refusing to start -- run dir(s) with this run name exist", log)
        # nothing ever trained, every stamp removed
        self.assertFalse(list(W.glob("train_calls_*.txt")))
        self.assertFalse(list(out.glob(".start-*")))

    @unittest.skipIf(SLOW_SKIP, "STAND5_SKIP_SLOW=1")
    def test_real_mode_array_pass_and_kill(self):
        """Two arrays at once, as Slurm runs them: A = (s0 pass, s1 negative) -> PASS (1/2);
        B = (s0 killed at model_1000, s1 negative) -> NEGATIVE, 1 killed."""
        WA = _fake_tree(self.root / "A", "777")
        WB = _fake_tree(self.root / "B", "777")
        procs = [_launch(WA, "9201", "0", "777", FAKE_MODE_S0="pass", FAKE_SLEEP_S0="125"),
                 _launch(WA, "9201", "1", "777", FAKE_MODE_S1="neg", FAKE_SLEEP_S1="140"),
                 _launch(WB, "9202", "0", "777", FAKE_MODE_S0="kill"),
                 _launch(WB, "9202", "1", "777", FAKE_MODE_S1="neg", FAKE_SLEEP_S1="125")]
        t0 = time.time()
        rcs = [p.wait(timeout=900) for p, _ in procs]
        logs = [lg.read_text() for _, lg in procs]
        self.assertEqual(rcs, [0, 0, 0, 0], "\n=====\n".join(logs))
        for W, aj in ((WA, "9201"), (WB, "9202")):
            for s in ("0", "1"):
                call = (W / f"train_calls_s{s}.txt").read_text().strip().splitlines()
                self.assertEqual(len(call), 1)
                self.assertIn(f"TASK=TaskV2-BHL-CubeToShelfStand5-Blind-v0 SEED={s} MAX_ITER=8000 NUM_ENVS=1024 "
                              f"RUN_NAME=v2-cubetoshelfstand5-blind-s{s} BHL_STACK=v60 ENABLE_CAMERAS=1 "
                              "OVERRIDE_SIZE=0", call[0])
                out = self.out_dir(W)
                rc = json.loads((out / f"runner_check_s{s}_{aj}_{s}.json").read_text())
                self.assertTrue(rc["ok"], rc)
                self.assertIn("BYTES-CHECK v2-cubetoshelfstand5-blind-s" + s + ": PASS",
                              "".join(logs))
                self.assertFalse(list(out.glob(".start-*")))
        outA, outB = self.out_dir(WA), self.out_dir(WB)
        a0 = json.loads((outA / "result_s0_9201.json").read_text())
        self.assertTrue(a0["complete"] and a0["passes"] and not a0["killed"])
        self.assertEqual(a0["rule"], s5.PREDECLARED_RULE5)
        pa = json.loads((outA / "result_pair_9201.json").read_text())
        self.assertEqual(pa["verdict"], f"{s5.PASS_LABEL5} (1/2 seeds)")
        self.assertEqual(pa["design"], list(s5.DESIGN_S))
        self.assertEqual(pa["stand4_rule"], s4.PREDECLARED_RULE)
        kb = json.loads((outB / "kill_s0_9202_0.json").read_text())
        self.assertEqual((kb["verdict"], kb["ready"]), ("KILL", True))
        b0 = json.loads((outB / "result_s0_9202.json").read_text())
        self.assertTrue(b0["killed"] and b0["decided"] and not b0["passes"])
        self.assertTrue((outB / "killed_s0_9202_0.flag").is_file())
        pb = json.loads((outB / "result_pair_9202.json").read_text())
        self.assertTrue(pb["verdict"].startswith("NEGATIVE (0/2 seeds") and "1 killed" in pb["verdict"], pb)
        self.assertLess(time.time() - t0, 900)


if __name__ == "__main__":
    unittest.main()
