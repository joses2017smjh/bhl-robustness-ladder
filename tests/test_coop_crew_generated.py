"""COOP-21: the generated crew configs against the Hydra round trip.

docs/REPO_TASKS.md records the crew-of-3/4 gate as "FAIL x4 and import error"
with the fault "isolated to the Hydra round trip of generated configclass
fields" and asks for an annotation fix in ``coop_crew_generated.py``. The
history says otherwise, and this file pins what is actually true:

* Isaac Lab's ``@configclass`` (2.3.2 and 3.0.0b2, identical code) annotates
  every un-annotated class attribute from ``type(default)`` before it hands the
  class to ``dataclass``. The only hard requirement is *annotation or default*;
  a bare ``MISSING`` without an annotation is the one thing it refuses. The
  generated classes use exactly the hand-written pair's style
  (``contact_a = ContactSensorCfg(...)``), so there is no annotation
  mismatch to fix. ``test_every_member_is_annotated_or_has_a_default`` holds
  that invariant on the source.
* The ``policy/object_pos_a: scene entity 'robot_a' does not exist`` error the
  four gates died on came from ``apply_depth_flags`` -- the post-Hydra hook
  ``scripts/train.py`` runs on every task -- whose restore branch *added* the
  pair's object-pose terms to any policy group lacking them. Commit ``73bc035``
  ("apply_depth_flags injected coop terms into every task. That was the crew
  bug.") guarded it on 2026-08-24, after the last gate run (21006649) and after
  the crew_diag job whose "CONCLUSION" the task entry quotes; that conclusion
  is the script's else-branch, not a measurement. No gate has run since.
  ``test_post_hydra_hooks_are_guarded`` keeps the guards in place.
* The gate's own "import FAIL" line is an artifact: the import check prints
  ``IMPORT OK`` into a redirected file without flushing and then calls
  ``SimulationApp.close()``, which tears the process down from C++. The
  rerun launcher (slurm/repo20260923/gpu_crew_gate.sbatch) writes the verdict
  to disk before closing.

Two tiers:

1. AST checks on the generated source, the generator and the two hook
   functions. Always run; no Isaac, no torch.
2. The round trip itself, in a subprocess: the real ``configclass``,
   ``class_to_dict`` / ``update_class_from_dict``, manager term cfgs,
   ``SceneEntityCfg``, ``InteractiveSceneCfg`` and noise cfgs are loaded from
   the pinned Isaac Lab source tree (its package ``__init__`` is bypassed
   because importing it boots Isaac Sim); only the simulator-bound leaves
   (``isaaclab.sim`` / ``assets`` / ``sensors`` / ``terrains`` / ``envs``,
   ``isaaclab_rl``, the upstream robot asset and ``mdp``) are stubbed as
   configclasses with the same field names. The real ``coop_lift_mdp``,
   ``coop_lift_env_cfg``, ``coop_depth_env_cfg`` and ``coop_crew_generated``
   modules are then imported unmodified, every crew class is pushed through
   ``to_dict`` -> ``ConfigStore`` -> ``hydra.compose`` (with the gate's own
   override) -> ``from_dict`` -> ``apply_strategy_flags`` /
   ``apply_depth_flags`` exactly as ``scripts/train.py`` does, and the pair
   configs ride the same path as the positive control. Skipped only when the
   source tree is not on this interpreter's path (a fresh test env).
"""
from __future__ import annotations

import ast
import json
import os
import re
import shutil
import site
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
GENERATED = REPO / "src/bhl_robust/tasks/coop_crew_generated.py"
GENERATOR = REPO / "scripts/gen_crew_cfg.py"
LIFT_CFG = REPO / "src/bhl_robust/tasks/coop_lift_env_cfg.py"
DEPTH_CFG = REPO / "src/bhl_robust/tasks/coop_depth_env_cfg.py"
BIPED_DEPTH_CFG = REPO / "src/bhl_robust/tasks/depth_env_cfg.py"

#: The four registered crew classes (src/bhl_robust/tasks/__init__.py) and their sizes.
CREW_CLASSES = {"Crew3Cfg": (3, False), "Crew4Cfg": (4, False),
                "Crew3DepthCfg": (3, True), "Crew4DepthCfg": (4, True)}
#: The override slurm/inner/crew_validate.sh writes into OVERRIDE_FILE for every gate task.
GATE_OVERRIDE = "env.stage_lift_on_pinch=true"
HARNESS_MARK = "COOP_CREW_HARNESS_JSON:"


# --------------------------------------------------------------------------
# Tier 1: source-level invariants
# --------------------------------------------------------------------------

def _is_configclass_decorator(node: ast.expr) -> bool:
    return (isinstance(node, ast.Name) and node.id == "configclass") or (
        isinstance(node, ast.Attribute) and node.attr == "configclass")


def _configclasses(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and any(_is_configclass_decorator(d) for d in node.decorator_list):
            yield node


def _is_missing(node: ast.expr | None) -> bool:
    return node is None or (isinstance(node, ast.Name) and node.id == "MISSING") or (
        isinstance(node, ast.Attribute) and node.attr == "MISSING")


def _class(tree: ast.AST, name: str) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"class {name} not found")


def _function(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name} not found")


def test_every_member_is_annotated_or_has_a_default():
    """The configclass contract: a field is an annotation or a non-MISSING default.

    ``_add_annotation_types`` supplies ``type(default)`` for the un-annotated
    ones, so the generated ``term = ObsTerm(...)`` style is a field, not a
    stray attribute. What it cannot do is annotate a bare ``MISSING``.
    """
    tree = ast.parse(GENERATED.read_text())
    members, problems = 0, []
    for cls in _configclasses(tree):
        for stmt in cls.body:
            if isinstance(stmt, ast.AnnAssign):
                members += 1
                if _is_missing(stmt.value):
                    problems.append(f"{cls.name}.{ast.unparse(stmt.target)}: annotated but no default")
            elif isinstance(stmt, ast.Assign):
                members += 1
                if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                    problems.append(f"{cls.name}: unusual assignment {ast.unparse(stmt)[:60]}")
                elif _is_missing(stmt.value):
                    problems.append(f"{cls.name}.{stmt.targets[0].id}: MISSING without an annotation")
            elif not isinstance(stmt, (ast.FunctionDef, ast.ClassDef, ast.Expr, ast.Pass)):
                problems.append(f"{cls.name}: unexpected statement {type(stmt).__name__}")
    assert not problems, "\n".join(problems)
    # Four crews x (scene, obs, actions, rewards, terminations, events, env) is
    # several hundred members; a walker that saw none would pass vacuously.
    assert members > 300, members


def test_no_pair_identifiers_in_generated_source():
    """Nothing named after the pair (``robot_a``, ``contact_b``, ``object_pos_a``...)."""
    src = GENERATED.read_text()
    pair_names = sorted(set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*_[ab]\b", src)))
    assert pair_names == [], pair_names


def test_crew_classes_declared_as_registered():
    tree = ast.parse(GENERATED.read_text())
    for name, (n, vision) in CREW_CLASSES.items():
        cls = _class(tree, name)
        assert [ast.unparse(b) for b in cls.bases] == ["CoopLiftEnvCfg"], name
        fields = {s.target.id: s for s in cls.body if isinstance(s, ast.AnnAssign)}
        assert ast.literal_eval(fields["crew_size"].value) == n, name
        tag = f"Crew{n}{'Depth' if vision else ''}"
        for part in ("Scene", "Observations", "Actions", "Rewards", "Terminations", "Events"):
            field = fields[part.lower()]
            assert ast.unparse(field.annotation) == f"{tag}{part}Cfg", (name, part)
            assert _class(tree, f"{tag}{part}Cfg") is not None
        assert ast.unparse(fields["curriculum"].annotation) == "CurriculumCfg", name
        # No crew declares the depth arm's selector: apply_depth_flags must stay a no-op on them.
        assert "drop_object_pose" not in fields, name
    src = GENERATED.read_text()
    assert "drop_object_pose" not in src


def test_post_hydra_hooks_are_guarded():
    """The two hooks train.py runs after Hydra must return early on configs
    that do not declare their flag. The unguarded ``apply_depth_flags`` is
    what killed the four crew gates (commit 73bc035)."""
    def first_guard(func: ast.FunctionDef) -> str | None:
        for stmt in func.body:
            if isinstance(stmt, ast.Expr):        # docstring
                continue
            if (isinstance(stmt, ast.If) and isinstance(stmt.test, ast.UnaryOp)
                    and isinstance(stmt.test.op, ast.Not) and isinstance(stmt.test.operand, ast.Call)
                    and ast.unparse(stmt.test.operand.func) == "hasattr"
                    and len(stmt.body) == 1 and isinstance(stmt.body[0], ast.Return)):
                return ast.literal_eval(stmt.test.operand.args[1])
            return None
        return None

    depth = _function(ast.parse(DEPTH_CFG.read_text()), "apply_depth_flags")
    assert first_guard(depth) == "drop_object_pose", ast.unparse(depth.body[1])[:200]
    lift_tree = ast.parse(LIFT_CFG.read_text())
    strategy = _function(lift_tree, "apply_strategy_flags")
    assert first_guard(strategy) == "privileged_critic"
    # CoopLiftEnvCfg, the crews' base, declares privileged_critic (so the
    # strategy hook acts on crews) and not drop_object_pose (so the depth hook does not).
    base = _class(lift_tree, "CoopLiftEnvCfg")
    declared = {s.target.id for s in base.body if isinstance(s, ast.AnnAssign)} | {
        s.targets[0].id for s in base.body if isinstance(s, ast.Assign) and isinstance(s.targets[0], ast.Name)}
    assert "privileged_critic" in declared and "stage_lift_on_pinch" in declared
    assert "drop_object_pose" not in declared


def test_generated_file_matches_generator(tmp_path):
    """The committed file is what ``scripts/gen_crew_cfg.py`` emits; a hand edit would drift."""
    (tmp_path / "scripts").mkdir()
    (tmp_path / "src/bhl_robust/tasks").mkdir(parents=True)
    shutil.copy(GENERATOR, tmp_path / "scripts/gen_crew_cfg.py")
    proc = subprocess.run([sys.executable, str(tmp_path / "scripts/gen_crew_cfg.py")],
                          cwd=tmp_path, capture_output=True, text=True, timeout=60,
                          stdin=subprocess.DEVNULL)
    assert proc.returncode == 0, proc.stderr
    regenerated = (tmp_path / "src/bhl_robust/tasks/coop_crew_generated.py").read_bytes()
    assert regenerated == GENERATED.read_bytes(), "coop_crew_generated.py differs from the generator's output"


# --------------------------------------------------------------------------
# Tier 2: the round trip, on the real configclass machinery, in a subprocess
# --------------------------------------------------------------------------

def _isaaclab_source_tree() -> Path | None:
    """The pinned Isaac Lab source tree inside this interpreter's site-packages, if any."""
    bases = {sysconfig.get_paths()["purelib"], *site.getsitepackages()}
    for base in sorted(bases):
        cand = Path(base) / "isaaclab/source/isaaclab/isaaclab"
        if (cand / "utils/configclass.py").exists() and (cand / "managers/manager_term_cfg.py").exists():
            return cand
    return None


def _module_constants(path: Path, prefix: str) -> dict:
    """Literal ``PREFIX_* = ...`` assignments of a module, read without importing it."""
    out = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id.startswith(prefix):
            out[node.targets[0].id] = ast.literal_eval(node.value)
    return out


_LEG_JOINTS = [f"leg_{s}_{j}_joint" for s in ("left", "right")
               for j in ("hip_roll", "hip_yaw", "hip_pitch", "knee_pitch", "ankle_pitch", "ankle_roll")]
_ARM_JOINTS = [f"arm_{s}_{j}_joint" for s in ("left", "right")
               for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow_pitch", "elbow_roll")]

# Stubs for the simulator-bound leaves. Every one is a real @configclass with the
# field names the four coop modules actually touch, so to_dict/from_dict treats
# them exactly as it treats the originals; none of them can boot Isaac Sim.
_STUB_SOURCES = {
    "isaaclab.sim": '''
from dataclasses import MISSING
from isaaclab.utils import configclass

@configclass
class RigidBodyMaterialCfg:
    friction_combine_mode: str = "average"
    restitution_combine_mode: str = "average"
    static_friction: float = 0.5
    dynamic_friction: float = 0.5
    restitution: float = 0.0

@configclass
class RigidBodyPropertiesCfg:
    disable_gravity: bool | None = None
    max_depenetration_velocity: float | None = None
    solver_position_iteration_count: int | None = None
    solver_velocity_iteration_count: int | None = None

@configclass
class CollisionPropertiesCfg:
    collision_enabled: bool | None = None

@configclass
class MassPropertiesCfg:
    mass: float | None = None

@configclass
class PreviewSurfaceCfg:
    diffuse_color: tuple = (0.18, 0.18, 0.18)

@configclass
class _ShapeCfg:
    rigid_props: RigidBodyPropertiesCfg | None = None
    collision_props: CollisionPropertiesCfg | None = None
    mass_props: MassPropertiesCfg | None = None
    visual_material: PreviewSurfaceCfg | None = None
    physics_material: RigidBodyMaterialCfg | None = None

@configclass
class CuboidCfg(_ShapeCfg):
    size: tuple = MISSING

@configclass
class SphereCfg(_ShapeCfg):
    radius: float = MISSING

@configclass
class DistantLightCfg:
    color: tuple = (1.0, 1.0, 1.0)
    intensity: float = 1.0

@configclass
class DomeLightCfg:
    color: tuple = (1.0, 1.0, 1.0)
    intensity: float = 1.0

@configclass
class PhysxCfg:
    gpu_max_rigid_patch_count: int = 5 * 2**15
    bounce_threshold_velocity: float = 0.5

@configclass
class SimulationCfg:
    device: str = "cuda:0"
    dt: float = 1.0 / 60.0
    render_interval: int = 1
    disable_contact_processing: bool = False
    physx: PhysxCfg = PhysxCfg()
    physics_material: RigidBodyMaterialCfg = RigidBodyMaterialCfg()
''',
    "isaaclab.assets": '''
from dataclasses import MISSING
from isaaclab.utils import configclass

class AssetBase: pass
class Articulation(AssetBase): pass
class RigidObject(AssetBase): pass

@configclass
class AssetBaseCfg:
    @configclass
    class InitialStateCfg:
        pos: tuple = (0.0, 0.0, 0.0)
        rot: tuple = (1.0, 0.0, 0.0, 0.0)
    prim_path: str = MISSING
    spawn: object | None = None
    init_state: InitialStateCfg = InitialStateCfg()
    collision_group: int = 0
    debug_vis: bool = False

@configclass
class RigidObjectCfg(AssetBaseCfg):
    @configclass
    class InitialStateCfg(AssetBaseCfg.InitialStateCfg):
        lin_vel: tuple = (0.0, 0.0, 0.0)
        ang_vel: tuple = (0.0, 0.0, 0.0)
    init_state: InitialStateCfg = InitialStateCfg()

@configclass
class ArticulationCfg(AssetBaseCfg):
    @configclass
    class InitialStateCfg(AssetBaseCfg.InitialStateCfg):
        lin_vel: tuple = (0.0, 0.0, 0.0)
        ang_vel: tuple = (0.0, 0.0, 0.0)
        joint_pos: dict = {".*": 0.0}
        joint_vel: dict = {".*": 0.0}
    init_state: InitialStateCfg = InitialStateCfg()
    articulation_root_prim_path: str | None = None
    soft_joint_pos_limit_factor: float = 1.0
    actuators: dict = MISSING
''',
    "isaaclab.sensors": '''
from dataclasses import MISSING
from isaaclab.utils import configclass

class SensorBase: pass
class ContactSensor(SensorBase): pass

@configclass
class SensorBaseCfg:
    prim_path: str = MISSING
    update_period: float = 0.0
    history_length: int = 0
    debug_vis: bool = False

@configclass
class ContactSensorCfg(SensorBaseCfg):
    track_pose: bool = False
    track_air_time: bool = False
    force_threshold: float = 1.0
    filter_prim_paths_expr: list = []

@configclass
class RayCasterCameraCfg(SensorBaseCfg):
    @configclass
    class OffsetCfg:
        pos: tuple = (0.0, 0.0, 0.0)
        rot: tuple = (1.0, 0.0, 0.0, 0.0)
        convention: str = "ros"
    offset: OffsetCfg = OffsetCfg()
    mesh_prim_paths: list = MISSING
    pattern_cfg: object = MISSING
    data_types: list = ["distance_to_image_plane"]
    depth_clipping_behavior: str = "none"
    max_distance: float = 1e6

@configclass
class MultiMeshRayCasterCameraCfg(RayCasterCameraCfg):
    @configclass
    class RaycastTargetCfg:
        prim_expr: str = MISSING
        is_shared: bool = False
        track_mesh_transforms: bool = True
    update_mesh_ids: bool = False
    reference_meshes: bool = True
''',
    "isaaclab.sensors.patterns": '''
from dataclasses import MISSING
from isaaclab.utils import configclass

@configclass
class PinholeCameraPatternCfg:
    focal_length: float = 24.0
    horizontal_aperture: float = 20.955
    width: int = MISSING
    height: int = MISSING
''',
    "isaaclab.terrains": '''
from dataclasses import MISSING
from isaaclab.utils import configclass
from isaaclab.sim import RigidBodyMaterialCfg

@configclass
class TerrainImporterCfg:
    prim_path: str = MISSING
    terrain_type: str = "generator"
    terrain_generator: object | None = None
    collision_group: int = -1
    num_envs: int = 1
    env_spacing: float | None = None
    usd_path: str | None = None
    physics_material: RigidBodyMaterialCfg = RigidBodyMaterialCfg()
    visual_material: object | None = None
    debug_vis: bool = False
''',
    "isaaclab.envs": '''
from dataclasses import MISSING
from isaaclab.utils import configclass
from isaaclab.sim import SimulationCfg
from isaaclab.scene import InteractiveSceneCfg

class ManagerBasedRLEnv: pass

@configclass
class ViewerCfg:
    eye: tuple = (7.5, 7.5, 7.5)
    lookat: tuple = (0.0, 0.0, 0.0)

@configclass
class DefaultEventManagerCfg:
    pass

@configclass
class ManagerBasedEnvCfg:
    viewer: ViewerCfg = ViewerCfg()
    sim: SimulationCfg = SimulationCfg()
    seed: int | None = None
    decimation: int = MISSING
    scene: InteractiveSceneCfg = MISSING
    observations: object = MISSING
    actions: object = MISSING
    events: object = DefaultEventManagerCfg()
    rerender_on_reset: bool = False
    wait_for_textures: bool = True

@configclass
class ManagerBasedRLEnvCfg(ManagerBasedEnvCfg):
    is_finite_horizon: bool = False
    episode_length_s: float = MISSING
    rewards: object = MISSING
    terminations: object = MISSING
    curriculum: object | None = None
    commands: object | None = None
''',
    "isaaclab.envs.mdp": '''
from dataclasses import MISSING
from isaaclab.utils import configclass
from isaaclab.managers import ActionTermCfg

def projected_gravity(env, asset_cfg): ...
def base_ang_vel(env, asset_cfg): ...
def base_lin_vel(env, asset_cfg): ...
def joint_pos_rel(env, asset_cfg): ...
def joint_vel_rel(env, asset_cfg): ...
def last_action(env): ...
def is_terminated(env): ...
def action_rate_l2(env): ...
def flat_orientation_l2(env, asset_cfg): ...
def joint_torques_l2(env, asset_cfg): ...
def undesired_contacts(env, sensor_cfg, threshold): ...
def time_out(env): ...
def reset_joints_by_offset(env, env_ids, position_range, velocity_range, asset_cfg): ...
def reset_root_state_uniform(env, env_ids, pose_range, velocity_range, asset_cfg): ...

class JointPositionAction: pass

@configclass
class JointActionCfg(ActionTermCfg):
    joint_names: list = MISSING
    scale: float = 1.0
    offset: float = 0.0
    preserve_order: bool = False

@configclass
class JointPositionActionCfg(JointActionCfg):
    class_type: type = JointPositionAction
    use_default_offset: bool = True
''',
    "isaaclab_rl.rsl_rl": '''
from dataclasses import MISSING
from isaaclab.utils import configclass

@configclass
class RslRlPpoActorCriticCfg:
    class_name: str = "ActorCritic"
    init_noise_std: float = 1.0
    actor_hidden_dims: list = MISSING
    critic_hidden_dims: list = MISSING
    activation: str = MISSING

@configclass
class RslRlPpoAlgorithmCfg:
    class_name: str = "PPO"
    value_loss_coef: float = MISSING
    use_clipped_value_loss: bool = MISSING
    clip_param: float = MISSING
    entropy_coef: float = MISSING
    num_learning_epochs: int = MISSING
    num_mini_batches: int = MISSING
    learning_rate: float = MISSING
    schedule: str = MISSING
    gamma: float = MISSING
    lam: float = MISSING
    desired_kl: float = MISSING
    max_grad_norm: float = MISSING

@configclass
class RslRlOnPolicyRunnerCfg:
    seed: int = 42
    device: str = "cuda:0"
    num_steps_per_env: int = MISSING
    max_iterations: int = MISSING
    empirical_normalization: bool = MISSING
    policy: RslRlPpoActorCriticCfg = MISSING
    algorithm: RslRlPpoAlgorithmCfg = MISSING
    save_interval: int = MISSING
    experiment_name: str = MISSING
    run_name: str = ""
    logger: str = "tensorboard"
    resume: bool = False
    load_run: str = ".*"
    load_checkpoint: str = "model_.*.pt"
''',
    "berkeley_humanoid_lite_assets.robots.berkeley_humanoid_lite": '''
from isaaclab.assets import ArticulationCfg

HUMANOID_LITE_LEG_JOINTS = %(legs)r
HUMANOID_LITE_ARM_JOINTS = %(arms)r
HUMANOID_LITE_JOINTS = HUMANOID_LITE_ARM_JOINTS + HUMANOID_LITE_LEG_JOINTS
HUMANOID_LITE_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5), joint_pos={j: 0.0 for j in HUMANOID_LITE_JOINTS}, joint_vel={".*": 0.0}),
    actuators={},
)
''' % {"legs": _LEG_JOINTS, "arms": _ARM_JOINTS},
}


def _run_harness() -> dict:
    """Executed in the subprocess. Returns the JSON-able verdict."""
    import importlib
    import traceback
    import types

    src = Path(os.environ["BHL_TEST_ISAACLAB_SRC"])
    try:
        import torch  # noqa: F401
        import warp  # noqa: F401
        import hydra
        import omegaconf
        from hydra import compose, initialize
        from hydra.core.config_store import ConfigStore
        from omegaconf import OmegaConf
    except ImportError as exc:                      # a fresh env without the pinned stack
        return {"status": "skip", "reason": f"round-trip dependencies unavailable: {exc}"}

    def stub_pkg(name: str, path: Path | None = None) -> types.ModuleType:
        mod = types.ModuleType(name)
        mod.__path__ = [str(path)] if path is not None else []
        mod.__file__ = f"<stub {name}>"
        sys.modules[name] = mod
        parent, _, child = name.rpartition(".")
        if parent:
            setattr(sys.modules[parent], child, mod)
        return mod

    def stub_module(name: str, source: str) -> types.ModuleType:
        mod = stub_pkg(name)
        exec(source, mod.__dict__)
        return mod

    try:
        # --- the real machinery under test, from the source tree, package __init__ bypassed
        stub_pkg("isaaclab")
        utils = stub_pkg("isaaclab.utils", src / "utils")
        cc = importlib.import_module("isaaclab.utils.configclass")
        utils.configclass = cc.configclass
        dict_mod = importlib.import_module("isaaclab.utils.dict")
        for n in ("class_to_dict", "update_class_from_dict",
                  "replace_slices_with_strings", "replace_strings_with_slices"):
            setattr(utils, n, getattr(dict_mod, n))
        noise = importlib.import_module("isaaclab.utils.noise")
        importlib.import_module("isaaclab.utils.math")
        managers = stub_pkg("isaaclab.managers", src / "managers")
        term_cfg = importlib.import_module("isaaclab.managers.manager_term_cfg")
        for n in ("ManagerTermBaseCfg", "ObservationGroupCfg", "ObservationTermCfg", "RewardTermCfg",
                  "EventTermCfg", "TerminationTermCfg", "CurriculumTermCfg", "ActionTermCfg"):
            setattr(managers, n, getattr(term_cfg, n))
        managers.SceneEntityCfg = importlib.import_module("isaaclab.managers.scene_entity_cfg").SceneEntityCfg
        scene = stub_pkg("isaaclab.scene", src / "scene")
        scene.InteractiveSceneCfg = importlib.import_module("isaaclab.scene.interactive_scene_cfg").InteractiveSceneCfg
        real_files = {n: sys.modules[n].__file__ for n in (
            "isaaclab.utils.configclass", "isaaclab.utils.dict", "isaaclab.utils.noise",
            "isaaclab.managers.manager_term_cfg", "isaaclab.managers.scene_entity_cfg",
            "isaaclab.scene.interactive_scene_cfg")}
        assert hasattr(noise, "AdditiveUniformNoiseCfg"), "pinned noise package lacks the alias the coop cfgs import"

        # --- simulator-bound leaves
        stub_module("isaaclab.sim", _STUB_SOURCES["isaaclab.sim"])
        stub_module("isaaclab.assets", _STUB_SOURCES["isaaclab.assets"])
        stub_module("isaaclab.sensors", _STUB_SOURCES["isaaclab.sensors"])
        stub_module("isaaclab.sensors.patterns", _STUB_SOURCES["isaaclab.sensors.patterns"])
        stub_module("isaaclab.terrains", _STUB_SOURCES["isaaclab.terrains"])
        stub_module("isaaclab.envs", _STUB_SOURCES["isaaclab.envs"])
        mdp = stub_module("isaaclab.envs.mdp", _STUB_SOURCES["isaaclab.envs.mdp"])
        stub_pkg("isaaclab_rl")
        stub_module("isaaclab_rl.rsl_rl", _STUB_SOURCES["isaaclab_rl.rsl_rl"])
        stub_pkg("berkeley_humanoid_lite_assets")
        stub_pkg("berkeley_humanoid_lite_assets.robots")
        stub_module("berkeley_humanoid_lite_assets.robots.berkeley_humanoid_lite",
                    _STUB_SOURCES["berkeley_humanoid_lite_assets.robots.berkeley_humanoid_lite"])
        stub_pkg("berkeley_humanoid_lite")
        stub_pkg("berkeley_humanoid_lite.tasks")
        stub_pkg("berkeley_humanoid_lite.tasks.locomotion")
        velocity = stub_pkg("berkeley_humanoid_lite.tasks.locomotion.velocity")
        # Upstream's velocity.mdp star-imports isaaclab.envs.mdp; the same module object serves both names.
        sys.modules["berkeley_humanoid_lite.tasks.locomotion.velocity.mdp"] = mdp
        velocity.mdp = mdp

        # --- the repo modules, unmodified; bhl_robust.tasks' registration __init__ is bypassed
        import bhl_robust
        tasks = stub_pkg("bhl_robust.tasks", REPO / "src/bhl_robust/tasks")
        bhl_robust.tasks = tasks
        cam = stub_pkg("bhl_robust.tasks.depth_env_cfg")
        for k, v in _module_constants(BIPED_DEPTH_CFG, "CAM_").items():
            setattr(cam, k, v)
        importlib.import_module("bhl_robust.tasks.coop_lift_mdp")
        lift = importlib.import_module("bhl_robust.tasks.coop_lift_env_cfg")
        depth = importlib.import_module("bhl_robust.tasks.coop_depth_env_cfg")
        crew = importlib.import_module("bhl_robust.tasks.coop_crew_generated")

        ObsTerm, RewTerm = term_cfg.ObservationTermCfg, term_cfg.RewardTermCfg
        EvTerm, DoneTerm, ActTerm = term_cfg.EventTermCfg, term_cfg.TerminationTermCfg, term_cfg.ActionTermCfg

        def names(group, kind) -> list[str]:
            return sorted(k for k, v in vars(group).items() if isinstance(v, kind))

        def snapshot(cfg) -> dict:
            groups = {"policy": names(cfg.observations.policy, ObsTerm),
                      "critic": names(cfg.observations.critic, ObsTerm),
                      "rewards": names(cfg.rewards, RewTerm), "events": names(cfg.events, EvTerm),
                      "terminations": names(cfg.terminations, DoneTerm), "actions": names(cfg.actions, ActTerm),
                      "scene": sorted(k for k, v in vars(cfg.scene).items() if hasattr(v, "prim_path"))}
            pair = sorted(f"{g}.{k}" for g, obj in (
                ("policy", cfg.observations.policy), ("critic", cfg.observations.critic), ("rewards", cfg.rewards),
                ("events", cfg.events), ("actions", cfg.actions), ("scene", cfg.scene))
                for k, v in vars(obj).items() if v is not None and re.search(r"_[ab]$", k))
            return {**groups, "pair_terms_live": pair,
                    "lift_weights": [cfg.rewards.lift_progress.weight, cfg.rewards.lifting_object.weight],
                    "stage_lift_on_pinch": bool(cfg.stage_lift_on_pinch),
                    "has_drop_object_pose": hasattr(cfg, "drop_object_pose"),
                    "policy_object_pos_a_live": getattr(cfg.observations.policy, "object_pos_a", None) is not None,
                    "contact_offset": cfg.contact_offset, "object_spawn_z": cfg.object_spawn_z,
                    "crew_size": getattr(cfg, "crew_size", None)}

        def norm(value):
            if isinstance(value, dict):
                return {str(k): norm(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [norm(v) for v in value]
            return value

        def hydra_round_trip(cfg_cls, task_name: str, overrides: list[str]) -> dict:
            """register_task_to_hydra + hydra_main + train.py's two post-Hydra hooks."""
            env_cfg = cfg_cls()
            agent_cfg = lift.CoopLiftPPORunnerCfg()
            before = snapshot(env_cfg)
            before_dict = norm(dict_mod.replace_slices_with_strings(env_cfg.to_dict()))
            cfg_dict = {"env": env_cfg.to_dict(), "agent": agent_cfg.to_dict()}
            cfg_dict = dict_mod.replace_slices_with_strings(cfg_dict)
            ConfigStore.instance().store(name=task_name, node=cfg_dict)
            with initialize(version_base="1.3", config_path=None, job_name="coop_crew_round_trip"):
                hydra_cfg = compose(config_name=task_name, overrides=overrides)
            container = OmegaConf.to_container(hydra_cfg, resolve=True)
            container = dict_mod.replace_strings_with_slices(container)
            env_cfg.from_dict(container["env"])
            agent_cfg.from_dict(container["agent"])
            lift.apply_strategy_flags(env_cfg)
            depth.apply_depth_flags(env_cfg)
            after = snapshot(env_cfg)
            after_dict = norm(dict_mod.replace_slices_with_strings(env_cfg.to_dict()))
            return {"before": before, "after": after, "dict_unchanged": before_dict == after_dict}

        result = {"status": "ok", "isaaclab_src": str(src), "real_files": real_files,
                  "hydra_version": hydra.__version__, "omegaconf_version": omegaconf.__version__,
                  "gate_override": GATE_OVERRIDE, "classes": {}, "controls": {}}
        for name, (n, vision) in CREW_CLASSES.items():
            cls = getattr(crew, name)
            result["classes"][name] = {
                "n": n, "vision": vision,
                "plain": hydra_round_trip(cls, f"CoopLift-BHL-Cube-Crew{n}{'-Depth' if vision else ''}-v0", []),
                "gate": hydra_round_trip(cls, f"CoopLift-BHL-Cube-Crew{n}{'-Depth' if vision else ''}-v0-gate",
                                         [GATE_OVERRIDE]),
            }
        # Positive controls: the pair trains under this exact path in production.
        result["controls"]["CoopLiftCubeCfg"] = {
            "plain": hydra_round_trip(lift.CoopLiftCubeCfg, "CoopLift-BHL-Cube-v0", []),
            "gate": hydra_round_trip(lift.CoopLiftCubeCfg, "CoopLift-BHL-Cube-v0-gate", [GATE_OVERRIDE])}
        result["controls"]["CoopLiftDepthCfg"] = {
            "plain": hydra_round_trip(depth.CoopLiftDepthCfg, "CoopLift-BHL-Cube-Depth-v0", []),
            "gate": hydra_round_trip(depth.CoopLiftDepthCfg, "CoopLift-BHL-Cube-Depth-v0-gate", [GATE_OVERRIDE])}
        return result
    except Exception:
        return {"status": "error", "traceback": traceback.format_exc()}


@pytest.fixture(scope="module")
def harness():
    src = _isaaclab_source_tree()
    if src is None:
        pytest.skip("pinned Isaac Lab source tree is not on this interpreter's path (fresh test env)")
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"), OMP_NUM_THREADS="2",
               BHL_TEST_ISAACLAB_SRC=str(src), HYDRA_FULL_ERROR="1")
    proc = subprocess.run([sys.executable, __file__, "--harness"], cwd=REPO, env=env,
                          stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=240)
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith(HARNESS_MARK)]
    assert lines, f"harness produced no verdict (rc {proc.returncode})\n{proc.stderr[-3000:]}"
    result = json.loads(lines[-1][len(HARNESS_MARK):])
    if result["status"] == "skip":
        pytest.skip(result["reason"])
    assert result["status"] == "ok", result.get("traceback", "") + proc.stderr[-2000:]
    return result


def test_round_trip_ran_on_the_real_machinery(harness):
    for mod, path in harness["real_files"].items():
        assert path.startswith(harness["isaaclab_src"]), (mod, path)
    assert harness["hydra_version"].startswith("1.") and harness["omegaconf_version"].startswith("2.")


@pytest.mark.parametrize("control", ["CoopLiftCubeCfg", "CoopLiftDepthCfg"])
def test_pair_controls_survive_the_same_path(harness, control):
    """A harness that fails the pair is a broken harness: the pair trains under Hydra."""
    for run in ("plain", "gate"):
        r = harness["controls"][control][run]
        for key in ("policy", "critic", "rewards", "events", "terminations", "actions", "scene"):
            assert r["before"][key] == r["after"][key], (control, run, key)
        assert "robot_a" in r["after"]["scene"] and "robot_b" in r["after"]["scene"]
    assert harness["controls"][control]["plain"]["dict_unchanged"], control
    # The depth hook acts where the flag is declared (drops the pose) and only there.
    live = harness["controls"][control]["plain"]["after"]["policy_object_pos_a_live"]
    assert live is (control == "CoopLiftCubeCfg"), (control, live)


@pytest.mark.parametrize("name", sorted(CREW_CLASSES))
def test_crew_terms_survive_hydra_round_trip(harness, name):
    r = harness["classes"][name]
    for run in ("plain", "gate"):
        for key in ("policy", "critic", "rewards", "events", "terminations", "actions", "scene"):
            assert r[run]["before"][key] == r[run]["after"][key], (name, run, key)
    assert r["plain"]["dict_unchanged"], f"{name}: to_dict changed across an override-free round trip"


@pytest.mark.parametrize("name", sorted(CREW_CLASSES))
def test_no_pair_terms_reach_a_crew_after_the_train_hooks(harness, name):
    """The COOP-21 failure mode: object_pos_a / robot_a appearing on a crew config."""
    for run in ("plain", "gate"):
        after = harness["classes"][name][run]["after"]
        assert after["pair_terms_live"] == [], (name, run, after["pair_terms_live"])
        assert after["policy_object_pos_a_live"] is False, (name, run)
        assert "robot_a" not in after["scene"] and "contact_a" not in after["scene"], (name, run)
        assert after["has_drop_object_pose"] is False, (name, run)


@pytest.mark.parametrize("name", sorted(CREW_CLASSES))
def test_crew_shape_and_gate_override(harness, name):
    n, vision = CREW_CLASSES[name]
    r = harness["classes"][name]
    after = r["gate"]["after"]
    per_robot = 7 if vision else 6
    assert len(after["policy"]) == per_robot * n + 1, (name, after["policy"])
    assert len(after["critic"]) == per_robot * n + 1 + 2 + n, (name, after["critic"])
    assert len(after["actions"]) == n and len(after["rewards"]) == 12 + 3 * n
    assert len(after["events"]) == 2 * n + 1 and len(after["terminations"]) == 2
    assert [f"robot_{i}" for i in range(n)] == sorted(s for s in after["scene"] if s.startswith("robot_"))
    assert len([s for s in after["scene"] if s.startswith("cam_")]) == (n if vision else 0)
    assert after["crew_size"] == n
    # The crew's __post_init__ values outlive the round trip and the hooks.
    edge = 0.28 * (n / 2.0) ** (1.0 / 3.0)
    assert abs(after["contact_offset"] - (edge / 2 + 0.02)) < 1e-5 and abs(after["object_spawn_z"] - edge / 2) < 1e-5
    # The gate's override reaches the constructed config through apply_strategy_flags.
    assert after["stage_lift_on_pinch"] is True and after["lift_weights"] == [0.0, 0.0], after
    plain = r["plain"]["after"]
    assert plain["stage_lift_on_pinch"] is False and plain["lift_weights"] == [2.0, 15.0], plain


if __name__ == "__main__" and "--harness" in sys.argv:
    print(HARNESS_MARK + json.dumps(_run_harness()), flush=True)
