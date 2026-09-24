"""SF-04 distillation recipe: the predeclared values in the cfg module, the
task registration and the sbatch/probe wiring, checked from source text and
AST so no Isaac import is needed (maze_distill.py imports isaaclab)."""
import ast
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODULE = REPO / "src/bhl_robust/tasks/maze_distill.py"
REGISTRY = REPO / "src/bhl_robust/tasks/__init__.py"
PROBE = REPO / "scripts/bench/maze_recovery_probe.py"
SBATCH = REPO / "slurm/repo20260923/gpu_sf04_distill.sbatch"
INNER = REPO / "slurm/repo20260923/inner_sf04_distill.sh"
TASK_ID = "Velocity-BHL-MazeRecovery-Full-BothRobustDistill-v0"


def _class(tree, name):
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"class {name} not found")


def _assigned(cls):
    out = {}
    for node in cls.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            out[node.targets[0].id] = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            out[node.target.id] = node.value
    return out


def _kw(call):
    return {k.arg: k.value for k in call.keywords}


def test_runner_cfg_predeclared_values():
    tree = ast.parse(MODULE.read_text())
    cls = _class(tree, "MazeStudentDistillCfg")
    assert any(getattr(b, "id", None) == "RslRlDistillationRunnerCfg" for b in cls.bases)
    a = _assigned(cls)
    assert ast.literal_eval(a["num_steps_per_env"]) == 24
    assert ast.literal_eval(a["max_iterations"]) == 2000
    assert ast.literal_eval(a["save_interval"]) == 100
    assert ast.literal_eval(a["experiment_name"]) == "biped"
    assert ast.literal_eval(a["obs_groups"]) == {"student": ["policy"], "teacher": ["teacher"]}
    student = _kw(a["student"])
    assert a["student"].func.id == "RslRlRNNModelCfg"
    assert ast.literal_eval(student["rnn_type"]) == "lstm"
    assert ast.literal_eval(student["rnn_hidden_dim"]) == 256
    assert ast.literal_eval(student["rnn_num_layers"]) == 1
    assert ast.literal_eval(student["obs_normalization"]) is False
    teacher = _kw(a["teacher"])
    assert a["teacher"].func.id == "RslRlMLPModelCfg"          # MLP, so the PPO actor_state_dict fits
    assert teacher["hidden_dims"].id == "TEACHER_HIDDEN_DIMS"
    assert ast.literal_eval(teacher["obs_normalization"]) is False
    consts = {n.targets[0].id: n.value for n in tree.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
    assert ast.literal_eval(consts["TEACHER_HIDDEN_DIMS"]) == [256, 128, 128]
    assert ast.literal_eval(consts["TEACHER_ACTIVATION"]) == "elu"
    assert ast.literal_eval(consts["TEACHER_RUN"]) == "2026-09-19_16-25-50_wknd-full-both-s0"
    assert ast.literal_eval(consts["TEACHER_CHECKPOINT"]) == "model_5997.pt"
    alg = _kw(a["algorithm"])
    assert ast.literal_eval(alg["gradient_length"]) == 24
    assert ast.literal_eval(alg["learning_rate"]) == 1.0e-3
    assert ast.literal_eval(alg["loss_type"]) == "mse"
    # The deprecated student/teacher cfg must not be used on rsl-rl 5.0.1.
    assert "RslRlDistillationStudentTeacherRecurrentCfg(" not in MODULE.read_text()
    # The cfg migrates itself because train_distill.py never calls the shim.
    post = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__post_init__"]
    assert post and "handle_deprecated_rsl_rl_cfg" in ast.unparse(post[0])


def test_observation_groups_and_env_cfg():
    tree = ast.parse(MODULE.read_text())
    obs = _assigned(_class(tree, "BothRobustDistillObsCfg"))
    assert ast.unparse(obs["policy"]) == "BothRobustObsCfg.PolicyCfg()"   # degraded student input
    assert ast.unparse(obs["teacher"]) == "BothObsCfg.PolicyCfg()"        # clean teacher target
    assert ast.unparse(obs["critic"]) == "BothObsCfg.CriticCfg()"
    env = _class(tree, "MazeBothRobustDistillEnvCfg")
    src = ast.unparse(env)
    assert "sf04_resample" in src and "stereo_l" in src and "projected_gravity" in src
    assert 'RECOVERY_CONFIGS["Velocity-BHL-MazeRecovery-Full-BothRobust-v0"]' in MODULE.read_text()
    doc = ast.get_docstring(tree)
    assert "teacher" in doc and "student" in doc and "OWN degraded" in doc


def test_registration_has_three_entry_points():
    text = REGISTRY.read_text()
    block = text[text.index(TASK_ID):]
    block = block[: block.index("except Exception")]
    for key in ("env_cfg_entry_point", "rsl_rl_cfg_entry_point", "rsl_rl_distillation_cfg_entry_point"):
        assert key in block, key
    assert "maze_distill.MazeBothRobustDistillEnvCfg" in block
    assert "maze_distill.MazeStudentDistillCfg" in block
    assert text.index("try:\n    from bhl_robust.tasks import maze_distill") < text.index(TASK_ID)


def test_probe_option_and_existing_path():
    text = PROBE.read_text()
    assert '"--runner", choices=("onpolicy", "distillation"), default="onpolicy"' in text
    assert 'if args.checkpoint and args.runner == "distillation":' in text
    assert "elif args.checkpoint:\n        from importlib.metadata import version" in text
    assert "DistillationRunner(" in text and "get_inference_policy" in text
    assert "return env, student.forward, obs" in text


def test_sbatch_and_inner_wiring():
    sb, inner = SBATCH.read_text(), INNER.read_text()
    assert "#SBATCH --time=04:00:00" in sb and "#SBATCH --gres=gpu:1" in sb and "#SBATCH --account=eecs" in sb
    assert "BHL_STACK=v60" in sb and "setup_node_cache" in sb and "v60_boot_gate" in sb
    for name in ("baseline", "delay1", "delay2", "lidar_off", "stereo_off", "both_off"):
        assert f'"name":"{name}"' in inner, name
    assert "--runner distillation" in inner and "--keep-corruption" in inner
    assert "--resume True" in inner and "--checkpoint \"$TEACHER_CKPT_NAME\"" in inner and "model_5997.pt" in inner
    assert "--enable_cameras" in inner and 'cd "$UPSTREAM"' in inner
    assert "unset BHL_POLICY" in sb and "unset BHL_POLICY" in inner
    assert "student-s0.json" in inner and "sf04-distill" in inner
    assert 'rule = {"baseline": 28, "delay1": 24, "lidar_off": 24, "stereo_off": 24}' in sb
    assert re.search(r'SF04-DISTILL RESULT: \{verdict\}', sb)
    for var in ("TASK", "LOAD_RUN", "RUN_NAME", "SEED", "NUM_ENVS", "MAX_ITER", "TRAIN_SCRIPT"):
        assert f"export {var}=" in sb, var
    assert "export RUN_NAME=sf04-distill-s0" in sb and "export MAX_ITER=2000" in sb and "export NUM_ENVS=1024" in sb
