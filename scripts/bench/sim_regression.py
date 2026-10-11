"""Build a portable, hash-verified real-policy MuJoCo replay artifact.

Uses a caller-specified trained checkpoint and upstream asset checkout. No
checkpoint download, dummy policy, Isaac dependency or GPU is used. Run the
generated replay.py in a fresh output directory on CPU, locally or in CI.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_bundle(args):
    from omegaconf import OmegaConf
    from bhl_robust.eval.mjcf_assets import prepare_mjcf

    cfg = OmegaConf.load(args.deploy)
    if (cfg.num_actions, cfg.num_joints, cfg.num_observations, cfg.history_length) != (12, 12, 45, 0):
        raise ValueError("this replay artifact currently supports the 12-DoF, 45-observation, no-history biped")
    checkpoint = args.checkpoint or Path(cfg.policy_checkpoint_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"actual checkpoint required: {checkpoint}; use --checkpoint to relocate it")
    if args.seconds <= 1.04:
        raise ValueError("seconds must leave time after the frozen 1-second fault trigger")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = {}
    files = ["src/bhl_robust/__init__.py", "src/bhl_robust/eval/harness.py",
             "src/bhl_robust/eval/terrain_field.py", "src/bhl_robust/eval/replay_gate.py"]
    for name in files:
        dst = out / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / name, dst)
        source[name] = digest(REPO / name)
    for name in ("src/bhl_robust/eval/__init__.py", "src/berkeley_humanoid_lite_lowlevel/__init__.py",
                 "src/berkeley_humanoid_lite_lowlevel/policy/__init__.py"):
        p = out / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("")
    upstream_controller = args.upstream / "source/berkeley_humanoid_lite_lowlevel/berkeley_humanoid_lite_lowlevel/policy/rl_controller.py"
    dst = out / "src/berkeley_humanoid_lite_lowlevel/policy/rl_controller.py"
    shutil.copyfile(upstream_controller, dst)
    source["upstream_rl_controller.py"] = digest(upstream_controller)
    for root, label in ((REPO, "overlay"), (args.upstream, "upstream")):
        for license_name in ("LICENSE", "LICENSE.md", "LICENSE.txt"):
            if (root / license_name).is_file():
                shutil.copyfile(root / license_name, out / f"LICENSE-{label}.txt")
                break
    shutil.copyfile(checkpoint, out / "policy.onnx")
    cfg.policy_checkpoint_path = "policy.onnx"
    OmegaConf.save(cfg, out / "deploy.yaml")
    with tempfile.TemporaryDirectory(prefix="bhl-portable-scene-") as tmp:
        scene = prepare_mjcf(args.upstream, Path(tmp))
        robot = scene.parent / "berkeley_humanoid_lite_biped.xml"
        scene_out = out / "scene"
        (scene_out / "meshes").mkdir(parents=True)
        shutil.copyfile(scene, scene_out / scene.name)
        robot_xml = robot.read_text()
        mesh_root = Path(re.search(r'meshdir="([^"]+)"', robot_xml).group(1))
        for name in sorted(set(re.findall(r'<mesh\b[^>]*file="([^"]+)"', robot_xml))):
            # Copy only assets referenced by this biped, preserving loadable XML.
            src = mesh_root / name
            dst = scene_out / "meshes" / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
        robot_xml = re.sub(r'meshdir="[^"]+"', 'meshdir="meshes"', robot_xml)
        (scene_out / robot.name).write_text(robot_xml)
    launcher = """from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'src'))
from bhl_robust.eval.replay_gate import main
if __name__ == '__main__':
    raise SystemExit(main())
"""
    (out / "replay.py").write_text(launcher)
    installed_packages = {k: importlib.metadata.version(k) for k in ("numpy", "mujoco", "onnxruntime", "omegaconf", "torch")}
    # Torch is a transitive import of the credited upstream controller, even
    # though this runner uses ONNX CPU inference. Record the exact installed
    # build in the receipt, allowing a portable CPU wheel of the same release.
    packages = {k: v.split("+")[0] if k == "torch" else v for k, v in installed_packages.items()}
    (out / "requirements.txt").write_text("\n".join(f"{k}=={v}" for k, v in packages.items()) + "\n")
    shutil.copyfile(REPO / "scripts/bench/verify_replay_bundle.py", out / "verify.py")
    (out / "USAGE.txt").write_text("""Portable trained-biped CPU replay and timing artifact
====================================================
Python 3.11. Install requirements.txt; torch is imported by the upstream
controller but CUDA/torch inference is not used. A CPU torch wheel is suitable.

From this directory (REPLAY_PY is your Python 3.11 interpreter):
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
    "$REPLAY_PY" replay.py --bundle . --out /tmp/bhl-replay-new --calls 10000
  "$REPLAY_PY" verify.py --bundle . --out /tmp/bhl-replay-check-new.json

CI entry point is the same command. Assets are mandatory: absent, changed or
wrong-version inputs fail before physics. Output directories cannot be reused.
All source/model/MJCF/mesh files are in this artifact, with relative paths and
hashes in manifest.json. No access to the builder's workspace is required.

Five frozen seeds, each repeated nominally and with eight faults. A regression
is a difference from the matched nominal trace or a validated input/scoring
error; it does not imply unsafe behavior or certify robustness. Timing excludes
physics, model load, networking and OS real-time scheduling, and tests the
40 ms policy period rather than the 4 ms lower-level controller period.

Upstream Berkeley Humanoid Lite retains its own controller and assets credit.
This overlay contributes packaging, repeatability/integrity gates, fault
scenarios, raw per-physics-substep contact checks and measured timing reports.
""")
    manifest = {"schema": "bhl-replay-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
                "packages": packages, "builder_installed_packages": installed_packages,
                "seconds": args.seconds, "policy_dt": float(cfg.policy_dt),
                "scenarios": [{"seed": 94000 + i, "command": cmd} for i, cmd in enumerate(
                    ([.3, 0., 0.], [.3, 0., .2], [.2, 0., -.2], [.1, .1, 0.], [0., 0., .5]))],
                "source": {"overlay_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                           "source_sha256": source, "builder_sha256": digest(Path(__file__)),
                           "source_deploy_sha256": digest(args.deploy), "source_checkpoint_sha256": digest(checkpoint),
                           "upstream_credit": "Berkeley Humanoid Lite Project Developers; see LICENSE-upstream.txt"},
                "files_sha256": {str(p.relative_to(out)): digest(p) for p in sorted(out.rglob("*")) if p.is_file()}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"bundle": str(out), "files": len(manifest["files_sha256"]),
                      "bytes": sum(p.stat().st_size for p in out.rglob("*") if p.is_file()),
                      "checkpoint_sha256": digest(out / "policy.onnx")}))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deploy", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=2.0)
    return build_bundle(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
