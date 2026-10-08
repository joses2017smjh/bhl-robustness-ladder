"""Freeze CPU locomotion and navigation confirmation without new prose files."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
REPO = Path('/nfs/stak/users/sanchej7/hpc-share/Humanoid_Lite/bhl-robustness-ladder')
UPSTREAM = REPO / 'external/Berkeley-Humanoid-Lite'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def copy_python(source, target):
    for file in source.rglob('*.py'):
        destination = target / file.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, destination)


def main():
    from omegaconf import OmegaConf
    if (ROOT / 'protocol.json').exists() or (ROOT / 'snapshot').exists():
        raise FileExistsError('refusing to overwrite campaign inputs')
    # Excludes all .md/.txt and caches: only executable Python sources copied.
    copy_python(REPO / 'src', ROOT / 'snapshot/src')
    copy_python(REPO / 'scripts/bench', ROOT / 'snapshot/scripts/bench')
    copy_python(UPSTREAM / 'source/berkeley_humanoid_lite_lowlevel/berkeley_humanoid_lite_lowlevel',
                ROOT / 'snapshot/upstream_lowlevel/berkeley_humanoid_lite_lowlevel')
    exports = UPSTREAM / 'logs/rsl_rl/biped'
    policies = []
    for rung in ('dr-off', 'dr-s0.5', 'dr-default', 'dr-s1.5', 'dr-aggressive'):
        for seed in range(3):
            name = f'{rung}-s{seed}'
            found = list(exports.glob(f'*_{name}/exported/deploy.yaml'))
            if len(found) != 1:
                raise ValueError(f'{name}: expected one final deploy, found {found}')
            cfg = OmegaConf.load(found[0])
            source_onnx = Path(cfg.policy_checkpoint_path)
            destination = ROOT / 'models' / name
            destination.mkdir(parents=True)
            shutil.copy2(source_onnx, destination / 'policy.onnx')
            # Only path relocation; physical/controller settings remain exact.
            cfg.policy_checkpoint_path = str(destination / 'policy.onnx')
            OmegaConf.save(cfg, destination / 'deploy.yaml')
            policies.append({'name': name, 'rung': rung, 'training_seed': seed,
                             'source_deploy': str(found[0]), 'source_sha256': digest(found[0]),
                             'onnx_sha256': digest(source_onnx)})
    actors = [f'armV5-s{s}' for s in (8, 9, 10)]
    for actor in actors:
        destination = ROOT / 'models' / actor
        destination.mkdir()
        shutil.copy2(REPO / f'results/navgym-v5-20261002/{actor}/actor.onnx', destination / 'actor.onnx')
    gait = ROOT / 'models/gait'
    gait.mkdir()
    source_gait = ROOT / 'models/dr-default-s0'
    shutil.copy2(source_gait / 'policy.onnx', gait / 'policy.onnx')
    cfg = OmegaConf.load(source_gait / 'deploy.yaml')
    cfg.policy_checkpoint_path = str(gait / 'policy.onnx')
    OmegaConf.save(cfg, gait / 'deploy.yaml')
    shutil.copy2(REPO / 'results/navgym-v5-transfer-20261002/verdict.json', ROOT / 'historical_navgym_transfer.json')
    shutil.copy2(REPO / 'results/navgym-v5-20261002/verdict_v5.json', ROOT / 'historical_navgym_training.json')
    assets = UPSTREAM / 'source/berkeley_humanoid_lite_assets/data/robots/berkeley_humanoid/berkeley_humanoid_lite'
    asset_files = [assets / f'mjcf/{f}' for f in ('bhl_biped_scene.xml', 'berkeley_humanoid_lite_biped.xml')]
    # Visual meshes are loaded even in headless runs; freeze all supplied meshes.
    asset_files += [f for f in (assets / 'meshes').rglob('*') if f.is_file()]
    versions = {package: importlib.metadata.version(package) for package in
                ('mujoco', 'numpy', 'onnxruntime', 'omegaconf', 'torch', 'gymnasium', 'opencv-python-headless', 'Pillow')}
    inputs = {str(f.relative_to(ROOT)): digest(f) for d in ('snapshot', 'models') for f in (ROOT / d).rglob('*') if f.is_file()}
    inputs.update({str(f): digest(f) for f in asset_files})
    inputs['navgym_confirmatory.py'] = digest(ROOT / 'navgym_confirmatory.py')
    inputs['dr_confirmatory.py'] = digest(ROOT / 'dr_confirmatory.py')
    import sys
    sys.path.insert(0, str(ROOT / 'snapshot/src'))
    from bhl_robust.eval import random_maze as rm
    maze_seeds = list(range(72000, 72048))
    layouts = {str(seed): hashlib.sha256(rm.world_xml(rm.generate(6, 6, seed, extra_openings=1), textured=False).encode()).hexdigest()[:16]
               for seed in maze_seeds}
    cells = [{'name': f'{actor}-{condition}', 'actor': actor, 'condition': condition,
              'sensor_mode': 'reactive' if condition == 'nominal' else 'reactive_dropout'}
             for actor in (*actors, 'astar') for condition in ('nominal', 'drop35')]
    common = {'frozen_utc': datetime.now(timezone.utc).isoformat(), 'upstream': str(UPSTREAM),
              'source_commit': subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip(),
              'input_sha256': inputs, 'package_versions': versions}
    write_new(ROOT / 'protocol.json', {**common, 'actors': actors, 'cells': cells, 'maze_seeds': maze_seeds,
              'layout_sha256': layouts, 'gait_sha256': digest(gait / 'policy.onnx'),
              'project_targets': {'nominal': {'min_goals': 40}, 'drop35': {'min_goals': 36}},
              'labels': {'learned': 'frozen Isaac PPO biped gait + final NavGym v5 PPO navigation actor',
                         'scripted': 'reactive speed brake; A* in matched reference cells only',
                         'oracle': 'pose and goal coordinate', 'scope': 'simulation confirmation, no hardware deployment'},
              'freshness': {'maze_seed_training_range': '0..9999', 'previous_v5_gym': '61000..61047',
                            'previous_v5_transfer': '63000..63011', 'new_confirmatory': '72000..72047'},
              'rules': 'Each of three final actors independently: >=40/48 nominal and >=36/48 dropout goals, zero falls. '
                       'Clean contacts reported separately. No post-hoc selection, parameter tuning, or seed deletion.'})
    write_new(ROOT / 'dr_protocol.json', {**common, 'policies': policies, 'reset_seeds': list(range(71000, 71020)),
              'commands': [[.3, 0, 0], [.5, 0, 0], [-.2, 0, 0], [0, .2, 0], [0, 0, .5], [.3, 0, .5]],
              'conditions': {'flat': {'push_speed': 0.0}, 'push04': {'push_speed': .4}},
              'settings': {'episode_s': 10.0, 'settle_s': 1.0, 'init_joint_noise': .02, 'init_vel_noise': .05,
                           'push_interval_s': 3.0, 'push_recovery_s': 1.5, 'terrain_difficulty': 0.0},
              'rule': 'Replication target on flat: all three dr-default policies zero falls; each dr-off policy has more '
                      'falls than its matched default. Tracking, commanded displacement and disturbed outcomes reported '
                      'for every rung, training seed and command; push04 descriptive, no promised success.'})
    print(json.dumps({'frozen_inputs': len(inputs), 'DR_episodes': 3600, 'navigation_episodes': 384,
                      'package_versions': versions}, indent=2))


if __name__ == '__main__':
    main()
