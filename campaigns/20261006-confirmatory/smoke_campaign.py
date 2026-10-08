"""Unscored deployment equivalence and observation-contract check."""
from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from dr_confirmatory import validate
from navgym_confirmatory import write_new, validate_inputs


def main():
    campaign = Path(__file__).resolve().parent
    p = validate(campaign)
    validate_inputs(campaign)
    sys.path[:0] = [str(campaign / 'snapshot/src'), str(campaign / 'snapshot/upstream_lowlevel'),
                    str(campaign / 'snapshot/scripts/bench')]
    from omegaconf import OmegaConf
    import numpy as np
    from bhl_robust.eval.harness import EvalConfig, HeadlessMujocoEnv, run_episode
    from bhl_robust.eval.mjcf_assets import prepare_mjcf
    from berkeley_humanoid_lite_lowlevel.policy.rl_controller import RlController, OnnxPolicy
    from team_airlock import CpuPolicy
    differences = {}
    probes = np.random.default_rng(99001).normal(size=(10, 1, 45)).astype(np.float32)
    for actor in p['policies']:
        file = campaign / f"models/{actor['name']}/policy.onnx"
        original, single = OnnxPolicy(str(file)), CpuPolicy(file)
        differences[actor['name']] = max(float(np.max(np.abs(original.forward(x) - single.forward(x)))) for x in probes)
        if differences[actor['name']] > 1e-6:
            raise ValueError(f"inference mismatch: {actor['name']}")
    rollout = {}
    with tempfile.TemporaryDirectory(prefix='bhl-smoke-') as cache:
        scene = prepare_mjcf(Path(p['upstream']), Path(cache), 'biped')
        for actor in ('dr-off-s0', 'dr-default-s0'):
            cfg = OmegaConf.load(campaign / f'models/{actor}/deploy.yaml')
            env = HeadlessMujocoEnv(cfg, scene)
            controller = RlController(cfg)
            values = []
            times = []
            for backend in (OnnxPolicy, CpuPolicy):
                controller.policy = backend(cfg.policy_checkpoint_path)
                start = time.monotonic()
                values.append(asdict(run_episode(env, controller, (.3, 0, 0), 70999, EvalConfig())))
                times.append(time.monotonic() - start)
            if values[0] != values[1]:
                raise ValueError(f'backend changed physical rollout: {actor}')
            rollout[actor] = {'result': values[1], 'upstream_wall_s': times[0], 'single_thread_wall_s': times[1],
                              'bit_identical_episode_metrics': True}
    smoke = campaign / 'smoke'
    smoke.mkdir(exist_ok=False)
    # Independent pilot layout; no confirmatory seed touched here.
    out = smoke / 'navigation'
    command = [sys.executable, str(campaign / 'snapshot/scripts/bench/maze_explore.py'), '--upstream', p['upstream'],
               '--gait', str(campaign / 'models/gait/deploy.yaml'), '--policy', str(campaign / 'models/armV5-s8/actor.onnx'),
               '--policy-capture-pose', '--seeds', '1', '--seed-start', '73999', '--n', '6', '--m', '6',
               '--extra-openings', '1', '--time-limit', '30', '--sensor-mode', 'reactive_dropout',
               '--dropout-probability', '.35', '--no-overwrite', '--out-dir', str(out)]
    env = os.environ.copy()
    env.update(PYTHONPATH=os.pathsep.join(sys.path[:3]), PYTHONDONTWRITEBYTECODE='1')
    with tempfile.TemporaryDirectory(prefix='bhl-nav-smoke-') as cache:
        command += ['--cache-dir', cache]
        with (smoke / 'navigation.log').open('x') as log:
            process = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    if process.returncode:
        raise ValueError(f'navigation smoke failed with code {process.returncode}')
    episode = json.loads((out / 'seed73999.json').read_text())
    if (episode['sensor_mode'] != 'reactive_dropout' or episode['policy_map_integration']['mode'] != 'capture_pose'
            or episode['policy_visitation']['obs_keys'] != ['lidar', 'near', 'map_visit', 'map_coarse', 'goal']):
        raise ValueError('navigation observation contract mismatch')
    value = {'status': 'PASS', 'unscored_seeds': [70999, 73999], 'backend_max_abs_differences': differences,
             'physical_rollout_backend_equivalence': rollout, 'navigation_exit_code': process.returncode,
             'navigation_outcome_not_gated': episode['outcome'], 'navigation_sensor_stats': episode['sensor_stats']}
    write_new(smoke / 'verification.json', value)
    print(json.dumps(value, indent=2))


if __name__ == '__main__':
    main()
