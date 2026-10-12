"""Read retained cn-gpu7 outputs only; print a compact post hoc diagnostic."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket

import numpy as np

ROOT = Path('/nfs/hpc/share/sanchej7/humanoid-research-20261011')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


state_path = ROOT / 'native-sensor-controls-dispatch-v3/state.json'
binding_path = ROOT / 'completion-report-observer-v2/native-result-bindings.json'
summary_path = ROOT / 'native-sensor-controls-dispatch-v3/scientific-summary.json'
bindings = {row['name']: row for row in read(binding_path)['shards']}
host = socket.gethostname().split('.')[0]
if host != 'cn-gpu7':
    raise ValueError('This post hoc subset is explicitly limited to retained cn-gpu7 outputs')
rows = []
source_pins = {}
for job in read(state_path)['jobs']:
    directory = ROOT / 'native-sensor-controls-v2' / (job['name'] + '-' + job['job_id'])
    launch_path = directory / 'launch.json'
    launch = read(launch_path)
    if launch['host'].split('.')[0] != host:
        continue
    scratch = Path(launch['scratch'])
    output = scratch / 'output'
    result_path = output / 'campaign_result.json'
    result = read(result_path)
    assert digest(result_path) == bindings[job['name']]['raw_result_sha256']
    source_key = 'scripts/bench/native_sensor_controls_campaign.py'
    source_path = scratch / 'source' / source_key
    assert digest(source_path) == result['source_sha256'][source_key]
    source_pins[source_key] = digest(source_path)
    for relative in result['episode_files']:
        path = output / relative
        episode = read(path)
        assert digest(path) == result['episode_sha256'][relative]
        truth_path = path.parent / 'evaluator_truth.npz'
        assert digest(truth_path) == episode['evaluator_truth_sha256']
        stream_path = path.parent / 'native/stream_receipt.json'
        assert digest(stream_path) == episode['stream_receipt_sha256']
        native_path = path.parent / 'native/native_frames.jsonl'
        frames = [json.loads(line) for line in native_path.read_text().splitlines()]
        assert len(frames) == episode['native_frames']
        client_p95 = float(np.quantile([frame['client_wall_seconds'] for frame in frames], .95) * 1000)
        assert abs(client_p95 - episode['client_wall_p95_ms']) < 1e-8
        sensor_total = sum(frame['additional_sensor_compute_seconds'] for frame in frames)
        assert abs(sensor_total - episode['additional_sensor_compute_total_s']) < 1e-8
        traces = [row for row in episode['trace'] if row['timestamp_s'] >= 10.0001]
        errors = []
        first = None
        with np.load(truth_path, allow_pickle=False) as truth:
            times, poses = truth['timestamp_s'], truth['T_W_I']
            registration = np.asarray(episode['initial_registration']) if episode['initial_registration'] is not None else None
            for frame in frames:
                if frame['tracked'] and registration is not None:
                    index = int(np.argmin(abs(times - frame['timestamp_s'])))
                    estimate = registration @ np.asarray(frame['T_W_I'])
                    error = float(np.linalg.norm(estimate[:2, 3] - poses[index, :2, 3]))
                    errors.append(error)
                    if first is None:
                        first = {'capture_s': frame['timestamp_s'], 'xy_error_m': error}
        row = {key: episode[key] for key in (
            'method', 'condition', 'case', 'seed', 'success', 'client_wall_p95_ms', 'full_pipeline_p95_ms',
            'native_states', 'native_frames', 'native_tracked_frames', 'additional_sensor_compute_total_s',
            'dropped_scans', 'tracking_fraction_unique_estimator_updates')}
        row.update(
            episode_id=path.parent.name, shard_name=job['name'], original_job_id=job['job_id'],
            original_episode_path=str(path), episode_sha256=digest(path), launch_sha256=digest(launch_path),
            raw_campaign_result_sha256=digest(result_path), evaluator_truth_sha256=digest(truth_path),
            stream_receipt_sha256=digest(stream_path), retained_native_frames_sha256=digest(native_path),
            navigation_steps=len(traces), reason_counts=dict(Counter(step['reason'] for step in traces)),
            nonzero_command_steps=sum(bool(np.linalg.norm(step['command']) > 1e-8) for step in traces),
            final_goal_distance_m=traces[-1]['evaluator_goal_distance_m'],
            first_accepted_registration_diagnostic=first,
            tracked_xy_rmse_m=float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
            native_client_total_s=sum(frame['client_wall_seconds'] for frame in frames))
        rows.append(row)
assert len(rows) == 60 and len({row['episode_id'] for row in rows}) == 60
print(json.dumps({
    'schema': 'bhl-native-pipeline-posthoc-diagnosis-v1',
    'observed_utc': datetime.now(timezone.utc).isoformat(),
    'profiling_script_sha256': digest(__file__), 'host': host,
    'source_sha256': source_pins, 'original_source_archive_sha256': launch['archive_sha256'],
    'state_sha256': digest(state_path), 'scientific_summary_sha256': digest(summary_path),
    'raw_result_binding_manifest_sha256': digest(binding_path),
    'coverage': {'profiled_episodes': 60, 'completed_campaign_episodes': 200,
                 'selection': 'All six shards retained on cn-gpu7; post hoc accessibility subset, not random sampling'},
    'scope': 'Read-only diagnostics of original measured outputs; no simulation or estimator rerun, no parameter changes, no causal treatment effect',
    'timing_limit': 'Sensor bucket combines capture/render, raycasting, PNG encoding, hashing and callback bookkeeping; existing timing does not separate components',
    'native_trace_verification_limit': 'Retained native trace hash recorded and checked against episode frame count, client p95 and sensor total; raw release archive not redownloaded in this diagnostic',
    'pose_diagnostic_limit': 'Nearest recorded truth sample at acquisition time, evaluator-only, conditional on tracked frames; not new online feedback or full-horizon pose reliability',
    'rows': rows,
}, indent=2, allow_nan=False))
