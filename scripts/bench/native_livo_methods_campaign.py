#!/usr/bin/env python3
"""Matched-input native FAST-LIO2 / FAST-LIVO2 / camera-ablation replay.

Smoke and scored runs must use a frozen protocol. All sensor variants are
recorded inputs; this runner never edits images, poses, or clouds at inference.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value


LIO = module('native_lio_method_baseline', 'scripts/native/lio_replay.py')
LIVO = module('native_livo_method_fusion', 'scripts/native/livo_replay.py')
SCORE = module('native_livo_method_scoring', 'scripts/bench/native_slam_campaign.py')


def run(manifest_path, lio_runtime, livo_runtime, protocol, output):
    plan = json.loads(Path(protocol).read_text())
    if plan.get('schema') != 'bhl-native-livo-methods-v1': raise ValueError('frozen LIVO-method protocol required')
    methods = plan['methods']; sequences = plan['sequences']; gate = plan['gate']
    if methods != ['fast_lio2', 'fast_livo2', 'fast_livo2_no_visual']:
        raise ValueError('all three native comparison arms must be predeclared')
    if len(sequences) != len(set(sequences)) or not sequences: raise ValueError('unique predeclared sequences required')
    manifest, root = SCORE.ADAPTER.read_manifest(manifest_path)
    available = {seq['id']: seq for seq in manifest['sequences']}
    if not set(sequences) <= set(available): raise ValueError('declared sequence is unavailable')
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    rows = []
    for sequence in sequences:
        for method in methods:
            destination = output / sequence / method; destination.parent.mkdir(exist_ok=True)
            begin = time.monotonic()
            if method == 'fast_lio2':
                measured = LIO.replay(manifest_path, sequence, lio_runtime, destination)
            else:
                measured = LIVO.replay(manifest_path, sequence, livo_runtime, destination,
                                       visual_enabled=method != 'fast_livo2_no_visual')
            native = [json.loads(line) for line in (destination / 'native_frames.jsonl').read_text().splitlines()]
            score = SCORE.evaluate(native, method='lio', replay_root=root, manifest=manifest,
                                   sequence=available[sequence], gate=gate)
            record = {'method': method, 'sequence': sequence, 'native': measured, 'evaluation': score,
                      'wall_seconds': time.monotonic() - begin}
            SCORE.write_json(destination / 'evaluation.json', record); rows.append(record)
    result = {'schema': 'bhl-native-livo-method-results-v1', 'status': 'PASS',
              'scientific_status': 'MEASURED_NATIVE_REPLAY_COMPARISON', 'closed_loop_episodes': 0,
              'protocol_sha256': SCORE.sha256(protocol), 'manifest_sha256': SCORE.sha256(manifest_path),
              'method_cells': len(rows), 'rows': rows,
              'comparison_scope': 'Same calibrated sensor bundle; FAST-LIO2 emits LiDAR-tail poses, FAST-LIVO2 emits camera-time poses. Fixed-scale gravity/yaw/translation alignment and bracketed evaluator truth. Tracking coverage is reported alongside conditional pose error.',
              'visual_contribution_ablation': 'FAST-LIVO2 camera+LiDAR+IMU versus its same-binary LiDAR+IMU arm; existing FAST-LIO2 has a different native map estimator.',
              'timing_scope': 'Native compute only unless explicitly named decode-and-compute or full-process wall; capture cost excluded.',
              'all_cells_qualified': all(r['evaluation']['replay_readiness_gate'] == 'PASS' for r in rows)}
    SCORE.write_json(output / 'campaign_result.json', result)
    lines = ['# Native tightly coupled fusion replay', '',
             'Native FAST-LIVO2 uses the left camera, timed 3D LiDAR and IMU. Right stereo images are not fused.', '',
             '| Sequence | Method | Tracked | Visual measurement frames | ATE RMSE (m) | Native p95 (ms) | Gate |',
             '|---|---|---:|---:|---:|---:|---|']
    for row in rows:
        metrics = row['evaluation']['segments'][0]['metrics']
        ate = f"{metrics['ate_translation_m']['rmse']:.5f}" if metrics else 'unavailable'
        visual = row['native'].get('frames_with_visual_measurements', 'n/a')
        lines.append(f"| {row['sequence']} | {row['method']} | {row['native']['tracked_frames']}/{row['native']['frames']} | {visual} | {ate} | {row['native']['native_compute_p95_ms']:.3f} | {row['evaluation']['replay_readiness_gate']} |")
    lines.extend(['', result['comparison_scope'], '', result['visual_contribution_ablation'], '',
                  'Qualification is separate from successful execution. No closed-loop or physical-hardware improvement is inferred.'])
    (output / 'report.md').write_text('\n'.join(lines) + '\n')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True, type=Path); p.add_argument('--lio-runtime', required=True, type=Path)
    p.add_argument('--livo-runtime', required=True, type=Path); p.add_argument('--protocol', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    result = run(args.manifest, args.lio_runtime, args.livo_runtime, args.protocol, args.output)
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2))


if __name__ == '__main__': main()
