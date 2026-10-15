"""Operational gates for immutable paired schedule execution."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location('processing_schedule_dispatch',
    Path(__file__).parents[1] / 'scripts/bench/processing_schedule_dispatch.py')
D = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(D)


class ProcessingScheduleDispatchTests(unittest.TestCase):
    def test_pair_inventory_contains_every_cell_exactly_once(self):
        jobs = D.declared_jobs(D.SHARED_ROOT / 'unit-campaign')
        self.assertEqual(len(jobs), 27)
        self.assertEqual(jobs[0]['kind'], 'evaluate')
        self.assertEqual(jobs[1]['kind'], 'smoke')
        identities = set()
        for job in jobs[2:-1]:
            a = job['args']
            identity = tuple(a[a.index(k) + 1] for k in ('--seed', '--condition', '--case', '--arm'))
            self.assertNotIn(identity, identities)
            identities.add(identity)
            self.assertEqual(job['resources'], jobs[0]['resources'])
            self.assertEqual(job['max_output_mb'], 2048)
        self.assertEqual(len(identities), 24)
        self.assertEqual(identities, {(str(seed), condition, case, arm)
            for seed in D.SEEDS for condition in D.CONDITIONS for case in D.CASES for arm in D.ARMS})

    def test_smoke_requires_full_new_cohort_and_true_gate(self):
        good = dict(status='PASS', phase='smoke', episodes=12, execution_gate=True)
        D.smoke_gate(good)
        for change in ({'status': 'NEGATIVE'}, {'phase': 'development'}, {'episodes': 11},
                       {'episodes': 14}, {'execution_gate': 1}, {'execution_gate': False}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                D.smoke_gate({**good, **change})

    def test_quota_uses_hard_limit_and_preserves_soft_grace(self):
        response = '/nfs/hpc/share 1700000000* 1610612736 2147483648 29d 2489844 0 0 -\n'
        with mock.patch.object(D.subprocess, 'run', return_value=mock.Mock(stdout=response)):
            row = D.quota_receipt(100 * D.GIB)
        self.assertGreater(row['used_bytes'], row['soft_limit_bytes'])
        self.assertEqual(row['hard_limit_bytes'], 2 * 1024**4)
        self.assertEqual(row['grace'], '29d')

    def test_quota_rejects_insufficient_hard_headroom_or_changed_limit(self):
        for response in ('/nfs/hpc/share 2140000000 1610612736 2147483648 - 1 0 0 -\n',
                         '/nfs/hpc/share 10000 1610612736 3000000000 - 1 0 0 -\n'):
            with self.subTest(response=response), mock.patch.object(D.subprocess, 'run',
                return_value=mock.Mock(stdout=response)), self.assertRaises(ValueError):
                D.quota_receipt(D.GIB)

    def test_directory_cannot_escape_authorized_root(self):
        for path in ('/tmp/test', D.SHARED_ROOT, D.SHARED_ROOT / '..' / 'other'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                D.bounded_directory(path)

    def test_verified_completion_rejects_changed_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            D.write(p / 'campaign_result.json', {'status': 'PASS', 'episodes': 2})
            D.write(p / 'launch.json', {'archive_sha256': 'frozen'})
            D.write(p / 'completion.json', {'archive_sha256': 'frozen', 'exit_status': 0,
                'error': None, 'status': 'PASS', 'files': {'campaign_result.json': {
                    'bytes': (p / 'campaign_result.json').stat().st_size,
                    'sha256': D.digest(p / 'campaign_result.json')}}})
            D.checked_completion(p, 'frozen')
            (p / 'campaign_result.json').write_text(json.dumps({'status': 'PASS', 'episodes': 48}))
            with self.assertRaises(ValueError):
                D.checked_completion(p, 'frozen')

    def test_compact_collector_evidence_keeps_hash_chain_and_excludes_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output, destination = root / 'output', root / 'compact'
            episode = output / 'test-episode'
            episode.mkdir(parents=True)
            fields = {}
            for name, field in (('inference_sensors.npz', 'inference_sensors_sha256'),
                    ('evaluator_truth.npz', 'evaluator_truth_sha256'),
                    ('native/stream_receipt.json', 'stream_receipt_sha256'),
                    ('sensors/processing_schedule.json', 'processing_schedule_sha256')):
                path = episode / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'retained raw data')
                fields[field] = D.digest(path)
            (episode / 'sensors' / 'unneeded.png').write_bytes(b'image bytes')
            D.write(episode / 'episode.json', fields)
            result = dict(episode_files=['test-episode/episode.json'],
                          episode_sha256={'test-episode/episode.json': D.digest(episode / 'episode.json')})
            D.write(output / 'campaign_result.json', result)
            inventory = D.retain_collector_inputs(output, destination, result)
            self.assertEqual(len(inventory), 6)
            self.assertFalse((destination / 'test-episode/sensors/unneeded.png').exists())
            self.assertEqual(D.digest(destination / 'test-episode/inference_sensors.npz'), fields['inference_sensors_sha256'])
            (episode / 'inference_sensors.npz').write_bytes(b'changed raw data')
            with self.assertRaises(ValueError):
                D.retain_collector_inputs(output, root / 'corrupt-copy', result)

    def test_all_gated_pair_retains_declared_unrun_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / 'output'
            output.mkdir()
            result = dict(episode_files=[], episode_sha256={}, episodes=0,
                          unrun_cells=[{'pipeline': 'legacy_cumulative'}, {'pipeline': 'bounded_latest'}])
            D.write(output / 'campaign_result.json', result)
            inventory = D.retain_collector_inputs(output, root / 'compact', result)
            self.assertEqual(set(inventory), {'campaign_result.json'})
            self.assertEqual(D.read(root / 'compact/campaign_result.json'), result)

    def test_scratch_retirement_preserves_evidence_on_archive_mismatch(self):
        with tempfile.TemporaryDirectory(prefix='bhl-h34-test-') as tmp:
            p = Path(tmp)
            (p / 'output').mkdir()
            (p / 'output' / 'evidence.json').write_text('{}')
            (p / 'outputs.tar.gz').write_text('corrupt')
            with self.assertRaises(ValueError):
                D.retain_and_retire_scratch(p, {'scratch': str(p)},
                    {'files': {'outputs.tar.gz': {'sha256': 'not-matching'}}})
            self.assertTrue((p / 'output' / 'evidence.json').exists())


    # --- design-derived constants (v1 reproducibility and order confirmation) ---

    def test_v1_jobs_scope_and_sbatch_reproduce_published_receipts(self):
        evidence = Path(__file__).parents[1] / 'results/processing-schedule-20261012'
        launch, submission = D.read(evidence / 'launch-protocol.json'), D.read(evidence / 'submission.json')
        campaign = D.SHARED_ROOT / 'processing-schedule-v1-20261012'
        self.assertEqual(D.declared_jobs(campaign), launch['jobs'])
        self.assertEqual(D.declared_jobs(campaign, D.V1_DESIGN), launch['jobs'])
        self.assertEqual(D.design_spec(D.V1_DESIGN)['scope'], launch['scientific_scope'])
        self.assertNotIn('design', launch)
        self.assertEqual(D.protocol_design(launch), D.V1_DESIGN)
        self.assertEqual(D.sbatch_command(campaign, submission['archive_sha256']), submission['command'])
        self.assertEqual(D.design_spec(D.V1_DESIGN)['resources'], submission['resources'])

    def test_design_spec_counts(self):
        v1, order = D.design_spec(D.V1_DESIGN), D.design_spec(D.ORDER_DESIGN)
        self.assertEqual((v1['shards'], v1['cells_per_shard'], v1['development_cells'], v1['smoke_cells'], v1['warmup_episodes']),
                         (24, 2, 48, 12, 0))
        self.assertEqual((order['shards'], order['cells_per_shard'], order['development_cells'], order['smoke_cells'],
                          order['warmup_episodes'], order['warmups_per_shard']), (24, 4, 96, 6, 50, 2))
        self.assertEqual(v1['cap_bytes'], 100 * D.GIB)
        self.assertEqual(order['cap_bytes'], 48 * D.GIB)
        with self.assertRaises(ValueError):
            D.design_spec('unknown-design')
        with self.assertRaises(ValueError):
            D.protocol_design({'design': 'unknown-design'})

    def test_order_design_jobs_fit_launcher_and_pin_rtx8000(self):
        campaign = D.SHARED_ROOT / 'unit-order-campaign'
        jobs = D.declared_jobs(campaign, D.ORDER_DESIGN)
        self.assertEqual(len(jobs), 27)
        self.assertEqual([j['kind'] for j in jobs[:2]] + [jobs[-1]['kind']], ['evaluate', 'smoke', 'evaluate'])
        launcher = D.helper(Path(D.__file__).with_name('h34_campaign.py'))
        launcher.validate_protocol(dict(schema_version=1, campaign_id='unit-order-campaign', design=D.ORDER_DESIGN,
            runtime=dict(python=D.PYTHON, share_root=str(D.SHARED_ROOT), sif='/nfs/hpc/share/sanchej7/Humanoid_Lite/container/bhl.sif'),
            jobs=jobs))
        identities = set()
        for job in jobs[2:-1]:
            a = job['args']
            self.assertEqual(job['kind'], 'run')
            self.assertNotIn('--case', a)
            identity = tuple(a[a.index(k) + 1] for k in ('--seed', '--condition', '--arm'))
            self.assertNotIn(identity, identities)
            identities.add(identity)
            self.assertEqual(job['name'], 'order-s{}-{}-{}'.format(*identity))
            self.assertEqual(a[a.index('--qualified-smoke') + 1], str(campaign / 'qualified-smoke.json'))
        self.assertEqual(identities, {(str(s), c, a) for s in D.ORDER_SEEDS for c in D.CONDITIONS for a in D.ORDER_ARMS})
        for job in jobs:
            self.assertEqual(job['resources'], D.ORDER_RESOURCES)
        self.assertEqual(D.ORDER_RESOURCES['constraint'], 'rtx8000')
        self.assertEqual((D.ORDER_RESOURCES['cpus'], D.ORDER_RESOURCES['memory_gb'], D.ORDER_RESOURCES['gpus']), (2, 16, 1))
        command = D.sbatch_command(campaign, 'f' * 64, D.ORDER_DESIGN)
        self.assertIn('--constraint=rtx8000', command)
        self.assertIn('--partition=gpu', command)
        self.assertIn('--job-name=schedule-order-96', command)
        self.assertFalse(any(x.startswith('--exclude') for x in command))
        self.assertEqual(command[-3:], [str(campaign / 'processing_schedule.sbatch'), str(campaign), 'f' * 64])

    def test_stage_acceptance_derives_from_design(self):
        shard = dict(phase='development', expected_declared_cells=4, episodes=4, unrun_cells=[], design=D.ORDER_DESIGN,
                     warmup_cells=[{}, {}])
        D.check_stage_result(2, {'kind': 'run'}, shard, D.ORDER_DESIGN)
        gated = dict(shard, episodes=2, unrun_cells=[{}, {}])
        D.check_stage_result(2, {'kind': 'run'}, gated, D.ORDER_DESIGN)
        for change in ({'expected_declared_cells': 2}, {'episodes': 3}, {'warmup_cells': [{}]}, {'warmup_cells': []},
                       {'design': D.V1_DESIGN}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                D.check_stage_result(2, {'kind': 'run'}, {**shard, **change}, D.ORDER_DESIGN)
        v1_pair = dict(phase='development', expected_declared_cells=2, episodes=2, unrun_cells=[])
        D.check_stage_result(2, {'kind': 'run'}, v1_pair, D.V1_DESIGN)
        with self.assertRaises(ValueError):
            D.check_stage_result(2, {'kind': 'run'}, v1_pair, D.ORDER_DESIGN)
        collection = dict(schema='bhl-processing-schedule-v1-collection', declared_cells=96, measured_cells=96,
                          unrun_after_nominal_gate=0, design=D.ORDER_DESIGN)
        D.check_stage_result(26, {'kind': 'evaluate'}, collection, D.ORDER_DESIGN)
        for change in ({'declared_cells': 48}, {'measured_cells': 95}, {'design': D.V1_DESIGN}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                D.check_stage_result(26, {'kind': 'evaluate'}, {**collection, **change}, D.ORDER_DESIGN)
        smoke = dict(status='PASS', phase='smoke', episodes=6, execution_gate=True, design=D.ORDER_DESIGN, warmup_cells=[{}, {}])
        D.smoke_gate(smoke, D.ORDER_DESIGN)
        D.check_stage_result(1, {'kind': 'smoke'}, smoke, D.ORDER_DESIGN)
        for change in ({'episodes': 12}, {'warmup_cells': []}, {'design': D.V1_DESIGN}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                D.smoke_gate({**smoke, **change}, D.ORDER_DESIGN)
        with self.assertRaises(ValueError):
            D.smoke_gate(smoke)

    def test_collector_inputs_keep_warmup_timing_receipts_but_no_warmup_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / 'output'
            warm = output / 'warmup' / 'warmup-legacy'
            fields = {}
            for name, field in (('native/stream_receipt.json', 'stream_receipt_sha256'),
                                ('sensors/processing_schedule.json', 'processing_schedule_sha256')):
                path = warm / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'warm-up receipt ' + name.encode())
                fields[field] = D.digest(path)
            D.write(warm / 'episode.json', dict(fields, inference_sensors_sha256='not-archived'))
            relative = 'warmup/warmup-legacy/episode.json'
            result = dict(episode_files=[], episode_sha256={}, warmup_episode_files=[relative],
                          warmup_episode_sha256={relative: D.digest(warm / 'episode.json')})
            D.write(output / 'campaign_result.json', result)
            inventory = D.retain_collector_inputs(output, root / 'compact', result)
            self.assertEqual(set(inventory), {'campaign_result.json', relative,
                'warmup/warmup-legacy/native/stream_receipt.json', 'warmup/warmup-legacy/sensors/processing_schedule.json'})
            (warm / 'sensors/processing_schedule.json').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                D.retain_collector_inputs(output, root / 'compact-2', result)


if __name__ == '__main__':
    unittest.main()
