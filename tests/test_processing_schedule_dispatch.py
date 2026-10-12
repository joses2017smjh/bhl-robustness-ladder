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
        self.assertEqual(len(jobs), 26)
        self.assertEqual(jobs[0]['kind'], 'evaluate')
        self.assertEqual(jobs[1]['kind'], 'smoke')
        identities = set()
        for job in jobs[2:]:
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


if __name__ == '__main__':
    unittest.main()
