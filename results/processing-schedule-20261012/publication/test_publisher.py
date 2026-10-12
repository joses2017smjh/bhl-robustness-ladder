import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location('schedule_publisher', Path(__file__).with_name('publish.py'))
P = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P)


class PublisherTests(unittest.TestCase):
    def test_api_digest_verifies_without_raw_redownload(self):
        asset = {'size': 12, 'state': 'uploaded', 'digest': 'sha256:abc',
                 'browser_download_url': f'https://github.com/{P.REPO}/releases/download/{P.TAG}/test.tar.gz'}
        with mock.patch.object(P.urllib.request, 'urlopen', side_effect=AssertionError('unexpected network download')):
            self.assertIn('API', P.verify_remote(asset, 'abc', 12))

    def test_existing_different_digest_refuses_overwrite(self):
        asset = {'size': 12, 'state': 'uploaded', 'digest': 'sha256:different',
                 'browser_download_url': f'https://github.com/{P.REPO}/releases/download/{P.TAG}/test.tar.gz'}
        with self.assertRaises(ValueError):
            P.verify_remote(asset, 'abc', 12)

    def test_remote_repository_and_size_are_checked(self):
        for asset in [dict(size=9, state='uploaded', digest='sha256:abc', browser_download_url='https://example.com'),
                      dict(size=12, state='uploaded', digest='sha256:abc', browser_download_url='https://example.com')]:
            with self.assertRaises(ValueError):
                P.verify_remote(asset, 'abc', 12)

    def test_upload_uses_original_hardlink_then_removes_only_the_link(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'staging').mkdir()
            original = root / 'original.bin'
            original.write_bytes(b'original frozen evidence')
            sha, size = P.digest(original), original.stat().st_size
            remote = dict(id=9, name='unique.tar.gz', size=size, state='uploaded', digest='sha256:' + sha,
                browser_download_url=f'https://github.com/{P.REPO}/releases/download/{P.TAG}/unique.tar.gz')
            def fake_gh(*args, **kwargs):
                self.assertNotIn('--clobber', args)
                self.assertTrue(P.os.path.samefile(root / 'staging/unique.tar.gz', original))
            with mock.patch.object(P, 'OPS', root), mock.patch.object(P, 'release_assets', side_effect=[[], [remote]]), mock.patch.object(P, 'gh', side_effect=fake_gh):
                row = P.publish(original, 'unique.tar.gz', sha, size, stage='test')
            self.assertEqual(row['sha256'], sha)
            self.assertTrue(original.exists())
            self.assertFalse((root / 'staging/unique.tar.gz').exists())
            self.assertTrue((root / 'publication-manifest.jsonl').exists())

    def test_completed_stage_rejects_modified_original_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job = {'name': 'stage', 'kind': 'run'}
            directory = root / ('stage-' + P.SCIENTIFIC_JOB)
            directory.mkdir()
            artifact = directory / 'outputs.tar.gz'
            artifact.write_bytes(b'raw evidence')
            P.write(directory / 'campaign_result.json', dict(status='PASS'))
            P.write(directory / 'launch.json', dict(archive_sha256=P.SOURCE_SHA, job_id=P.SCIENTIFIC_JOB, job=job))
            files = {p.name: dict(sha256=P.digest(p), bytes=p.stat().st_size) for p in directory.iterdir() if p.is_file()}
            P.write(directory / 'completion.json', dict(archive_sha256=P.SOURCE_SHA, status='PASS', exit_status=0, error=None, files=files))
            with mock.patch.object(P, 'CAMPAIGN', root):
                self.assertIsNotNone(P.checked_stage(job))
                artifact.write_bytes(b'changed')
                with self.assertRaises(ValueError):
                    P.checked_stage(job)


if __name__ == '__main__':
    unittest.main()
