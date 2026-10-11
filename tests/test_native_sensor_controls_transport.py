"""Operational transport must preserve science and reject corrupt evidence."""
import importlib.util
import io
import json
from pathlib import Path
import tarfile

import pytest

SPEC = importlib.util.spec_from_file_location(
    'native_transport_under_test', Path(__file__).parents[1] / 'scripts/bench/native_sensor_controls_transport.py')
transport = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(transport)


def archive_at(path, rows):
    with tarfile.open(path, 'w:gz') as archive:
        for name, data in rows:
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    return path


def test_packaging_override_preserves_all_scientific_fields():
    original = dict(name='job', args=['--seed', '730001'], env={'RATE': '30'},
                    resources={'cpus': 2}, max_output_mb=2048, entrypoint='frozen.py')
    changed = transport.packaging_job(original, 8*transport.GIB)
    assert changed['max_output_mb'] == 8192
    assert original['max_output_mb'] == 2048
    changed.pop('max_output_mb')
    assert changed == {k: v for k, v in original.items() if k != 'max_output_mb'}
    changed['args'].append('extra')
    assert original['args'] == ['--seed', '730001']
    with pytest.raises(ValueError):
        transport.packaging_job(original, 9*transport.GIB)


@pytest.mark.parametrize('rows', [
    [('../escape', b'a')], [('/absolute', b'a')],
    [('same', b'a'), ('same', b'b')], [('a/../escape', b'a')],
])
def test_output_extractor_refuses_unsafe_paths_and_duplicates(tmp_path, rows):
    archive = archive_at(tmp_path/'raw.tar.gz', rows)
    with pytest.raises(ValueError):
        transport.extract_output(archive, tmp_path/'output')


def test_output_extractor_rejects_links(tmp_path):
    path = tmp_path/'links.tar.gz'
    with tarfile.open(path, 'w:gz') as archive:
        member = tarfile.TarInfo('link')
        member.type, member.linkname = tarfile.SYMTYPE, '/etc/passwd'
        archive.addfile(member)
    with pytest.raises(ValueError):
        transport.extract_output(path, tmp_path/'out')


def test_output_budget_is_explicit_and_enforced_before_writing(tmp_path):
    path = archive_at(tmp_path/'raw.tar.gz', [('a', b'123'), ('nested/b', b'456')])
    with pytest.raises(ValueError, match='budget'):
        transport.extract_output(path, tmp_path/'small', raw_cap_bytes=5)
    assert not list((tmp_path/'small').iterdir())
    assert transport.extract_output(path, tmp_path/'okay', raw_cap_bytes=6) == 6
    assert (tmp_path/'okay/nested/b').read_bytes() == b'456'
    with pytest.raises(FileExistsError):
        transport.extract_output(path, tmp_path/'okay')


class FakeRelease:
    REPOSITORY = 'test/repo'

    def __init__(self):
        self.assets = {}
        self.downloads = []

    def gh(self, *args):
        if args[0] == 'api':
            return json.dumps({'assets': [dict(name=name, size=len(data),
                               browser_download_url='https://example.test/'+name)
                               for name, data in self.assets.items()]})
        if args[1] == 'upload':
            source = Path(args[3])
            assert source.name not in self.assets
            self.assets[source.name] = source.read_bytes()
            return b''
        if args[1] == 'download':
            name = args[args.index('--pattern')+1]
            dest = Path(args[args.index('--dir')+1]) / name
            dest.write_bytes(self.assets[name])
            self.downloads.append(name)
            return b''
        raise AssertionError(args)


def test_multipart_round_trip_fresh_downloads_and_whole_hash(tmp_path):
    raw = archive_at(tmp_path/'raw.tar.gz', [('nested/episode.json', b'{"seed":730000}')])
    remote = FakeRelease()
    publication = transport.publish(raw, 'test-raw', remote, part_bytes=40)
    assert len(publication['parts']) > 1
    assert len(remote.downloads) == len(publication['parts'])
    restored = transport.download(publication, tmp_path/'download', remote)
    assert restored.read_bytes() == raw.read_bytes()
    assert len(remote.downloads) == 2*len(publication['parts'])
    transport.extract_output(restored, tmp_path/'out')
    assert json.loads((tmp_path/'out/nested/episode.json').read_text())['seed'] == 730000


def test_corrupt_publication_part_is_rejected(tmp_path):
    raw = archive_at(tmp_path/'raw.tar.gz', [('episode', b'original')])
    remote = FakeRelease()
    publication = transport.publish(raw, 'test', remote, part_bytes=40)
    name = publication['parts'][0]['asset_name']
    remote.assets[name] = b'x' * len(remote.assets[name])
    with pytest.raises(ValueError, match='identity differs'):
        transport.download(publication, tmp_path/'download', remote)


def test_conflicting_release_asset_is_not_overwritten(tmp_path):
    original = tmp_path/'raw'
    original.write_bytes(b'original')
    remote = FakeRelease()
    remote.assets['conflict'] = b'wrong length'
    with pytest.raises(ValueError, match='replacement refused'):
        transport.publish_file(original, 'conflict', remote)
    assert remote.assets['conflict'] == b'wrong length'


def test_legacy_verified_asset_remains_supported(tmp_path):
    raw = archive_at(tmp_path/'legacy.tar.gz', [('episode', b'old raw')])
    publication = dict(asset_name=raw.name, bytes=raw.stat().st_size,
                       sha256=transport.digest(raw), local_verified_mirror=str(raw))
    restored = transport.download(publication, tmp_path/'download', FakeRelease())
    assert restored.read_bytes() == raw.read_bytes()


def test_interrupted_download_retries_without_poisoning_final_name(tmp_path, monkeypatch):
    raw = archive_at(tmp_path/'raw.tar.gz', [('episode', b'kept exactly')])
    remote = FakeRelease()
    publication = transport.publish(raw, 'test', remote, part_bytes=40)
    original = remote.gh
    failures = []

    def interrupted(*args):
        if args[:2] == ('release', 'download') and not failures:
            failures.append(True)
            name = args[args.index('--pattern')+1]
            (Path(args[args.index('--dir')+1]) / name).write_bytes(b'partial')
            raise RuntimeError('temporary transfer failure')
        return original(*args)

    remote.gh = interrupted
    monkeypatch.setattr(transport.time, 'sleep', lambda _: None)
    restored = transport.download(publication, tmp_path/'download', remote)
    assert failures and restored.read_bytes() == raw.read_bytes()
    assert not list((tmp_path/'download').glob('assemble-*'))


def test_exhausted_transfer_keeps_final_archive_absent(tmp_path, monkeypatch):
    raw = archive_at(tmp_path/'raw.tar.gz', [('episode', b'original')])
    remote = FakeRelease()
    publication = transport.publish(raw, 'test', remote, part_bytes=40)

    def failing(*args):
        raise RuntimeError('network unavailable')

    remote.gh = failing
    monkeypatch.setattr(transport.time, 'sleep', lambda _: None)
    with pytest.raises(RuntimeError, match='network unavailable'):
        transport.download(publication, tmp_path/'download', remote)
    assert not (tmp_path/'download'/(publication['sha256']+'.tar.gz')).exists()
