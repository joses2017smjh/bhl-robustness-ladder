"""Deferred continuation rejects wrong cohort, unsafe archives and stale pins."""
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import pytest

PATH=Path(__file__).resolve().parents[1]/'scripts/bench/sensor_followup_deferred.py'
spec=importlib.util.spec_from_file_location('sensor_followup_deferred_tests',PATH)
D=importlib.util.module_from_spec(spec);spec.loader.exec_module(D)


def tar(path, name, *, kind=tarfile.REGTYPE):
    with tarfile.open(path,'w:gz') as t:
        m=tarfile.TarInfo(name);m.type=kind;m.linkname='another' if kind!=tarfile.REGTYPE else ''
        if kind==tarfile.REGTYPE:m.size=4
        t.addfile(m,io.BytesIO(b'data') if kind==tarfile.REGTYPE else None)


@pytest.mark.parametrize('name,kind',[('../escape',tarfile.REGTYPE),('/absolute',tarfile.REGTYPE),
    ('replay/link',tarfile.SYMTYPE),('replay/hard',tarfile.LNKTYPE)])
def test_extract_rejects_links_and_escape(tmp_path,name,kind):
    p=tmp_path/'bad.tar.gz';tar(p,name,kind=kind)
    with pytest.raises(ValueError,match='regular'):D.extract_regular(p,tmp_path/'out')


def test_extract_only_original_replay_namespace(tmp_path):
    p=tmp_path/'raw.tar.gz'
    with tarfile.open(p,'w:gz') as t:
        for name in ['mapping/a','replay/inference/x']:
            m=tarfile.TarInfo(name);m.size=4;t.addfile(m,io.BytesIO(b'data'))
    assert D.extract_regular(p,tmp_path/'out',prefix='replay/')==4
    assert not (tmp_path/'out/mapping').exists()
    assert (tmp_path/'out/replay/inference/x').read_bytes()==b'data'


def test_download_rejects_wrong_digest(tmp_path,monkeypatch):
    def fake(*args):
        dest=Path(args[args.index('--dir')+1]);(dest/'a.tar.gz').write_bytes(b'changed');return b''
    monkeypatch.setattr(D,'gh',fake)
    with pytest.raises(ValueError,match='differs'):
        D.download(dict(asset_name='a.tar.gz',bytes=7,sha256='0'*64),tmp_path/'download')


def source(tmp_path):
    root=tmp_path/'sensor';root.mkdir()
    protocol=root/'protocol.json';protocol.write_text('{}')
    (root/'intake.json').write_text(json.dumps(dict(archive_sha256='source-pin')))
    spec=dict(job_name='sensor-methods-run-00-20261010',group=D.GROUPS[0])
    (root/(spec['job_name']+'-submission.json')).write_text(json.dumps(dict(status='SUBMITTED',archive_sha256='source-pin',job_id=123)))
    job=root/(spec['job_name']+'-123');job.mkdir()
    result=dict(phase='run',cell=dict(geometry_group=D.GROUPS[0],condition='nominal'),capture=dict(frames=150,manifest_sha256='manifest-pin'))
    (job/'campaign_result.json').write_text(json.dumps(result))
    completion=dict(status='PASS',exit_status=0,error=None,archive_sha256='source-pin',files={
        'campaign_result.json':dict(sha256=D.digest(job/'campaign_result.json')),
        'outputs.tar.gz':dict(sha256='raw-pin')})
    (job/'completion.json').write_text(json.dumps(completion))
    publication=dict(schema='bhl-methods-archive-v1',sha256='raw-pin',source_archive_sha256='source-pin')
    (job/'publication.json').write_text(json.dumps(publication))
    plan=dict(sensor_campaign=str(root),sensor_protocol_sha256=D.digest(protocol),sensor_archive_sha256='source-pin')
    return plan,spec,job,result,completion


def test_source_cell_requires_full_nominal_frozen_lineage(tmp_path):
    plan,spec,job,result,completion=source(tmp_path)
    assert D.source_cell(plan,spec)[1]=='manifest-pin'
    result['phase']='smoke';(job/'campaign_result.json').write_text(json.dumps(result))
    completion['files']['campaign_result.json']['sha256']=D.digest(job/'campaign_result.json')
    (job/'completion.json').write_text(json.dumps(completion))
    with pytest.raises(ValueError,match='full nominal'):D.source_cell(plan,spec)


def test_source_cell_refuses_modified_result_even_if_scheduler_passed(tmp_path):
    plan,spec,job,result,_=source(tmp_path)
    result['capture']['frames']=149;(job/'campaign_result.json').write_text(json.dumps(result))
    with pytest.raises(ValueError,match='receipt differs'):D.source_cell(plan,spec)


def test_source_cell_refuses_mismatched_published_archive(tmp_path):
    plan,spec,job,_,_=source(tmp_path)
    p=json.loads((job/'publication.json').read_text());p['sha256']='different'
    (job/'publication.json').write_text(json.dumps(p))
    with pytest.raises(ValueError,match='publication lineage'):D.source_cell(plan,spec)


def test_plan_requires_unchanged_self_and_exact_matrices(tmp_path,monkeypatch):
    container=tmp_path/'container';container.write_bytes(b'container')
    plan=dict(schema='bhl-sensor-followup-deferred-v1',durable_dir=str(tmp_path/'durable'),
        native_faults=D.NATIVE_FAULTS,fusion_faults=D.FUSION_FAULTS,
        inputs=[dict(group=g) for g in D.GROUPS],maximum_seconds=3600,
        dispatcher_sha256=D.digest(PATH),container=str(container),container_sha256=D.digest(container))
    monkeypatch.setattr(D,'AUTHORIZED',tmp_path)
    p=tmp_path/'plan.json';p.write_text(json.dumps(plan));D.validate_plan(p,D.digest(p))
    plan['native_faults']=D.NATIVE_FAULTS[:-1];p.write_text(json.dumps(plan))
    with pytest.raises(ValueError,match='continuation plan'):D.validate_plan(p,D.digest(p))
