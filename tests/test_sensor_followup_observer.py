"""Terminal publication never upgrades partial or negative science to success."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

PATH=Path(__file__).resolve().parents[1]/'scripts/bench/sensor_followup_observer.py'
spec=importlib.util.spec_from_file_location('sensor_followup_observer_tests',PATH)
O=importlib.util.module_from_spec(spec);spec.loader.exec_module(O)
GROUPS=[f'{f}-g{s}' for f in ('textured_boxes','thin_posts','ramp_step') for s in (610000,610001)]
NF=['nominal','imu_clock_plus20ms','imu_clock_plus80ms','lidar_extrinsic_yaw2deg_x2cm','gyro_z_bias_after5s','accel_x_bias_after5s','lidar_keep_half','lidar_keep_quarter']
FF=['nominal','lidar_extrinsic_yaw2deg_x2cm','lidar_keep_half','lidar_keep_quarter']


def fixture(tmp_path):
    parent=tmp_path/'parent';parent.mkdir()
    plan=dict(groups=GROUPS,native_faults=NF,fusion_faults=FF,parent_job_id='123',parent_dir=str(parent),
        science_source_archive_sha256='source',science_protocol_source_sha256={'code':'source-hash'},
        fault_specifications={f:{} for f in NF},gate={'minimum_tracking_fraction':.9},
        runtime_identity={'lio':dict(binary='fastlio_headless',binary_sha256='lio',receipt_sha256='lio-receipt'),
                          'livo':dict(binary='fastlivo_headless',binary_sha256='livo',receipt_sha256='livo-receipt')},
        sensor_source_archive_sha256='sensor-source')
    O.write(parent/'plan.json',{'source':{'sha256':'source'}});plan['parent_plan_sha256']=O.digest(parent/'plan.json')
    protocol=dict(schema='bhl-sensor-fault-protocol-v1',phase='run',source_sha256=plan['science_protocol_source_sha256'],
        faults=plan['fault_specifications'],methods=O.METHODS,gate=plan['gate'],
        runtimes={k:{**v,'path':'/tmp/old-runtime/'+k} for k,v in plan['runtime_identity'].items()},
        inputs=[dict(group=g,sequence=g+'-nominal',source_frames=150,manifest_sha256='capture-'+g) for g in GROUPS])
    O.write(parent/'full-protocol.json',protocol)
    for group in GROUPS:O.write(parent/'inputs'/(group+'.json'),dict(manifest_sha256='capture-'+group,publication={'source_archive_sha256':'sensor-source'}))
    native=dict(schema='bhl-sensor-fault-collection-v1',status='PASS',phase='run',protocol_sha256=O.digest(parent/'full-protocol.json'),
        expected_cells=48,verified_cells=48,missing=[],problems=[],paired_group_rows=[dict(group=g,fault=f,method=m,nominal_gate='PASS',fault_gate='NEGATIVE' if g==GROUPS[0] and f==NF[0] and m==O.METHODS[0] else 'PASS',tracking_fraction_delta=0.,conditional_ate_rmse_delta_m=None if g==GROUPS[0] else -.01) for g in GROUPS for f in NF for m in O.METHODS])
    fusion=dict(schema='bhl-stereo-consistency-collection-v1',status='PASS',phase='run',expected_cells=24,verified_cells=24,missing=[],problems=[],
        group_metrics=[dict(group=g,fault=f,arm=a,mae_m=None if a=='lidar_sparse' else .1,coverage=.1,near_obstacle_pixel_recall=None,unknown_obstacle_pixels=10,false_free_obstacle_pixels=1) for g in GROUPS for f in FF for a in O.ARMS])
    O.write(parent/'native-collection.json',native);O.write(parent/'fusion-collection.json',fusion)
    return parent,plan,native,fusion


def test_complete_comparison_preserves_negative_and_unavailable_cells(tmp_path):
    parent,plan,_,_=fixture(tmp_path);facts,issues=O.check_collections(parent,plan)
    assert issues==[]
    assert facts['native']['estimator_cells']==144 and facts['native']['negative_cells']==1
    assert facts['native']['conditional_ate_unavailable_cells']==24
    assert facts['fusion']['arm_cells']==120


@pytest.mark.parametrize('change',['missing','duplicate','smoke','wrong_source'])
def test_native_partial_or_forged_matrix_never_qualifies(tmp_path,change):
    parent,plan,native,_=fixture(tmp_path)
    if change=='missing':native['paired_group_rows'].pop()
    elif change=='duplicate':native['paired_group_rows'][-1]=native['paired_group_rows'][0]
    elif change=='smoke':native['phase']='smoke'
    else:native['protocol_sha256']='different'
    O.write(parent/'native-collection.json',native)
    facts,issues=O.check_collections(parent,plan)
    assert any(i.startswith('native collector') for i in issues) and 'native' not in facts


def test_fusion_duplicate_and_missing_hazard_accounting_rejected(tmp_path):
    parent,plan,_,fusion=fixture(tmp_path)
    fusion['group_metrics'][-1]=fusion['group_metrics'][0]
    O.write(parent/'fusion-collection.json',fusion)
    assert any(i.startswith('fusion collector') for i in O.check_collections(parent,plan)[1])
    fusion['group_metrics'][0].pop('false_free_obstacle_pixels')
    O.write(parent/'fusion-collection.json',fusion)
    assert any(i.startswith('fusion collector') for i in O.check_collections(parent,plan)[1])


def test_runtime_identity_cannot_change_under_matching_collector_counts(tmp_path):
    parent,plan,_,_=fixture(tmp_path)
    protocol=O.read(parent/'full-protocol.json');protocol['runtimes']['livo']['binary_sha256']='other'
    O.write(parent/'full-protocol.json',protocol)
    _,issues=O.check_collections(parent,plan)
    assert any('runtime identity' in i for i in issues)


def test_exact_73_raw_assets_must_match_live_server_hashes(tmp_path):
    parent,plan,_,_=fixture(tmp_path);assets=[]
    for name in O.expected_assets(plan):
        O.write(parent/'publications'/(name+'.json'),dict(asset_name=name,bytes=17,sha256='a'*64))
        assets.append(dict(name=name,size=17,digest='sha256:'+'a'*64))
    facts,issues=O.check_publications(parent,plan,assets)
    assert facts['verified']==facts['expected']==73 and issues==[]
    assets[0]['digest']='sha256:'+'b'*64
    facts,issues=O.check_publications(parent,plan,assets)
    assert facts['verified']==72 and len(issues)==1


def test_metadata_staging_excludes_symlink_escape_and_enforces_byte_cap(tmp_path):
    parent=tmp_path/'parent';parent.mkdir();external=tmp_path/'external';external.mkdir()
    (external/'secret.json').write_text('do not publish')
    (parent/'publications').symlink_to(external,target_is_directory=True)
    (parent/'state.json').write_text('{}')
    result=O.stage_metadata(parent,tmp_path/'staged',limit=2)
    assert result['bytes']==2 and 'publications/secret.json' in result['skipped']
    assert not (tmp_path/'staged/publications/secret.json').exists()


def test_incomplete_terminal_evidence_is_still_published_honestly(tmp_path,monkeypatch):
    parent,plan,_,_=fixture(tmp_path);root=tmp_path/'observer';root.mkdir()
    plan_path=root/'plan.json';O.write(plan_path,plan)
    def publish(bundle,name,receipts):
        summary=O.read(bundle/'terminal-summary.json')
        assert summary['status']=='INCOMPLETE'
        return dict(asset_name=name,sha256='verified',bytes=100,verification='fresh download')
    helper=SimpleNamespace(REPOSITORY='repo',TAG='tag',gh=lambda *args:b'{"assets":[]}',publish=publish)
    monkeypatch.setattr(O,'validate_bootstrap',lambda *args:(plan,root,helper))
    monkeypatch.setattr(O,'scheduler',lambda job:dict(job_id='123',state='FAILED',exit_code='1:0'))
    monkeypatch.setenv('SLURM_JOB_ID','456')
    result=O.main(plan_path,O.digest(plan_path))
    assert result['publication_status']=='PUBLISHED_AND_FRESH_DOWNLOAD_VERIFIED'
    assert result['scientific_status']=='INCOMPLETE'
    assert 'parent scheduler did not complete successfully' in result['summary']['issues']
