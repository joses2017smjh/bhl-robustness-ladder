"""Geometry/abstention/independent-collection failure tests without GPU mocks."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pytest
from bhl_robust.research.fusion_occlusion import ARMS, score, visibility_mask, evaluate_predictions
from bhl_robust.research.stereo_consistency import fuse

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('fusion_occlusion_campaign_test',ROOT/'scripts/bench/fusion_occlusion_campaign.py')
CAMPAIGN=importlib.util.module_from_spec(spec);spec.loader.exec_module(CAMPAIGN)


def test_abstaining_cannot_hide_missed_obstacles():
    truth=np.array([[1.,1.,1.,4.]],np.float32);obstacle=np.ones_like(truth,bool)
    result=score(np.array([[1.,2.,np.nan,4.]]),np.array([[True,True,False,True]]),truth,obstacle)
    assert result['eligible_pixels']==4 and result['coverage']==.75
    assert result['near_obstacle_pixels']==3 and result['missed_obstacle_pixels']==2
    assert result['false_free_obstacle_pixels']==1 and result['unknown_obstacle_pixels']==1
    assert result['obstacle_pixel_recall']==pytest.approx(1/3)


def test_empty_support_has_explicit_zero_denominator():
    result=score(np.array([[np.nan]]),np.array([[False]]),np.array([[1.]]),np.array([[True]]),support=np.array([[False]]))
    assert result['eligible_pixels']==0 and result['mae_m'] is None and result['coverage'] is None


def test_pairwise_support_not_limited_by_sparse_lidar():
    truth=np.ones((2,4),np.float32);full=np.ones_like(truth,bool);sparse=full.copy();sparse[:,1:]=False
    predictions={arm:(truth.copy(),sparse if arm=='lidar_sparse' else full.copy()) for arm in ARMS}
    result=evaluate_predictions(predictions,truth,full,np.zeros_like(full),np.zeros_like(full),full)
    assert result['arms']['sgbm']['all_arm_common']['eligible_pixels']==2
    assert result['pairwise_common']['sgbm_cost__vs__sgbm']['sgbm']['eligible_pixels']==8
    assert result['pairwise_common']['sgbm_cost__vs__lidar_sparse']['sgbm_cost']['eligible_pixels']==2


def test_visibility_parallax_is_geometry_not_bad_depth():
    left=np.full((1,6),2.,np.float32);right=np.full_like(left,2.);right[:,2]=1.
    occluded,inside=visibility_mask(left,right,10.,.2)
    assert not inside[0,0] and occluded[0,3] and occluded.sum()==1
    assert not visibility_mask(left,left,10.,0.)[0].any()


def test_outside_view_and_unknown_other_depth_not_scored_as_occlusion():
    left=np.full((1,3),2.);other=np.full_like(left,np.nan)
    occluded,inside=visibility_mask(left,other,10.,.2)
    assert not inside.any() and not occluded.any()


def test_conflict_gate_reports_coverage_loss():
    stereo=np.array([[2.,2.]],np.float32);lidar=np.array([[1.,2.]],np.float32);valid=np.ones_like(stereo,bool)
    simple,sv,_=fuse(stereo,valid,lidar,valid,np.ones_like(stereo),gated=False)
    gated,gv,conflict=fuse(stereo,valid,lidar,valid,np.ones_like(stereo),gated=True)
    assert sv.all() and simple[0,0]==1.5 and conflict[0,0]
    assert not gv[0,0] and np.isnan(gated[0,0])


def test_geometry_splits_and_occlusion_conditions_are_grouped():
    design=json.loads(CAMPAIGN.DESIGN.read_text());cells=CAMPAIGN.cells(design)
    assert len(cells)==30
    for phase,expected in [('smoke',6),('dev',12),('test',12)]:
        subset=[c for c in cells if c['phase']==phase];assert len(subset)==expected
        assert all(len([x for x in subset if x['group']==c['group']])==3 for c in subset)
    assert {c['group'] for c in cells if c['phase']=='dev'}.isdisjoint({c['group'] for c in cells if c['phase']=='test'})


def test_condition_changes_real_geometry_not_texture(tmp_path):
    design=json.loads(CAMPAIGN.DESIGN.read_text());texture=tmp_path/'texture.png'
    clear=ET.fromstring(CAMPAIGN.scene_xml('panel_background',741000,'clear',texture,design))
    partial=ET.fromstring(CAMPAIGN.scene_xml('panel_background',741000,'partial',texture,design))
    heavy=ET.fromstring(CAMPAIGN.scene_xml('panel_background',741000,'heavy',texture,design))
    assert clear.find("worldbody/geom[@name='foreground_occluder']") is None
    assert partial.find("worldbody/geom[@name='foreground_occluder']").get('size')!='.025 .24 .60'
    assert heavy.find("worldbody/geom[@name='foreground_occluder']").get('size')=='.025 0.24 .60'
    assert clear.find('asset/texture').attrib==partial.find('asset/texture').attrib==heavy.find('asset/texture').attrib
    assert clear.find("worldbody/geom[@name='target_0']").attrib==heavy.find("worldbody/geom[@name='target_0']").attrib


def test_scored_run_rejects_missing_or_negative_smoke(tmp_path):
    protocol=tmp_path/'protocol';protocol.write_text('{}')
    with pytest.raises(ValueError,match='GPU smoke'):CAMPAIGN.verify_smoke(protocol,None)
    result=tmp_path/'result';result.write_text(json.dumps({'status':'PASS','phase':'smoke','protocol_sha256':CAMPAIGN.sha256_file(protocol),'readiness_gate':False,'verified_cells':6}))
    with pytest.raises(ValueError,match='negative smoke'):CAMPAIGN.verify_smoke(protocol,result)


def test_pooled_error_weights_pixels_not_number_of_views():
    first=score(np.array([[2.,2.]]),np.array([[True,True]]),np.array([[1.,1.]]),np.array([[True,True]]))
    second=score(np.array([[1.,np.nan]]),np.array([[True,False]]),np.array([[1.,1.]]),np.array([[True,True]]))
    result=CAMPAIGN.pooled_scores([first,second]);assert result['mae_m']==pytest.approx(2/3)
    assert result['predicted_pixels']==3 and result['eligible_pixels']==4


def test_score_requires_boolean_visibility_masks():
    with pytest.raises(ValueError,match='boolean'):
        score(np.ones((1,1)),np.ones((1,1)),np.ones((1,1)),np.ones((1,1),bool))


def test_changed_protocol_design_rejected_before_execution(tmp_path):
    design=json.loads(CAMPAIGN.DESIGN.read_text());design['geometry_seeds']['test']=[741000]
    plan={'schema':'bhl-fusion-occlusion-protocol-v1','design':design,'cells':CAMPAIGN.cells(design),'arms':list(ARMS),'settings':CAMPAIGN.SETTINGS}
    path=tmp_path/'protocol.json';path.write_text(json.dumps(plan))
    with pytest.raises(ValueError,match='predeclared'):CAMPAIGN.read_plan(path)


def test_collection_cannot_pass_with_missing_declared_cells(tmp_path,monkeypatch):
    design=json.loads(CAMPAIGN.DESIGN.read_text());plan={'design':design,'cells':CAMPAIGN.cells(design)}
    monkeypatch.setattr(CAMPAIGN,'read_plan',lambda _:plan)
    protocol=tmp_path/'protocol.json';protocol.write_text('{}')
    result=CAMPAIGN.collect(protocol,'test',[],tmp_path/'collection.json')
    assert result['status']=='INCOMPLETE' and not result['readiness_gate']
    assert len(result['missing'])==12 and result['verified_cells']==0


def test_source_pin_tamper_is_rejected(tmp_path):
    design=json.loads(CAMPAIGN.DESIGN.read_text())
    plan={'schema':'bhl-fusion-occlusion-protocol-v1','design':design,'cells':CAMPAIGN.cells(design),'arms':list(ARMS),
          'settings':CAMPAIGN.SETTINGS,'source_sha256':{name:CAMPAIGN.sha256_file(ROOT/name) for name in CAMPAIGN.SOURCE_FILES}}
    plan['source_sha256'][CAMPAIGN.SOURCE_FILES[0]]='0'*64
    path=tmp_path/'protocol.json';path.write_text(json.dumps(plan))
    with pytest.raises(ValueError,match='source pin'):CAMPAIGN.read_plan(path)


def test_protocol_serialization_cannot_reorder_frozen_cells():
    design=json.loads(CAMPAIGN.DESIGN.read_text())
    sorted_roundtrip=json.loads(json.dumps(design,sort_keys=True))
    assert CAMPAIGN.cells(design)==CAMPAIGN.cells(sorted_roundtrip)


def test_serial_negative_smoke_never_starts_scored_cells(tmp_path,monkeypatch):
    serialspec=importlib.util.spec_from_file_location('fusion_serial_test',ROOT/'scripts/bench/fusion_occlusion_serial.py')
    serial=importlib.util.module_from_spec(serialspec);serialspec.loader.exec_module(serial)
    inputs=tmp_path/'inputs';inputs.mkdir();(inputs/'ort.whl').write_bytes(b'fixture')
    (inputs/'fusion-protocol.json').write_text('{}');protocol=tmp_path/'protocol.json'
    protocol.write_text(json.dumps({'input_files':{'ort.whl':{'sha256':CAMPAIGN.sha256_file(inputs/'ort.whl')}}}))
    output=tmp_path/'output';monkeypatch.setenv('H34_OUTPUT_DIR',str(output));monkeypatch.setenv('H34_WORK_DIR',str(tmp_path/'work'))
    plan={'design':{'maximum_campaign_bytes':2*1024**3},'cells':[{'id':p,'phase':p} for p in ['smoke','dev','test']]}
    monkeypatch.setattr(serial.CAMPAIGN,'read_plan',lambda _:plan)
    monkeypatch.setattr(serial,'unpack_runtime',lambda *args:{'status':'fixture'})
    calls=[]
    monkeypatch.setattr(serial.CAMPAIGN,'run',lambda protocol,cell,*args:(calls.append(cell) or {'status':'PASS'}))
    def negative_collect(protocol,phase,dirs,destination):
        result={'status':'PASS','readiness_gate':False,'verified_cells':1};serial.CAMPAIGN.write(destination,result);return result
    monkeypatch.setattr(serial.CAMPAIGN,'collect',negative_collect)
    assert serial.execute(protocol)==0
    result=json.loads((output/'campaign_result.json').read_text())
    assert result['status']=='NEGATIVE' and result['scored_cells_started']==0 and calls==['smoke']
    assert result['follow_on']=='UNRUN_AFTER_NEGATIVE_SMOKE'
