#!/usr/bin/env python3
"""Execute frozen same-source GPU smoke, then bounded declared fusion cohorts."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import shutil
import traceback

sys.path.insert(0,str(Path(__file__).parent))
import fusion_occlusion_campaign as CAMPAIGN
from sensor_campaign import unpack_runtime


def execute(protocol):
    protocol=Path(protocol);outer=json.loads(protocol.read_text());inputs=protocol.parent/'inputs'
    output=Path(os.environ['H34_OUTPUT_DIR']);work=Path(os.environ['H34_WORK_DIR'])
    output.mkdir(parents=True,exist_ok=True);work.mkdir(parents=True,exist_ok=True)
    result={'schema':'bhl-fusion-occlusion-gated-campaign-v1','status':'INCOMPLETE','phase_order':['smoke','dev','test'],
        'completed_phases':[],'completed_cells':[],'scored_cells_started':0,'ground_truth_inputs':[],
        'scientific_status':'UNRUN','outer_protocol_sha256':CAMPAIGN.sha256_file(protocol)}
    CAMPAIGN.write(output/'campaign_result.json',result)
    try:
        for name,spec in outer['input_files'].items():
            if CAMPAIGN.sha256_file(inputs/name)!=spec['sha256']:raise ValueError('pinned input changed: '+name)
        inner=inputs/'fusion-protocol.json';plan=CAMPAIGN.read_plan(inner)
        result['scientific_protocol_sha256']=CAMPAIGN.sha256_file(inner)
        runtime=unpack_runtime(inputs/'ort.whl',outer['input_files']['ort.whl']['sha256'],work/'ort')
        sys.path.insert(0,str(work/'ort'))
        os.environ['FUSION_MODEL_PATH']=str(inputs/'model.onnx')
        result['onnx_runtime']=runtime
        smoke_receipt=None
        for phase in ('smoke','dev','test'):
            directories=[]
            for cell in (c for c in plan['cells'] if c['phase']==phase):
                if phase!='smoke':result['scored_cells_started']+=1
                destination=output/phase/cell['id']
                record=CAMPAIGN.run(inner,cell['id'],destination,smoke_receipt)
                directories.append(destination);result['completed_cells'].append(cell['id'])
                if record['status']!='PASS':raise RuntimeError('cell execution incomplete')
                payload=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
                if payload>plan['design']['maximum_campaign_bytes']:raise RuntimeError('aggregate retained payload exceeds2GiB')
                CAMPAIGN.write(output/'campaign_result.json',result)
            collection=output/(phase+'-collection.json')
            summary=CAMPAIGN.collect(inner,phase,directories,collection)
            result['completed_phases'].append({'phase':phase,'status':summary['status'],
                'verified_cells':summary['verified_cells'],'readiness_gate':summary['readiness_gate'],
                'collection':collection.name,'sha256':CAMPAIGN.sha256_file(collection)})
            if summary['status']!='PASS':raise RuntimeError('independent phase collection failed')
            if phase=='smoke':
                if not summary['readiness_gate']:
                    result.update(status='NEGATIVE',scientific_status='NEGATIVE_GEOMETRIC_SMOKE',follow_on='UNRUN_AFTER_NEGATIVE_SMOKE')
                    CAMPAIGN.write(output/'campaign_result.json',result);return 0
                smoke_receipt=collection
        payload=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
        if shutil.disk_usage(CAMPAIGN.DURABLE.parent).free < plan['design']['minimum_share_free_bytes'] + payload:
            raise RuntimeError('insufficient8GiBreserve before raw archive writeback')
        result.update(status='PASS',scientific_status='COMPLETE_PREDECLARED_SIMULATION_COMPARISON',
            acceptance_meaning='execution and independent collection complete; improvement requires full accuracy/coverage/recall tradeoff analysis')
    except Exception as error:
        result.update(status='INCOMPLETE',scientific_status='EXECUTION_OR_COLLECTION_INCOMPLETE',error=f'{type(error).__name__}: {error}')
        traceback.print_exc()
    CAMPAIGN.write(output/'campaign_result.json',result)
    return 0 if result['status'] in ('PASS','NEGATIVE') else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol',type=Path,required=True)
    raise SystemExit(execute(parser.parse_args().protocol))
