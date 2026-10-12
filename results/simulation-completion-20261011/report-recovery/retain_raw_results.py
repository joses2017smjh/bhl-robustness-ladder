"""Retain only exact, collector-pinned result JSON from original worker scratch."""
import hashlib,json,pathlib,socket
ROOT=pathlib.Path('/nfs/hpc/share/sanchej7/humanoid-research-20261011')
DST=ROOT/'completion-report-observer-v2/raw-results'
controller=ROOT/'native-sensor-controls-dispatch-v3'
state=json.loads((controller/'state.json').read_text())
sources={pathlib.Path(r['path']).parent.name:r['sha256'] for r in json.loads((controller/'scientific-summary.json').read_text())['source_results']}
host=socket.gethostname().split('.')[0]
retained=[]
for job in state['jobs']:
    launch=json.loads((ROOT/'native-sensor-controls-v2'/(job['name']+'-'+job['job_id'])/'launch.json').read_text())
    if launch['host'].split('.')[0]!=host:continue
    source=pathlib.Path(launch['scratch'])/'output/campaign_result.json'
    if source.is_symlink() or not source.is_file() or source.stat().st_size>8*1024**2:raise ValueError('missing or oversized raw result: '+str(source))
    raw=source.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=sources[job['name']]:raise ValueError('raw result SHA differs: '+job['name'])
    destination=DST/(job['name']+'.json')
    if destination.exists():
        if destination.read_bytes()!=raw:raise ValueError('conflicting retained raw result')
    else:
        with destination.open('xb') as stream:stream.write(raw)
    retained.append({'name':job['name'],'raw_source':str(source),'sha256':sources[job['name']],'bytes':len(raw)})
receipt={'schema':'bhl-original-worker-result-retention-v1','host':host,'scientific_rerun':False,'retained':retained}
out=DST.parent/('retention-'+host+'.json')
with out.open('x') as stream:json.dump(receipt,stream,indent=2)
print(json.dumps({'host':host,'retained':len(retained),'bytes':sum(r['bytes'] for r in retained)}))
