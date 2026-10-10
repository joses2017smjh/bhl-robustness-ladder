import json, os, subprocess, sys
from pathlib import Path
repo=Path('/tmp/bhl-methods-20261010')
py='/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python'
nodeids=json.loads(Path('/tmp/perceptive-review-history-retry-nodeids.json').read_text())
records=[]
cases=[('history-and-finalizer', [py,'-m','pytest','-q',*nodeids,'tests/test_methods_finalize.py','--junitxml=/tmp/perceptive-review-final-integration-20261010.xml'],0),('perceptive-real-intake', [py,'scripts/bench/perceptive_methods_collect.py','--campaign-dir','/nfs/stak/users/sanchej7/humanoid-methods-20261010/perceptive-v4','--output','/tmp/perceptive-review-real-intake-20261010.json'],1)]
for name,command,expected in cases:
    log=Path('/tmp/perceptive-review-'+name+'-20261010.log')
    with log.open('x') as stream:
        run=subprocess.run(command,cwd=repo,stdout=stream,stderr=subprocess.STDOUT,check=False)
    record={'case':name,'command':command,'exit_code':run.returncode,'expected_exit':expected,'log':str(log)}
    records.append(record)
    print(json.dumps(record),flush=True)
Path('/tmp/perceptive-review-final-integration-driver-20261010.json').write_text(json.dumps(records,indent=2)+'\n')
sys.exit(any(r['exit_code']!=r['expected_exit'] for r in records))
