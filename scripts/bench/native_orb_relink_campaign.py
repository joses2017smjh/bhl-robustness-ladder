#!/usr/bin/env python3
"""Finish a pinned actual failed-link build; never fabricate native poses."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--compiled-receipt',type=Path,required=True)
    a=p.parse_args(); receipt=json.loads(a.compiled_receipt.read_text())
    source=Path(__file__).resolve().parents[2];work=Path(os.environ['H34_WORK_DIR']);output=Path(os.environ['H34_OUTPUT_DIR'])
    original=Path(receipt['root']);scratch=original.parents[1];sif=Path(receipt['sif'])
    if sha(sif)!=receipt['sif_sha256']:raise ValueError('Pinned build SIF changed')
    runtime=work/'native-runtime';home=work/'container-home';home.mkdir()
    # Old source/sysroot is read-only. Only its existing CMake build directory
    # may write the new executable; unchanged source objects remain inventoried.
    cmd=['apptainer','exec','--containall','--cleanenv','--home',str(home),
         '--bind',receipt['share_root']+':'+receipt['share_root']+':ro',
         '--bind',str(scratch)+':'+str(scratch)+':ro',
         '--bind',str(original/'build')+':'+str(original/'build')+':rw',
         '--bind',str(work.parent)+':'+str(work.parent),
         '--bind',str(source)+':'+str(source)+':ro',
         str(sif),sys.executable,str(source/'scripts/native/orb_build.py'),
         '--work',str(original),'--output',str(runtime),'--jobs','2','--resume-compiled',str(a.compiled_receipt)]
    env={k:v for k,v in os.environ.items() if not k.startswith('APPTAINERENV_')};start=time.monotonic()
    with (output/'build.log').open('x') as log:process=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
    result={'route':'native_runtime_build','method':'orb','returncode':process.returncode,
            'wall_seconds':time.monotonic()-start,'command':cmd,'scientific_status':'BUILD_ONLY_NO_ESTIMATED_TRAJECTORIES',
            'source_build':receipt['source_build'],'reuse_scope':'Same genuine native source/compiled objects; non-multiarch GDAL library search repair'}
    try:
        native=json.loads((runtime/'runtime.json').read_text())
        if process.returncode or native['status']!='PASS':raise ValueError('Actual executable/ldd build smoke failed')
        shutil.copytree(runtime,output/'runtime')
        result.update(status='PASS',runtime_schema=native['schema'],runtime_files={
            str(f.relative_to(runtime)):{'sha256':sha(f),'bytes':f.stat().st_size} for f in runtime.rglob('*') if f.is_file()})
    except Exception as error:result.update(status='INCOMPLETE',error=str(error))
    (output/'campaign_result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='runtime_files'}),flush=True)
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
