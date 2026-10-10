#!/usr/bin/env python3
"""Remove directory entries from a verified native runtime, preserving file bytes."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): value.update(block)
    return value.hexdigest()


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--source-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--receipt',type=Path,required=True)
    args=parser.parse_args()
    if sha(args.source)!=args.source_sha256 or args.output.exists() or args.receipt.exists():
        raise ValueError('verified original archive and new destinations required')
    files={}
    with tarfile.open(args.source) as source, tarfile.open(args.output,'w:gz') as target:
        names=set()
        for member in source.getmembers():
            path=Path(member.name)
            if path.is_absolute() or '..' in path.parts or member.name in names or not(member.isfile() or member.isdir()):
                raise ValueError('unsafe or duplicate original runtime entry')
            names.add(member.name)
            if member.isdir(): continue
            with source.extractfile(member) as stream:
                raw=stream.read()
            files[member.name]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
            import io
            target.addfile(member,io.BytesIO(raw))
    with tarfile.open(args.output) as check:
        members=check.getmembers()
        if any(not member.isfile() for member in members) or {m.name for m in members}!=set(files):
            raise ValueError('regular-file inventory differs after repack')
        for member in members:
            with check.extractfile(member) as stream:
                if hashlib.sha256(stream.read()).hexdigest()!=files[member.name]['sha256']:
                    raise ValueError('payload bytes changed during repack')
    receipt={'schema':'bhl-runtime-file-only-repack-v1','status':'PASS','original_archive_sha256':args.source_sha256,
             'archive_sha256':sha(args.output),'archive_bytes':args.output.stat().st_size,'files':files,
             'scope':'Packaging-only removal of directory headers; every file payload remains byte-identical, including native binary, libraries, settings and runtime receipt'}
    args.receipt.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({key:value for key,value in receipt.items() if key!='files'}),flush=True)


if __name__=='__main__':main()
