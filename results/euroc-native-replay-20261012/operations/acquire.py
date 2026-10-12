import urllib.request,pathlib,json,hashlib,time,datetime,zipfile,stat
root=pathlib.Path(__file__).resolve().parent
expected='a920fe5b5e69a6ad19b32f1cfaf90ac2f59d45dfd1ca18cc9722e07684ba45fa'
url='https://huggingface.co/datasets/pepijn223/euroc-mirror/resolve/d05bd6a92a5f456562153f83d65805fea27cb57d/V1_01_easy.zip'
archive=root/'V1_01_easy.zip'; cap=12*1024**3
receipt={'schema':'bhl-euroc-intake-v1','start_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'url':url,'mirror_revision':'d05bd6a92a5f456562153f83d65805fea27cb57d','mirror_expected_sha256':expected,'mirror_expected_bytes':1149702102,'combined_budget_bytes':cap,'official_master_byte_identity':'NOT VERIFIED; official host HTTP429','published_original_dataset_data':False}
h=hashlib.sha256();n=0;start=time.monotonic()
with urllib.request.urlopen(url,timeout=45) as r,archive.with_suffix('.partial').open('xb') as f:
 receipt['response_headers']=dict(r.headers)
 while True:
  b=r.read(1024**2)
  if not b:break
  n+=len(b)
  if n>1149702102:raise ValueError('Archive exceeds pinned mirror size')
  if time.monotonic()-start>1800:raise TimeoutError('Download deadline')
  f.write(b);h.update(b)
  if n%(128*1024**2)==0:print(n,flush=True)
if n!=1149702102 or h.hexdigest()!=expected:raise ValueError('Download integrity mismatch')
archive.with_suffix('.partial').rename(archive)
receipt.update(bytes=n,sha256=h.hexdigest(),download_seconds=time.monotonic()-start)
with zipfile.ZipFile(archive) as z:
 members=z.infolist();total=sum(i.file_size for i in members)
 receipt.update(zip_members=len(members),zip_uncompressed_bytes=total,zip_root_names=sorted(set(i.filename.split('/')[0] for i in members)))
 if total+n>cap:raise ValueError('Download and extraction exceeds cap')
 for i in members:
  p=pathlib.PurePosixPath(i.filename)
  if p.is_absolute() or '..' in p.parts or stat.S_ISLNK(i.external_attr>>16) or '\\' in i.filename:raise ValueError('Unsafe ZIP member')
 if len({i.filename for i in members})!=len(members):raise ValueError('Duplicate ZIP member')
 dest=root/'dataset';dest.mkdir(exist_ok=False)
 for i in members:
  if i.is_dir():continue
  target=dest/i.filename;target.parent.mkdir(parents=True,exist_ok=True)
  with z.open(i) as f,target.open('xb') as out:
   while True:
    b=f.read(1024**2)
    if not b:break
    out.write(b)
receipt.update(status='MIRROR_DOWNLOAD_HASH_AND_ZIP_CRC_VERIFIED',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
root.joinpath('intake.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
