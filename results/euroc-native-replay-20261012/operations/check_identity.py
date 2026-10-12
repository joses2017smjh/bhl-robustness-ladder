import pathlib,sys,json,hashlib,urllib.request,re
import numpy as np,yaml
root=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(root/'source/scripts/bench'))
from euroc_native_validation import camera_rows,rectification,checked_transform
cameras=[camera_rows(root/f'dataset/mav0/cam{c}/data.csv') for c in (0,1)]
upstream=[int(x) for x in (root/'references/V101.txt').read_text().split()]
assert sorted(cameras[0])==sorted(cameras[1])==upstream
configs=[yaml.safe_load((root/f'dataset/mav0/{n}/sensor.yaml').read_text()) for n in ['cam0','cam1','imu0']]
ref=(root/'references/EuRoC.yaml').read_text()
checks={}
for i,c in enumerate(configs[:2],1):
 for j,n in enumerate(['fx','fy','cx','cy']):
  expected=float(re.search(r'^Camera'+str(i)+r'\.'+n+r': ([^\n]+)',ref,re.M).group(1));assert c['intrinsics'][j]==expected
 for j,n in enumerate(['k1','k2','p1','p2']):
  expected=float(re.search(r'^Camera'+str(i)+r'\.'+n+r': ([^\n]+)',ref,re.M).group(1));assert c['distortion_coefficients'][j]==expected
ref_values=[float(x) for x in re.search(r'Stereo.T_c1_c2:.*?data: \[([^]]+)\]',ref,re.S).group(1).split(',')]
t=np.linalg.inv(checked_transform(configs[0]))@checked_transform(configs[1])
assert np.allclose(t,np.array(ref_values).reshape(4,4),atol=1e-11,rtol=0)
receipt={'schema':'bhl-euroc-reference-identity-v1','status':'PASS','all_camera_timestamps_match_ORB_SLAM3_V101':True,'timestamps':len(upstream),'first_timestamp_ns':upstream[0],'last_timestamp_ns':upstream[-1],'both_camera_intrinsics_and_distortion_match_upstream':True,'camera_extrinsics_match_upstream':True,'upstream_reference_receipts':{p.name:json.loads(p.read_text()) for p in sorted((root/'references').glob('*.receipt.json'))},'limitation':'Independent metadata and timestamp identity established; image bytes authenticated to public mirror LFS hash, not inaccessible ETH master.'}
(root/'reference-identity.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
