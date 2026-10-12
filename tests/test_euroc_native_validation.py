import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

BENCH = Path(__file__).resolve().parents[1] / "scripts/bench"
SPEC = importlib.util.spec_from_file_location("euroc_validation", BENCH / "euroc_native_validation.py")
MODULE = importlib.util.module_from_spec(SPEC)
ORIGINAL_PATH = sys.path.copy()
try:
    sys.path.insert(0, str(BENCH))
    SPEC.loader.exec_module(MODULE)
finally:
    sys.path[:] = ORIGINAL_PATH


def matrix(value):
    return dict(rows=4, cols=4, data=np.asarray(value).reshape(-1).tolist())


def calibration():
    left = np.eye(4)
    left[:3, 3] = [.03, -.02, .05]
    relative = np.eye(4)
    angle = .035
    relative[:3,:3] = [[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]]
    relative[:3,3] = [-.11,.002,-.001]
    right = left @ np.linalg.inv(relative)
    imu = np.eye(4)
    imu[:3,3] = [-.04, .02, .005]
    common = dict(camera_model="pinhole",distortion_model="radial-tangential",resolution=[752,480],
                  rate_hz=20,intrinsics=[460.,457.,374.,240.],distortion_coefficients=[-.1,.02,.001,-.002])
    return ({**common,"T_BS":matrix(left)}, {**common,"T_BS":matrix(right)},
            dict(T_BS=matrix(imu),rate_hz=200,gyroscope_noise_density=.00017,
                 accelerometer_noise_density=.002,gyroscope_random_walk=.000019,
                 accelerometer_random_walk=.003))


def test_radtan_rectification_equal_rows_positive_disparity():
    cv2=pytest.importorskip("cv2")
    cameras=calibration()
    _,stereo,_,_,receipt=MODULE.rectification(*cameras)
    points=np.array([[-.4,.1,2.],[.4,-.3,3.],[.2,.4,4.]])
    relative=np.linalg.inv(MODULE.checked_transform(cameras[1])) @ MODULE.checked_transform(cameras[0])
    right=(relative[:3,:3] @ points.T).T + relative[:3,3]
    pixels=[]
    for camera,values,side in zip(cameras[:2],(points,right),("left","right")):
        k=MODULE.camera_matrix(camera)
        d=np.array(camera["distortion_coefficients"])
        distorted,_=cv2.projectPoints(values,np.zeros(3),np.zeros(3),k,d)
        pixels.append(cv2.undistortPoints(distorted,k,d,R=np.array(receipt[f"R_{side}_rectification"]),
                                         P=np.array(receipt[f"P_{side}"])).reshape(-1,2))
    np.testing.assert_allclose(pixels[0][:,1],pixels[1][:,1],atol=1e-6)
    assert np.all(pixels[0][:,0] > pixels[1][:,0])
    assert stereo["baseline_m"] == pytest.approx(np.linalg.norm(relative[:3,3]))


def test_rectified_camera_to_nonidentity_imu_transform():
    pytest.importorskip("cv2")
    cameras=calibration()
    _,_,t_i_c,noise,receipt=MODULE.rectification(*cameras)
    p_rect=np.array([.2,-.1,3.,1.])
    unrect=np.eye(4)
    unrect[:3,:3]=np.array(receipt["R_left_rectification"]).T
    expected=MODULE.checked_transform(cameras[0]) @ unrect @ p_rect
    np.testing.assert_allclose(MODULE.checked_transform(cameras[2]) @ t_i_c @ p_rect,expected,atol=1e-12)
    assert noise["noise_gyro"] == cameras[2]["gyroscope_noise_density"]


@pytest.mark.parametrize("field,value",[("camera_model","fisheye"),("distortion_model","equidistant"),
    ("resolution",[512,512]),("rate_hz",30),("intrinsics",[-1.,400.,300.,200.]),
    ("distortion_coefficients",[float("nan"),0,0,0])])
def test_wrong_camera_contract_rejected(field,value):
    pytest.importorskip("cv2")
    cameras=copy.deepcopy(calibration())
    cameras[0][field]=value
    with pytest.raises(ValueError): MODULE.rectification(*cameras)


def test_bad_imu_transform_and_noise_rejected():
    pytest.importorskip("cv2")
    cameras=copy.deepcopy(calibration())
    cameras[2]["T_BS"]["data"][0]=2
    with pytest.raises(ValueError): MODULE.rectification(*cameras)
    cameras=copy.deepcopy(calibration())
    cameras[2]["accelerometer_noise_density"]=0
    with pytest.raises(ValueError): MODULE.rectification(*cameras)


def test_pixel_values_preserved_and_wrong_modes_rejected():
    image=np.zeros((480,752),dtype=np.uint8)
    image[0,:3]=[0,127,255]
    assert MODULE.original_image(image) is image
    with pytest.raises(ValueError): MODULE.original_image(image.astype(np.uint16))
    with pytest.raises(ValueError): MODULE.original_image(image.T)


def test_exact_nanosecond_prefix_without_future_or_duplicate_imu(tmp_path):
    origin=1403636579763555584
    stamps=[origin+i*50_000_000 for i in range(1300)]
    base,selected=MODULE.selected_timestamps(stamps,stamps+[origin-1])
    assert base == origin and len(selected)==1200 and selected[-1]-base==59_950_000_000
    csv=tmp_path/"imu.csv"
    csv.write_text("\n".join(f"{origin+i*5_000_000},1,2,3,4,5,6" for i in range(-5,12)))
    rows,_=MODULE.read_imu(csv,origin,origin+50_000_000)
    batches,receipt=MODULE.partition_imu(MODULE.relative_seconds(selected[:2],origin),rows)
    assert receipt["consumed_imu_samples"] == 15
    assert batches[0][-1][0] == 0 and batches[1][-1][0] == .05
    assert batches[1][0][0] == .005
    np.testing.assert_array_equal(np.asarray(batches[1])[:,1:],np.tile([1,2,3,4,5,6],(10,1)))


def test_csv_duplicate_and_path_escape_rejected(tmp_path):
    csv=tmp_path/"data.csv"
    csv.write_text("#timestamp,filename\n1,1.png\n1,1.png\n")
    with pytest.raises(ValueError): MODULE.camera_rows(csv)
    csv.write_text("1,../1.png\n")
    with pytest.raises(ValueError): MODULE.camera_rows(csv)
    with pytest.raises(ValueError): MODULE.bounded_path(tmp_path,"../outside")
    (tmp_path/"link").symlink_to(tmp_path/"data.csv")
    with pytest.raises(ValueError): MODULE.bounded_path(tmp_path,"link")


def synthetic_prefix(tmp_path):
    cv2=pytest.importorskip("cv2")
    yaml=pytest.importorskip("yaml")
    dataset=tmp_path/"synthetic-fixture-not-real-data"
    origin=1403636579763555584
    stamps=[origin+i*50_000_000 for i in range(45)]
    for name,sensor in zip(("cam0","cam1","imu0"),calibration()):
        folder=dataset/"mav0"/name
        folder.mkdir(parents=True)
        (folder/"sensor.yaml").write_text(yaml.safe_dump(sensor))
        if name.startswith("cam"):
            (folder/"data").mkdir()
            (folder/"data.csv").write_text("\n".join(f"{t},{t}.png" for t in stamps))
            for stamp in stamps:
                assert cv2.imwrite(str(folder/"data"/f"{stamp}.png"),np.zeros((480,752),dtype=np.uint8))
        else:
            (folder/"data.csv").write_text("\n".join(
                f"{origin+i*5_000_000},1,2,3,4,5,6" for i in range(-4,441)))
    protocol=tmp_path/"protocol.json"
    protocol.write_text(json.dumps(dict(dataset=MODULE.DATASET,runtime_sha256=MODULE.RUNTIME_SHA256,
                                       scope="Synthetic software test only")))
    selection=tmp_path/"selection.json"
    MODULE.prepare(dataset,protocol,selection)
    return dataset,protocol,selection


def test_handoff_runs_both_modes_with_causal_imu_and_keeps_smoke_separate(tmp_path,monkeypatch):
    dataset,protocol,selection=synthetic_prefix(tmp_path)
    delivered=[]
    class FixtureClient:
        """Protocol fixture; never a native SLAM or dataset-performance result."""
        def __init__(self,*args,sensor_mode,**kwargs):
            self.mode=sensor_mode
            self.ready={"fixture":True}
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def track(self,stamp,*images,**kwargs):
            imu=kwargs.get("imu_samples",[])
            if self.mode=="stereo": assert not imu
            else: assert imu and max(x[0] for x in imu) <= stamp
            delivered.append((self.mode,stamp,imu))
            frame=dict(schema="bhl-orb-native-frame-v1",timestamp_s=stamp,tracking_state=1,
                       tracked=False,map_id=None,T_W_C=None,compute_seconds=.001)
            if self.mode=="stereo_inertial":
                frame["inertial"]=dict(map_imu_initialized=False,imu_samples=len(imu))
            return frame
    monkeypatch.setattr(MODULE,"NativeOrbClient",FixtureClient)
    monkeypatch.setitem(sys.modules,"native_stereo_navigation_campaign",
                        SimpleNamespace(unpack_runtime=lambda *args: tmp_path/"fixture-runtime"))
    smoke=tmp_path/"smoke"
    args=SimpleNamespace(dataset=dataset,protocol=protocol,selection=selection,output=smoke,
                         runtime_archive=tmp_path/"fixture-runtime.tar",stage="smoke",smoke_receipt=None)
    result=MODULE.run(args)
    assert result["paired_frames_per_arm"]==40 and result["scored"] is False
    assert all(a["accepted_initialized_frames"]==0 for a in result["arms"])
    assert len(delivered)==80
    full=SimpleNamespace(**{**vars(args),"stage":"full","output":tmp_path/"full",
                            "smoke_receipt":smoke/"result.json"})
    measured=MODULE.run(full)
    assert measured["paired_frames_per_arm"]==45 and measured["scored"] is True
    assert len(delivered)==170
    assert measured["window_last_sensor_time_s"]==pytest.approx(2.2)
    assert measured["arms"][1]["accepted_initialized_frames"]==0
    assert measured["arms"][0]["ate"]["status"]=="UNMEASURED"
    assert "neither read nor hashed" in measured["arms"][0]["ate"]["reason"]
    (smoke/"stereo"/"frames.jsonl").write_text("changed")
    with pytest.raises(ValueError,match="checksum"):
        MODULE.check_smoke(smoke/"result.json",protocol,selection,MODULE.source_identity())


def test_preparation_rejects_missing_data_and_full_rejects_missing_smoke(tmp_path):
    with pytest.raises(FileNotFoundError):
        MODULE.prepare(tmp_path/"missing-data",tmp_path/"missing-protocol",tmp_path/"selection.json")
    with pytest.raises(ValueError,match="same-source"):
        MODULE.check_smoke(None,tmp_path/"protocol",tmp_path/"selection",{})


def test_native_runner_rejects_manifest_missing_original_imu_before_inference(tmp_path,monkeypatch):
    dataset,protocol,selection=synthetic_prefix(tmp_path)
    value=json.loads(selection.read_text())
    del value["files"]["mav0/imu0/data.csv"]
    selection.write_text(json.dumps(value))
    monkeypatch.setitem(sys.modules,"native_stereo_navigation_campaign",
                        SimpleNamespace(unpack_runtime=lambda *args: pytest.fail("Native inference reached")))
    args=SimpleNamespace(dataset=dataset,protocol=protocol,selection=selection,output=tmp_path/"run",
                         runtime_archive=tmp_path/"runtime.tar",stage="smoke",smoke_receipt=None)
    with pytest.raises(ValueError,match="Complete original input manifest"):
        MODULE.run(args)
