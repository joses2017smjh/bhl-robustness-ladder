"""Offline navigation scorer retains losses and never extrapolates truth."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

SPEC=importlib.util.spec_from_file_location("native_navigation_observer_test",Path(__file__).parents[1]/"scripts/bench/native_navigation_observe.py")
OBS=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(OBS)


class NativeNavigationObserverTests(unittest.TestCase):
    def setUp(self):
        self.truth=np.repeat(np.eye(4)[None],3,axis=0);self.truth[:,0,3]=[0,1,2]
        pose1=np.eye(4);pose1[0,3]=1
        pose2=np.eye(4);pose2[0,3]=2.1
        self.native=[{"timestamp_s":0.,"tracked":False,"T_W_I":None,"map_reset_id":0},
                     {"timestamp_s":1.,"tracked":True,"T_W_I":pose1,"map_reset_id":0},
                     {"timestamp_s":2.,"tracked":True,"T_W_I":pose2,"map_reset_id":0}]
        self.row={"native_frames":3,"case":"straight","seed":1,"success":False,"fell":False,"collision":False,"nonfinite":False,
                  "client_wall_p95_ms":20.,"dropped_scans":0,"elapsed_simulation_s":3.,
                  "trace":[{"reason":"estimated_route","timestamp_s":2.1,"evaluator_goal_distance_m":3.,"native_available_at_s":2.04,"native_last_consumed_at_s":2.08}]}

    def score(self):return OBS.score_episode(self.row,self.native,np.arange(3,dtype=float),self.truth)

    def test_null_initial_pose_is_excluded_from_metric_error(self):
        result=self.score()
        self.assertEqual(result["native_tracked_frames"],2)
        self.assertEqual(result["metrics"]["associated_poses"],2)
        self.assertEqual(result["metrics"]["tracking_lost_frames"],1)
        self.assertAlmostEqual(result["metrics"]["ate_translation_m"]["rmse"],.05)
        self.assertAlmostEqual(result["metrics"]["endpoint_drift_m"],.1)

    def test_lost_frame_cannot_hide_identity_as_estimate(self):
        self.native[0]["T_W_I"]=np.eye(4)
        with self.assertRaisesRegex(ValueError,"null pose"):self.score()

    def test_arrival_after_visibility_is_rejected(self):
        self.row["trace"][0]["native_available_at_s"]=2.2
        with self.assertRaisesRegex(ValueError,"before arrival"):self.score()

    def test_future_controller_consumption_is_rejected(self):
        self.row["trace"][0]["native_last_consumed_at_s"]=2.2
        with self.assertRaisesRegex(ValueError,"consumption occurs in the future"):self.score()


if __name__=="__main__":unittest.main()
