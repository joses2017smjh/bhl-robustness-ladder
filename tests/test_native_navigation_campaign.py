"""Navigation inference accepts actual native packets, never truth pose."""
import importlib.util
from pathlib import Path
import unittest
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np

SPEC=importlib.util.spec_from_file_location("native_navigation_test",Path(__file__).parents[1]/"scripts/bench/native_navigation_campaign.py")
NAV=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(NAV)


class NativeNavigationBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.packet={"timestamp_s":1.,"tracked":True,"map_reset_id":0,"T_W_I":np.eye(4).tolist()}
        self.rays=np.tile([2.,.5,.1],(20,1))

    def command(self,**changes):
        values=dict(native=self.packet,registration=np.eye(4),t_body_imu=np.eye(4),route=[[5.,0]],
                    waypoint=0,now=1.1,raw_points_b=self.rays)
        values.update(changes)
        return NAV.navigation_command(**values)

    def test_forward_command_from_native_estimate(self):
        command,_,reason,pose=self.command()
        self.assertAlmostEqual(command[0],.3)
        self.assertEqual(reason,"estimated_route")
        np.testing.assert_allclose(pose,np.eye(4))

    def test_tracking_loss_and_missing_pose_stop(self):
        self.packet.update(tracked=False,T_W_I=None)
        command,_,reason,pose=self.command()
        np.testing.assert_array_equal(command,[0,0,0])
        self.assertEqual(reason,"native_untracked_stop")
        self.assertIsNone(pose)

    def test_stale_or_future_native_clock_stops(self):
        for now in (.8,1.351):
            command,_,reason,_=self.command(now=now)
            self.assertEqual(reason,"native_stale_stop")
            np.testing.assert_array_equal(command,[0,0,0])

    def test_unknown_map_reset_is_never_relocalized_by_truth(self):
        self.packet["map_reset_id"]=1
        command,_,reason,pose=self.command()
        self.assertEqual(reason,"native_unregistered_or_reset_stop")
        self.assertIsNone(pose)
        np.testing.assert_array_equal(command,[0,0,0])

    def test_missing_start_registration_stops(self):
        command,_,reason,_=self.command(registration=None)
        self.assertEqual(reason,"native_unregistered_or_reset_stop")
        np.testing.assert_array_equal(command,[0,0,0])

    def test_fixed_imu_extrinsic_is_removed_from_body_pose(self):
        imu=np.eye(4);imu[:3,3]=[.2,0,.4]
        self.packet["T_W_I"]=imu.tolist()
        _,_,_,pose=self.command(t_body_imu=imu)
        np.testing.assert_allclose(pose,np.eye(4))

    def test_goal_stop_uses_estimate_only(self):
        self.packet["T_W_I"][0][3]=4.8
        command,_,reason,_=self.command()
        self.assertEqual(reason,"estimated_goal_stop")
        np.testing.assert_array_equal(command,[0,0,0])

    def test_unknown_returns_cannot_be_declared_free_space(self):
        command,_,reason,_=self.command(raw_points_b=np.empty((0,3)))
        self.assertEqual(reason,"lidar_unknown_stop")
        np.testing.assert_array_equal(command,[0,0,0])

    def test_actual_near_obstacle_return_stops(self):
        self.rays[0]=[.3,0,.25]
        command,_,reason,_=self.command()
        self.assertEqual(reason,"raw_obstacle_stop")
        np.testing.assert_array_equal(command,[0,0,0])

    def test_turn_in_place_towards_external_waypoint(self):
        command,_,reason,_=self.command(route=[[0.,5.]])
        self.assertEqual(reason,"estimated_route")
        self.assertEqual(command[0],0)
        self.assertAlmostEqual(command[2],.45)

    def test_waypoint_progress_uses_estimated_position(self):
        self.packet["T_W_I"][0][3]=2.1
        _,index,reason,_=self.command(route=[[2.2,0.],[2.5,2.8]])
        self.assertEqual(index,1)
        self.assertEqual(reason,"estimated_route")

    def test_nonrigid_native_pose_rejected(self):
        self.packet["T_W_I"][0][0]=2
        with self.assertRaisesRegex(ValueError,"SO\\(3\\)"):self.command()


class FixedImuCalibrationTests(unittest.TestCase):
    def model(self):
        return SimpleNamespace(site_bodyid=[2],site_pos=[[.01,0,.02]],site_quat=[[1,0,0,0]],
            body_parentid=[0,0,1],body_jntnum=[0,1,0],body_mocapid=[-1,-1,-1],
            body_pos=[[0,0,0],[0,0,0],[.06,0,.68]],body_quat=[[1,0,0,0]]*3)

    def test_fixed_child_imu_mount_is_calibrated(self):
        pose=NAV.fixed_site_transform(self.model(),1,0)
        np.testing.assert_allclose(pose[:3,3],[.07,0,.70])

    def test_articulated_imu_mount_is_rejected(self):
        model=self.model();model.body_jntnum[2]=1
        with self.assertRaisesRegex(ValueError,"fixed body chain"):NAV.fixed_site_transform(model,1,0)

    def test_unrelated_site_does_not_get_pose_registration(self):
        model=self.model();model.body_parentid[2]=0
        with self.assertRaisesRegex(ValueError,"fixed body chain"):NAV.fixed_site_transform(model,1,0)


class NativeArrivalCausalityTests(unittest.TestCase):
    def test_pending_loss_is_invisible_during_earlier_physics(self):
        delivery=NAV.DelayedNativeState()
        tracked={"tracked":True,"timestamp_s":1.,"map_reset_id":0,"T_W_I":np.eye(4).tolist()}
        lost={"tracked":False,"timestamp_s":1.1,"map_reset_id":0,"T_W_I":None}
        delivery.available=tracked
        ticks=delivery.stage(lost,np.empty((0,3)),10,.081,.04)
        self.assertEqual(ticks,3)
        for step in (10,11,12):
            self.assertFalse(delivery.advance(step))
            self.assertIs(delivery.available,tracked)
            self.assertTrue(delivery.available["tracked"])
        self.assertTrue(delivery.advance(13))
        self.assertIs(delivery.available,lost)
        self.assertFalse(delivery.available["tracked"])
        self.assertAlmostEqual(delivery.available_at_s,.52)

    def test_future_pose_and_raw_scan_cannot_replace_available_inputs(self):
        delivery=NAV.DelayedNativeState();old=np.array([[1,0,0]])
        delivery.raw=old
        future={"tracked":True,"timestamp_s":2.,"map_reset_id":0,"T_W_I":np.eye(4).tolist()}
        new=np.array([[2,0,0]])
        delivery.stage(future,new,5,.05,.04)
        self.assertIsNone(delivery.available);self.assertIs(delivery.raw,old)
        self.assertFalse(delivery.advance(6))
        self.assertTrue(delivery.advance(7));self.assertIs(delivery.raw,new)

    def test_second_pending_result_and_invalid_latencies_rejected(self):
        delivery=NAV.DelayedNativeState();packet={"timestamp_s":0.}
        with self.assertRaisesRegex(ValueError,"finite native wall"):
            delivery.stage(packet,np.empty((0,3)),0,float("nan"),.04)
        delivery.stage(packet,np.empty((0,3)),0,.01,.04)
        with self.assertRaisesRegex(ValueError,"already pending"):
            delivery.stage(packet,np.empty((0,3)),0,.01,.04)


if __name__=="__main__":unittest.main()
