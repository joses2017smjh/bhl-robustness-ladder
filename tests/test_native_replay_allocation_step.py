"""Allocation receipt fixtures are metadata tests, not native SLAM runs."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SPEC=importlib.util.spec_from_file_location("native_allocation_step_tests",Path(__file__).parents[1]/"scripts/bench/native_replay_allocation_step.py")
STEP=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(STEP)


class AllocationStepBoundaryTests(unittest.TestCase):
    def test_actual_owner_running_capacity_required(self):
        text="JobId=123 UserId=test(45) JobState=RUNNING NumCPUs=2 AllocTRES=cpu=2,mem=31800M,node=1 NodeList=node1 Partition=dgx2 EndTime=2026-10-09T16:45:23"
        result=STEP.allocation_resources(text,"123",45)
        self.assertEqual(result["cpus"],2);self.assertEqual(result["memory_mib"],31800)
        for invalid in (text.replace("(45)","(46)"),text.replace("RUNNING","PENDING"),text.replace("31800M","8G")):
            with self.assertRaises(ValueError):STEP.allocation_resources(invalid,"123",45)

    def fixture(self,root):
        environment=root/"smoke-environment.json"
        environment.write_text(json.dumps({"SLURM_JOB_ID":"123","SLURM_STEP_ID":"4"})+"\n")
        completed=root/"smoke-123";completed.mkdir();completion=completed/"completion.json"
        completion.write_text(json.dumps({"status":"PASS","exit_status":0,"archive_sha256":"abc"})+"\n")
        receipt=root/"smoke-step.json"
        value={"status":"ALLOCATION_STEP_COMPLETE","returncode":0,"job_name":"smoke","job_id":"123","slurm_step_id":"4",
            "archive_sha256":"abc","step_environment_path":str(environment),"step_environment_sha256":STEP.digest(environment),"completion_sha256":STEP.digest(completion)}
        receipt.write_text(json.dumps(value)+"\n")
        return receipt,value,environment,completion

    def test_fresh_actual_step_environment_and_completion_are_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);receipt,_,_,_=self.fixture(root)
            result=STEP.validate_smoke_step(root,{"archive_sha256":"abc"},{"jobs":[{"name":"smoke","kind":"smoke"}]},receipt)
            self.assertEqual(result["slurm_step_id"],"4")
            with self.assertRaisesRegex(ValueError,"source/allocation pin"):
                STEP.validate_smoke_step(root,{"archive_sha256":"different"},{"jobs":[{"name":"smoke","kind":"smoke"}]},receipt)

    def test_changed_smoke_environment_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);receipt,_,environment,_=self.fixture(root);environment.write_text("{}")
            with self.assertRaisesRegex(ValueError,"environment hash"):
                STEP.validate_smoke_step(root,{"archive_sha256":"abc"},{"jobs":[{"name":"smoke","kind":"smoke"}]},receipt)

    def test_failed_or_changed_smoke_completion_cannot_release_full_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);receipt,value,_,completion=self.fixture(root)
            completion.write_text(json.dumps({"status":"INCOMPLETE","exit_status":1,"archive_sha256":"abc"}))
            value["completion_sha256"]=STEP.digest(completion);receipt.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError,"completion must PASS"):
                STEP.validate_smoke_step(root,{"archive_sha256":"abc"},{"jobs":[{"name":"smoke","kind":"smoke"}]},receipt)

    def test_wrapper_captures_environment_without_shell_interpolating_arguments(self):
        # Synthetic environment verifies only wrapper serialization and argv.
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);environment=root/"step env.json";launcher=root/"launcher.sh"
            launcher.write_text("#!/bin/bash\nexit 0\n")
            env=dict(os.environ,SLURM_JOB_ID="123",SLURM_STEP_ID="4")
            subprocess.run(["/bin/bash","--noprofile","--norc","-c",STEP.STEP_WRAPPER,"fixture",str(environment),sys.executable,str(launcher),str(root),"name-with-$-literal","pin"],env=env,check=True,capture_output=True)
            result=json.loads(environment.read_text());self.assertEqual(result["SLURM_STEP_ID"],"4")


if __name__=="__main__":unittest.main()
