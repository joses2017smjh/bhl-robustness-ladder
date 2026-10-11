"""Provenance gates for the CPU controller; fixtures are never estimators."""
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("orb_after_build", Path(__file__).resolve().parents[1]/"scripts/bench/native_orb_after_build.py")
M = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(M)


def add(archive, name, value, mode=0o644):
    info = tarfile.TarInfo(name); info.size = len(value); info.mode = mode
    archive.addfile(info, io.BytesIO(value))


class PromotionGates(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def runtime_fixture(self, *, pin=M.PIN, lost_file=False, algorithm_modified=False):
        import hashlib
        files = {name: b"UNIT FIXTURE: NOT AN ESTIMATOR" for name in
                 ("bin/orb_native", "ORBvoc.txt", "ORB-SLAM3-LICENSE.txt", "upstream-source.tar.gz", "ldd-runtime.txt")}
        runtime = {"schema": "bhl-native-orb-runtime-v1", "status": "PASS", "upstream_commit": pin,
                   "headless_visualization_only": True, "native_algorithm_sources_modified": algorithm_modified,
                   "version_smoke": "UNIT FIXTURE " + pin,
                   "files_sha256": {k: hashlib.sha256(v).hexdigest() for k, v in files.items()}}
        files["runtime.json"] = json.dumps(runtime).encode()
        result = {"runtime_files": {k: {"sha256": hashlib.sha256(v).hexdigest(), "bytes": len(v)} for k, v in files.items()}}
        job = self.root/"job"; job.mkdir()
        with tarfile.open(job/"outputs.tar.gz", "w:gz") as archive:
            add(archive, "campaign_result.json", json.dumps(result).encode())
            for name, body in files.items():
                if lost_file and name == "ORBvoc.txt": continue
                add(archive, "runtime/"+name, body, mode=0o755 if name.endswith("orb_native") else 0o644)
        return job, result

    def test_compact_pack_keeps_complete_verified_runtime_and_binary_mode(self):
        job, result = self.runtime_fixture()
        receipt = M.compact_runtime(job, result, self.root/"pack/runtime.tar.gz")
        self.assertEqual(receipt["runtime_files"], 6)
        with tarfile.open(receipt["path"]) as archive:
            self.assertTrue(all(m.isfile() and m.name.startswith("runtime/") for m in archive.getmembers()))
            self.assertEqual(archive.getmember("runtime/bin/orb_native").mode, 0o755)
            self.assertEqual(len(archive.getmembers()), 6)

    def test_missing_vocabulary_cannot_be_promoted(self):
        job, result = self.runtime_fixture(lost_file=True)
        with self.assertRaisesRegex(ValueError, "missing"):
            M.compact_runtime(job, result, self.root/"pack/runtime.tar.gz")
        self.assertFalse((self.root/"pack/runtime.tar.gz").exists())

    def test_modified_native_algorithm_or_wrong_pin_cannot_be_promoted(self):
        for case, kwargs in (("pin", {"pin": "0"*40}), ("algorithm", {"algorithm_modified": True})):
            with self.subTest(case=case):
                root = self.root; self.root = root/case; self.root.mkdir()
                job, result = self.runtime_fixture(**kwargs)
                with self.assertRaisesRegex(ValueError, "source contract"):
                    M.compact_runtime(job, result, self.root/"pack/runtime.tar.gz")
                self.root = root

    def test_tampered_runtime_member_fails_before_published_archive(self):
        job, result = self.runtime_fixture()
        result["runtime_files"]["ORBvoc.txt"]["sha256"] = "0"*64
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            M.compact_runtime(job, result, self.root/"pack/runtime.tar.gz")
        self.assertFalse((self.root/"pack/runtime.tar.gz").exists())

    def test_native_build_negative_is_not_a_successful_runtime(self):
        freeze = self.root/"freeze"; freeze.mkdir()
        (freeze/"frozen-campaign.tar.gz").write_bytes(b"frozen source fixture")
        (freeze/"protocol.json").write_text(json.dumps({"jobs": [{"name": "orb-native-build-v5-20261009", "entrypoint": "scripts/bench/native_build_campaign.py"}]}))
        pins = {"archive_sha256": M.sha(freeze/"frozen-campaign.tar.gz"), "protocol_sha256": M.sha(freeze/"protocol.json")}
        (freeze/"intake.json").write_text(json.dumps(pins))
        target = freeze/"orb-native-build-v5-20261009-123"; target.mkdir()
        (target/"completion.json").write_text(json.dumps({"status": "NEGATIVE", "exit_status": 0, "error": None}))
        plan = {"build_freeze": str(freeze), "build_result_directory": target.name, "build_job_id": "123", **pins}
        with self.assertRaisesRegex(ValueError, "did not complete successfully"):
            M.verify_build(plan)

    def test_both_templates_verified_before_runtime_or_first_submission(self):
        plan = {"schema": "bhl-orb-post-build-plan-v1", "campaigns": [{"template": "offline"}, {"template": "stereo"}]}
        with patch.object(M, "verify_template", side_effect=[None, ValueError("second template changed")]) as templates, \
             patch.object(M, "verify_build") as build, patch.object(M.subprocess, "run") as submit:
            with self.assertRaisesRegex(ValueError, "second template"):
                M.promote(plan, self.root/"output")
            self.assertEqual(templates.call_count, 2); build.assert_not_called(); submit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
