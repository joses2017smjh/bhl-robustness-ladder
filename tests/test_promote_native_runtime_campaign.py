"""Immutable deferred-runtime promotion; fixture bytes are never SLAM evidence."""
import importlib.util
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("native_runtime_promotion_test", ROOT/"scripts/bench/promote_native_runtime_campaign.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PromotionTests(unittest.TestCase):
    def fixture(self, root, *, altered_runtime=False, pin=None):
        template = root/"template"
        template.mkdir()
        staged = root/"staged"
        launcher = staged/"source/scripts/bench/h34_campaign.py"
        launcher.parent.mkdir(parents=True)
        launcher.write_bytes((ROOT/"scripts/bench/h34_campaign.py").read_bytes())
        actor = staged/"inputs/actor.onnx"
        actor.parent.mkdir()
        actor.write_bytes(b"TEST FIXTURE ONLY: immutable actor bytes")
        resources = {"cpus": 1, "gpus": 0, "memory_gb": 1, "partition": "test",
                     "time_limit": "00:01:00", "requeue": False}
        protocol = {"schema_version": 1, "campaign_id": "promotion-test-only",
                    "runtime": {"python": "/test/python", "share_root": "/test", "stack": "cpu"},
                    "input_files": {"actor.onnx": {"path": str(actor), "sha256": MODULE.sha256(actor)}},
                    "jobs": [{"name": "smoke", "kind": "smoke", "entrypoint": "scripts/bench/h34_campaign.py",
                              "args": [MODULE.SHA_TOKEN], "resources": resources},
                             {"name": "run", "kind": "run", "entrypoint": "scripts/bench/h34_campaign.py",
                              "args": [MODULE.SHA_TOKEN], "resources": resources}],
                    "scientific_setting": {"groups": [17, 18], "horizon_s": 40},
                    "deferred_native_orb_runtime": {"status": "TEMPLATE_ONLY", "build_job_id": "failed-test-build",
                                                    "build_archive_sha256": "f"*64}}
        manifest = {str(p.relative_to(staged)): {"bytes": p.stat().st_size, "sha256": MODULE.sha256(p)}
                    for p in (launcher, actor)}
        for name, value in (("protocol.json", protocol), ("manifest.json", manifest)):
            MODULE.write_json(template/name, value)
        archive = template/"frozen-campaign.tar.gz"
        with tarfile.open(archive, "w:gz") as stream:
            for name in ("protocol.json", "manifest.json"):
                stream.add(template/name, arcname=name, recursive=False)
            for name in manifest:
                stream.add(staged/name, arcname=name, recursive=False)
        (template/"h34_campaign.sbatch").write_text("#!/bin/sh\n# TEST FIXTURE ONLY\n")
        MODULE.write_json(template/"intake.json", {"archive_sha256": MODULE.sha256(archive),
            "protocol_sha256": MODULE.sha256(template/"protocol.json"),
            "launcher_sha256": MODULE.sha256(template/"h34_campaign.sbatch"), "source_commit": "test-only"})
        runtime = root/"native/runtime"
        runtime.mkdir(parents=True)
        inventory = {}
        for name in ("bin/orb_native", "ORBvoc.txt", "ORB-SLAM3-LICENSE.txt", "upstream-source.tar.gz"):
            path = runtime/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(("TEST FIXTURE ONLY: "+name).encode())
            inventory[name] = MODULE.sha256(path)
        MODULE.write_json(runtime/"runtime.json", {"schema": "bhl-native-orb-runtime-v1", "status": "PASS",
            "upstream_commit": pin or MODULE.ORB_PIN, "native_algorithm_sources_modified": False,
            "files_sha256": inventory})
        if altered_runtime:
            (runtime/"bin/orb_native").write_bytes(b"TAMPERED AFTER RECEIPT")
        runtime_archive = root/"runtime.tar.gz"
        with tarfile.open(runtime_archive, "w:gz") as stream:
            for path in sorted(runtime.rglob("*")):
                if path.is_file():
                    stream.add(path, arcname=str(path.relative_to(runtime.parent)), recursive=False)
        return template, runtime_archive, protocol, manifest

    def test_promotion_preserves_actor_and_science_and_needs_no_git_checkout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            template, runtime, science, original_manifest = self.fixture(root)
            output = root/"promoted"
            receipt = MODULE.promote(template, runtime, MODULE.sha256(runtime), output)
            actual = json.loads((output/"protocol.json").read_text())
            manifest = json.loads((output/"manifest.json").read_text())
            self.assertEqual(actual["scientific_setting"], science["scientific_setting"])
            self.assertEqual(actual["jobs"][0]["args"], [MODULE.sha256(runtime)])
            self.assertEqual(actual["input_files"]["actor.onnx"], science["input_files"]["actor.onnx"])
            provenance = actual["deferred_native_orb_runtime"]
            self.assertNotIn("build_job_id", provenance)
            self.assertNotIn("build_archive_sha256", provenance)
            self.assertEqual(provenance["original_requested_build_job_id"], "failed-test-build")
            self.assertEqual(provenance["archive_sha256"], MODULE.sha256(runtime))
            for key, value in original_manifest.items():
                self.assertEqual(manifest[key], value)
            self.assertTrue(receipt["all_original_source_and_actor_entries_unchanged"])
            self.assertEqual(receipt["submissions"], [])
            self.assertFalse((template/".git").exists())
            with self.assertRaisesRegex(ValueError, "duplicate"):
                MODULE.promote(template, runtime, MODULE.sha256(runtime), output)

    def test_real_receipt_cannot_hide_modified_binary_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            template, runtime, _, _ = self.fixture(root, altered_runtime=True)
            with self.assertRaisesRegex(ValueError, "member checksum mismatch"):
                MODULE.promote(template, runtime, MODULE.sha256(runtime), root/"rejected")
            self.assertFalse((root/"rejected").exists())

    def test_success_label_does_not_override_wrong_upstream_pin(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            template, runtime, _, _ = self.fixture(root, pin="0"*40)
            with self.assertRaisesRegex(ValueError, "pinned unmodified"):
                MODULE.promote(template, runtime, MODULE.sha256(runtime), root/"rejected")
            self.assertFalse((root/"rejected").exists())


if __name__ == "__main__":
    unittest.main()
