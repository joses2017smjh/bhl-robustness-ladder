"""Actual-asset integration and negative controls for a portable replay bundle.

Uses only the standard library test runner, alongside the bundle's installed
runtime dependencies. A missing trained bundle is an error, never a skipped CI
check. Run from any working directory with --bundle and a fresh JSON --out.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    if args.out.exists():
        parser.error("output exists; use a fresh path")
    sys.path.insert(0, str(bundle / "src"))
    from bhl_robust.eval.replay_gate import (compare_replay, digest, guard_decision_check,
                                           load_runtime, replay, verify_bundle)

    manifest = verify_bundle(bundle)
    scenario = manifest["scenarios"][1]

    class ActualBundleChecks(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            cls.nominal = replay(bundle, scenario, manifest["seconds"])

        def test_real_checkpoint_and_physics_load(self):
            cfg, policy, _, env = load_runtime(bundle)
            self.assertEqual(cfg.num_actions, 12)
            self.assertEqual(policy.session.get_providers()[0], "CPUExecutionProvider")
            self.assertGreater(env.mj_model.ngeom, 1)
            self.assertIsNone(self.nominal["runtime_error"])
            self.assertEqual(len(self.nominal["trace"]), int(manifest["seconds"] / cfg.policy_dt))
            self.assertGreater(sum(any(t["contact_counts"]) for t in self.nominal["trace"]), 0)

        def test_independent_nominal_reset_repeats(self):
            again = replay(bundle, scenario, manifest["seconds"])
            self.assertTrue(compare_replay(self.nominal, again)["passed"])

        def test_different_reset_seed_changes_actual_state(self):
            other = replay(bundle, {**scenario, "seed": scenario["seed"] + 100}, manifest["seconds"])
            self.assertFalse(compare_replay(self.nominal, other)["passed"])

        def test_fault_label_alone_cannot_trigger_gate(self):
            labeled = deepcopy(self.nominal)
            labeled["fault"] = "observation_nan"
            self.assertTrue(compare_replay(self.nominal, labeled)["passed"])

        def test_contact_audit_uses_raw_physics_evidence(self):
            wrong_score = deepcopy(self.nominal)
            wrong_score["scored_contact_steps"] = 0
            wrong_score["fault"] = None
            verdict = compare_replay(self.nominal, wrong_score)
            self.assertIn("contact_score_inconsistent_with_physics_samples", verdict["reasons"])

        def test_real_quaternion_order_fault_changes_trajectory(self):
            mutant = replay(bundle, scenario, manifest["seconds"], "quaternion_xyzw")
            mutant["fault"] = None
            verdict = compare_replay(self.nominal, mutant)
            self.assertFalse(verdict["passed"])
            self.assertGreater(verdict["max_state_or_target_abs_delta"], 1e-6)

        def test_cached_decisions_include_actual_data_and_invalid_inputs(self):
            decisions = guard_decision_check(bundle)
            self.assertTrue(decisions["decisions_match"])
            self.assertEqual(decisions["cases"], 208)
            self.assertEqual(decisions["accepted"], 202)
            self.assertEqual(decisions["rejected"], 6)

        def test_cli_rejects_changed_checkpoint_before_creating_outputs(self):
            self._cli_asset_failure("policy.onnx", missing=False)

        def test_cli_rejects_missing_scene_before_creating_outputs(self):
            self._cli_asset_failure("scene/bhl_biped_scene.xml", missing=True)

        def _cli_asset_failure(self, relative, missing):
            with tempfile.TemporaryDirectory(prefix="bhl-negative-control-") as tmp:
                copied = Path(tmp) / "bundle"
                shutil.copytree(bundle, copied)
                asset = copied / relative
                if missing:
                    asset.unlink()
                else:
                    with asset.open("r+b") as f:
                        value = f.read(1)
                        f.seek(0)
                        f.write(bytes([value[0] ^ 1]))
                output = Path(tmp) / "unexpected-results"
                process = subprocess.run([sys.executable, str(copied / "replay.py"), "--bundle", str(copied),
                                          "--out", str(output), "--calls", "1", "--warmup", "0"],
                                         capture_output=True, text=True)
                self.assertNotEqual(process.returncode, 0)
                self.assertIn(f"artifact mismatch or missing file: {relative}", process.stderr)
                self.assertFalse(output.exists())

    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ActualBundleChecks))
    verify_bundle(bundle)
    report = {"status": "PASS" if result.wasSuccessful() else "FAIL", "tests_run": result.testsRun,
              "failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped),
              "manifest_sha256": digest(bundle / "manifest.json"), "log": stream.getvalue(),
              "scope": "Actual trained checkpoint and MuJoCo; fault-label blindness, contact audit, guard decisions and CLI integrity negative controls."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(stream.getvalue(), end="")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
