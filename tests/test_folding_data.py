"""Checks for the learning-data failure modes seen in the folding jobs."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bhl_robust.cloth.folding_data import capture_manifest, split_captures, stratified_success_split


class FoldingDataTests(unittest.TestCase):
    def test_garment_identity_split_and_failure_exclusion(self):
        rows = [dict(garment="Top_Short_Seen_0", episode=0, success=True),
                dict(garment="Top_Short_Seen_0", episode=1, success=True),
                dict(garment="Top_Short_Seen_8", episode=208, success=True),
                dict(garment="Top_Short_Seen_9", episode=228, success=False)]
        split = split_captures(rows)
        self.assertEqual(len(split["train"]), 2)
        self.assertEqual(len(split["validation"]), 1)
        self.assertEqual(len(split["excluded_failed_replays"]), 1)
        self.assertFalse({r["garment"] for r in split["train"]} &
                         {r["garment"] for r in split["validation"]})

    def test_single_action_supervision_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            np.savez(Path(tmp) / "capture.npz", garment="Pant_Short_Seen_0", episode=500,
                     success=True, state=np.zeros((2, 12)), action=np.zeros((2, 12)))
            with self.assertRaisesRegex(ValueError, "50-step chunks"):
                capture_manifest(str(Path(tmp) / "*.npz"))

    def test_nonfinite_supervision_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            action = np.zeros((2, 50, 12))
            action[1, 20, 2] = np.nan
            np.savez(Path(tmp) / "capture.npz", garment="Pant_Short_Seen_0", episode=500,
                     success=True, state=np.zeros((2, 12)), action=action)
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                capture_manifest(str(Path(tmp) / "*.npz"))

    def test_no_validation_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "both train and held-out"):
            split_captures([dict(garment="Pant_Short_Seen_0", episode=500, success=True)])

    def test_stratified_split_covers_every_class_without_identity_leakage(self):
        categories = ("Top_Short", "Top_Long", "Pant_Short", "Pant_Long")
        rows = [dict(garment=f"{category}_Seen_{i}", episode=i, success=True)
                for category in categories for i in (0, 3)]
        split = stratified_success_split(rows)
        self.assertEqual({r["garment"].split("_Seen_")[0] for r in split["train"]}, set(categories))
        self.assertEqual({r["garment"].split("_Seen_")[0] for r in split["validation"]}, set(categories))
        self.assertFalse({r["garment"] for r in split["train"]} &
                         {r["garment"] for r in split["validation"]})

    def test_stratified_split_refuses_a_class_with_one_successful_identity(self):
        with self.assertRaisesRegex(ValueError, "distinct garments"):
            stratified_success_split([dict(garment="Top_Short_Seen_0", episode=0, success=True)])


if __name__ == "__main__":
    unittest.main()
