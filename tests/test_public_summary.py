"""Public result reports must not require unpublished scheduler receipts."""
import json
from pathlib import Path
import subprocess
import sys


def test_artifact_only_summary_and_missing_scheduler_receipts(tmp_path):
    artifact = {
        "first_episode_success_rate": 0.75,
        "passed": True,
    }
    (tmp_path / "maze-eval-test.json").write_text(json.dumps(artifact))
    script = Path(__file__).resolve().parents[1] / "scripts/summarize_weekend.py"
    command = [sys.executable, str(script), "--directory", str(tmp_path)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    report = json.loads((tmp_path / "summary.json").read_text())
    assert report["evidence"][0]["verdict"] == "first-episode success 75.0%; passed=True"
    assert "slurm" not in report
    assert "submitted_jobs" not in report
    missing = subprocess.run(command + ["--slurm"], capture_output=True, text=True)
    assert missing.returncode == 2
    assert "requires local submissions.jsonl receipts" in missing.stderr
