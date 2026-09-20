#!/usr/bin/env python3
"""Refresh the campaign report from measured artifacts and Slurm accounting."""
from __future__ import annotations
import argparse
from collections import Counter
import datetime as dt
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path(__file__).resolve().parents[1]/"results/weekend-20260919")
    parser.add_argument("--slurm", action="store_true")
    a = parser.parse_args()
    directory = a.directory.resolve()
    receipts = [json.loads(line) for line in (directory/"submissions.jsonl").read_text().splitlines() if line]
    states = {}
    if a.slurm:
        ids = ",".join(r["job_id"] for r in receipts)
        raw = subprocess.check_output(["sacct", "-j", ids, "-X", "-n", "-P",
            "--format=JobID,State,Elapsed,NodeList"], text=True, timeout=45)
        for line in raw.splitlines():
            fields = line.split("|")
            if len(fields) >= 4:
                states[fields[0]] = dict(state=fields[1], elapsed=fields[2], node=fields[3])
    rows = []
    patterns = ("maze-eval-*.json", "inspection-maze*.json", "team*.json", "fold-baseline-*.json", "fold-adapt0-*.json", "fold-adapt1-*.json")
    for pattern in patterns:
        for path in sorted(directory.glob(pattern)):
            try:
                data = json.loads(path.read_text())
            except json.JSONDecodeError:
                # Evaluators flush progress while running; never call a partial
                # write a result or fail the whole reporting job on that race.
                rows.append(dict(file=path.name, verdict="progress write incomplete; refresh later", scope="not scored"))
                continue
            if path.name.startswith("maze"):
                verdict = f"first-episode success {data['first_episode_success_rate']:.1%}; passed={data['passed']}"
                scope = "Oracle-route Isaac navigation; paired ray depth where enabled"
            elif path.name.startswith(("team", "inspection-maze")):
                summary = data.get("summary")
                verdict = "evaluation incomplete"
                if summary:
                    verdict = ", ".join(f"{k}={v:.0%}" for k,v in summary["success_rates"].items())
                    verdict += f"; controls_complete={summary['controls_complete']}"
                scope = ("Two-turn inspection maze, oracle route/pose, sensor braking"
                         if path.name.startswith("inspection-maze") else
                         "Shared-world cooperation, frozen locomotion; " + str(data.get("sensor_mode", "off")))
            else:
                verdict = f"completed={data['completed']}"
                for split in ("seen", "unseen"):
                    if split in data:
                        result = data[split]
                        verdict += f"; {split}={result['successes']}/{result['episodes']}"
                scope = "Official scorer with strict RGB/particle health"
            rows.append(dict(file=path.name, verdict=verdict, scope=scope))
    path = directory/"cloth-mujoco-gate.json"
    if path.exists():
        data = json.loads(path.read_text())
        rows.append(dict(file=path.name, verdict=f"fold success={data['fold_success_rate']:.0%}; gate={data['gate_pass']}",
                         scope="Released deformable-towel physics diagnostic; idealized pickers"))
    training = []
    for folder in sorted(directory.glob("fold-adapt-s*")):
        if (folder/"latest.json").exists():
            current = json.loads((folder/"latest.json").read_text())
            training.append(dict(run=folder.name, checkpoint_step=current["step"],
                                 complete=(folder/"completed.json").exists()))
    report = dict(updated_utc=dt.datetime.now(dt.timezone.utc).isoformat(), evidence=rows,
                  training=training, slurm=states, submitted_jobs=len(receipts))
    (directory/"summary.json").write_text(json.dumps(report, indent=2)+"\n")
    lines = ["# Weekend campaign results", "", f"Updated {report['updated_utc']}", "",
        "Only completed, physically scored evaluations support task-success claims. Training losses and smoke tests do not.", "",
        "Historical folding job 21214241 has invalid visual inputs after garment switches; its old 6/24 is not the baseline.", "",
        "| Evidence | Measured result | Scope |", "| --- | --- | --- |"]
    lines += [f"| [{r['file']}]({r['file']}) | {r['verdict']} | {r['scope']} |" for r in rows]
    lines += ["", "## Training checkpoints", ""]
    lines += [f"- {r['run']}: checkpoint step {r['checkpoint_step']}; completed={r['complete']}" for r in training]
    if states:
        lines += ["", "## Scheduler", "", "| Submission | Name | Current accounting |", "| --- | --- | --- |"]
        for receipt in receipts:
            matching = [v["state"] for key,v in states.items()
                        if key == receipt["job_id"] or key.startswith(receipt["job_id"]+"_")]
            status = ", ".join(f"{count} {state}" for state,count in sorted(Counter(matching).items())) or "not returned"
            lines.append(f"| {receipt['job_id']} | {receipt['name']} | {status} |")
    lines += ["", "[Campaign scope and research](../../docs/WEEKEND_CAMPAIGN.md)", ""]
    (directory/"SUMMARY.md").write_text("\n".join(lines))
    print(json.dumps({"report":str(directory/"SUMMARY.md"), "evidence_files":len(rows), "training":training}))


if __name__ == "__main__":
    main()
