"""Engineering check (not science): run the frozen-candidate run_episode for both arms on CPU.

Seed 4242 is outside every declared cohort. Verifies the closed-loop plumbing:
Atlas telemetry on every native frame, command logging, the resume gate and
client, episode/receipt lineage and the v1 schedule-evidence validator.
"""
import hashlib, json, os, pathlib, sys, time
src, inputs, orb_archive, out = (pathlib.Path(x) for x in sys.argv[1:5])
seconds = float(sys.argv[5])
sys.path.insert(0, str(src / "src")); sys.path.insert(0, str(src / "scripts/bench"))
import stereo_recovery_campaign as C
ORB_SHA = "df8f927939837cde5edc7c81d75b41c94d27be06f44636e5aa362cb48a37c72c"
LIO_SHA = "6fc66e9318be7e8f5d2ee545652928b071ccbe949f1abc21f1247170f0e6b6f4"
work = out / "work"
orb = C.STEREO.unpack_runtime(orb_archive, ORB_SHA, work / "orb")
lio = C.HYBRID.unpack_lio(inputs / "runtime/lio-runtime.tar.gz", LIO_SHA, work / "lio")
actor = {"name": "frozen_dr_default_s0", "deploy": str(inputs / "actors/dr-default-s0/deploy.yaml"),
         "checkpoint": str(inputs / "actors/dr-default-s0/policy.onnx")}
summary = {}
for arm in C.ARMS:
    started = time.monotonic()
    row = C.run_episode(src / "external/Berkeley-Humanoid-Lite", actor, {"orb": orb, "lio": lio}, "straight", 4242,
                        out / arm, condition="nominal", arm=arm, seconds=seconds)
    ep = out / arm
    for name, field in C.EVIDENCE:
        assert C.sha256(ep / name) == row[field], name
    receipt = json.loads((ep / "native/recovery_receipt.json").read_text())
    frames = [json.loads(l) for l in (ep / "native/native_frames.jsonl").read_text().splitlines()]
    camera = [f for f in frames if f.get("new_native_camera_frame")]
    assert row["atlas_telemetry_frames"] == row["native_camera_frames"] == len(camera) > 0
    assert all(f["raw_T_W_I"] is not None for f in camera if f["native_ORB_tracking_state"] == 2)
    assert len(receipt["applied_commands"]) == len(row["trace"])
    summary[arm] = {k: row[k] for k in ("success", "fell", "collision", "native_camera_frames", "native_raw_ok_frames",
        "native_tracked_frames", "atlas_telemetry_frames", "verified_resumes", "shadow_verified_resumes",
        "atlas_coordinate_invalidations", "native_coordinate_invalidations", "stopped_navigation_time_s",
        "navigation_decisions", "client_wall_p95_ms", "accepted_pose_error_p95_m", "elapsed_simulation_s")}
    summary[arm]["permanent_stop"] = row["permanent_stop"]
    summary[arm]["loss_mechanisms"] = row["recovery"]["loss_mechanisms"]
    summary[arm]["gate_final_state"] = receipt["gate"]["final_state"]
    summary[arm]["wall_s"] = time.monotonic() - started
    summary[arm]["episode_bytes"] = sum(p.stat().st_size for p in ep.rglob("*") if p.is_file())
result = {"schema": "bhl-stereo-recovery-cpu-episode-check-v1", "status": "PASS", "seed": 4242, "seconds": seconds,
          "mujoco_gl": os.environ.get("MUJOCO_GL"), "arms": summary,
          "scope": "engineering plumbing check on CPU software rendering; not a scored or predeclared episode"}
(out / "check_result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
print(json.dumps(result, indent=1))
