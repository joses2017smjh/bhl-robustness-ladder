#!/usr/bin/env python3
"""Matched native stereo permanent-stop versus LIO-verified map-frame recovery.

Both arms run original ORB and FAST-LIO2 on the same sensor packets and charge
all capture and both native clients' wall time. The baseline keeps LIO in shadow
mode. Recovery is explicitly hybrid; it never substitutes a LIO pose for stereo.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"src"))
sys.path.insert(0, str(Path(__file__).parent))
import native_stereo_navigation_campaign as STEREO
import native_navigation_campaign as NAV
from bhl_robust.research.native_orb import sha256
from bhl_robust.research.native_lio import NativeLioClient
from bhl_robust.research.stereo_recovery import VerifiedMapRecovery

ARMS = ("permanent_stop", "verified_lio_recovery")


class HybridOrbNavigationClient:
    def __init__(self, runtime, t_imu_lidar, directory, *, max_frames=1000):
        self.runtime, self.directory = runtime, Path(directory)
        self.arm = runtime["arm"]
        if self.arm not in ARMS:
            raise ValueError("unknown frozen stereo method")
        self.t_imu_lidar = np.asarray(t_imu_lidar)
        self.maximum = max_frames
        self.recovery = VerifiedMapRecovery()
        self.records = []

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        self.stack = ExitStack()
        try:
            self.stereo = self.stack.enter_context(STEREO.OrbNavigationClient(
                self.runtime["orb"], self.t_imu_lidar, self.directory/"stereo", max_frames=self.maximum))
            self.lio = self.stack.enter_context(NativeLioClient(
                self.runtime["lio"], self.t_imu_lidar, self.directory/"lio", max_frames=self.maximum))
        except BaseException:
            self.stack.close()
            raise
        return self

    def track(self, scan, imu):
        reference = self.lio.track(scan, imu)
        baseline = self.stereo.track(scan, imu)
        native = self.stereo.client.responses[-1]
        if native.get("diagnostics", {}).get("schema") != "bhl-orb-frame-diagnostics-v1":
            raise ValueError("This experiment requires the real native read-only diagnostics wrapper")
        packet = dict(baseline)
        packet["native_ORB_diagnostics"] = native["diagnostics"]
        packet["reference_native_LIO"] = reference
        if self.arm == "verified_lio_recovery":
            observation = {"timestamp_s": native["timestamp_s"], "tracked": native["tracked"],
                           "map_id": native["map_id"], "T_W_I": None if not native["tracked"] else
                           STEREO.native_camera_to_imu(native["T_W_C"], self.t_imu_lidar).tolist()}
            packet.update(self.recovery.update(observation, reference))
        packet["method_arm"] = self.arm
        packet["lio_mode"] = "shadow_only" if self.arm == "permanent_stop" else "registration_and_validation_no_pose_fallback"
        packet["compute_seconds"] = float(native["compute_seconds"]+reference["compute_seconds"])
        self.records.append(packet)
        return packet

    def __exit__(self, kind, value, traceback):
        try:
            return self.stack.__exit__(kind, value, traceback)
        finally:
            with (self.directory/"native_frames.jsonl").open("x") as stream:
                for row in self.records:
                    stream.write(json.dumps(row, allow_nan=False)+"\n")
            NAV.write_json(self.directory/"recovery.json", self.recovery.receipt())
            NAV.write_json(self.directory/"stream_receipt.json", {
                "schema": "bhl-stereo-methods-stream-v1", "arm": self.arm, "frames": len(self.records),
                "orb_runtime_sha256": sha256(self.runtime["orb"]/"runtime.json"),
                "lio_runtime_sha256": sha256(self.runtime["lio"]/"runtime.json"),
                "recovery_sha256": sha256(self.directory/"recovery.json"),
                "recovery_count": self.recovery.recoveries,
                "latency_scope": "render+PNG+hash plus sequential native LIO and ORB clients including recovery computation, charged before delivery",
                "reference_inputs": ["original timed LiDAR returns", "causal high-rate IMU"],
                "ground_truth_inputs": [], "lio_pose_fallback": False})


def unpack_lio(archive, checksum, output):
    from h34_campaign import safe_extract
    if sha256(archive) != checksum:
        raise ValueError("frozen native LIO archive checksum differs")
    output.mkdir(parents=True, exist_ok=False)
    safe_extract(archive, output)
    runtime = output/"runtime"
    receipt = json.loads((runtime/"runtime.json").read_text())
    if receipt.get("status") != "PASS" or sha256(runtime/"fastlio_headless") != receipt.get("binary_sha256"):
        raise ValueError("successful verified native LIO runtime required")
    (runtime/"fastlio_headless").chmod(0o700)
    return runtime


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--phase", choices=("smoke", "development"), required=True)
    parser.add_argument("--case", choices=NAV.CASES, help="One predeclared paired development shard")
    parser.add_argument("--seed", type=int, choices=(480000, 480001, 480002))
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--deploy", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--orb-runtime-archive", type=Path, required=True)
    parser.add_argument("--orb-runtime-sha256", required=True)
    parser.add_argument("--lio-runtime-archive", type=Path, required=True)
    parser.add_argument("--lio-runtime-sha256", required=True)
    parser.add_argument("--output", type=Path, default=Path(os.getenv("H34_OUTPUT_DIR", "output")))
    args = parser.parse_args(argv)
    if (args.case is None) != (args.seed is None) or (args.case is not None and args.phase != "development"):
        parser.error("--case and --seed must be supplied together for a development pair")
    args.output.mkdir(parents=True, exist_ok=True)
    result = {"schema": "bhl-stereo-methods-campaign-v1", "status": "INCOMPLETE", "phase": args.phase,
              "scientific_status": "SMOKE_ONLY" if args.phase == "smoke" else "DEVELOPMENT_ONLY_NO_CONFIRMATION",
              "episodes": 0, "problems": [], "ground_truth_inputs_to_estimator_or_planner": []}
    began = time.monotonic()
    try:
        work = Path(os.getenv("H34_WORK_DIR", tempfile.mkdtemp(prefix="bhl-stereo-methods-")))
        orb = STEREO.unpack_runtime(args.orb_runtime_archive, args.orb_runtime_sha256, work/"orb")
        if "diagnostic_wrapper" not in json.loads((orb/"runtime.json").read_text()):
            raise ValueError("frozen runtime must contain native diagnostic provenance")
        lio = unpack_lio(args.lio_runtime_archive, args.lio_runtime_sha256, work/"lio")
        inputs = [args.deploy, args.checkpoint, args.orb_runtime_archive, args.lio_runtime_archive]
        inventory = {str(path.resolve()): sha256(path) for path in inputs}
        actor = {"name": "frozen_dr_default_s0", "deploy": str(args.deploy.resolve()), "checkpoint": str(args.checkpoint.resolve())}
        cases = ("straight",) if args.phase == "smoke" else ((args.case,) if args.case else NAV.CASES)
        seeds = (470000, 470001) if args.phase == "smoke" else ((args.seed,) if args.seed is not None else (480000, 480001, 480002))
        rows = []
        for case in cases:
            for seed in seeds:
                for arm in ARMS:
                    output = args.output/f"{arm}-{case}-s{seed}"
                    sensors_created = []
                    def sensor_factory(runner, slot, owners):
                        sensors = STEREO.StereoSensors(runner, slot, owners, output=output/"stereo")
                        sensors_created.append(sensors)
                        return sensors
                    try:
                        row = NAV.episode(args.upstream.resolve(), actor, {"orb": orb, "lio": lio, "arm": arm}, case, seed, output,
                            seconds=8. if args.phase == "smoke" else 40., client_factory=HybridOrbNavigationClient,
                            world_factory=lambda c, s: STEREO.textured_world(c, s, output/"visual_assets"),
                            build_factory=STEREO.build_stereo_model, sensor_factory=sensor_factory,
                            method=arm, scope="Matched fresh-seed native stereo+LiDAR obstacle brake versus LIO-verified stereo-map recovery; both native clients and capture charged; 12-DoF frozen gait; simulation development only")
                    finally:
                        for sensors in sensors_created:
                            sensors.close()
                    row["episode_file"] = f"{arm}-{case}-s{seed}/episode.json"
                    recovery = json.loads((output/"native/recovery.json").read_text())
                    row["verified_map_recoveries"] = recovery["recoveries"]
                    rows.append(row)
                    print(json.dumps({key: row[key] for key in ("method", "case", "seed", "success", "fell", "collision", "verified_map_recoveries")}), flush=True)
        for path, checksum in inventory.items():
            if sha256(path) != checksum:
                raise ValueError("frozen stereo method input mutated")
        arms = {arm: {"episodes": sum(row["method"] == arm for row in rows),
                      "goals": sum(row["success"] for row in rows if row["method"] == arm),
                      "falls": sum(row["fell"] for row in rows if row["method"] == arm),
                      "contacts": sum(row["collision"] for row in rows if row["method"] == arm),
                      "verified_recoveries": sum(row["verified_map_recoveries"] for row in rows if row["method"] == arm)} for arm in ARMS}
        ready = all(row["native_tracked_frames"] > 0 and not row["nonfinite"] for row in rows)
        improved = (arms[ARMS[1]]["goals"] > arms[ARMS[0]]["goals"] and arms[ARMS[1]]["falls"] <= arms[ARMS[0]]["falls"]
                    and arms[ARMS[1]]["contacts"] <= arms[ARMS[0]]["contacts"])
        shard = args.phase == "development" and args.case is not None
        result.update(status=("PASS" if ready else "INCOMPLETE") if args.phase == "smoke" else ("PASS" if shard or improved else "NEGATIVE"),
                      episodes=len(rows), arms=arms, input_sha256=inventory, episode_files=[r["episode_file"] for r in rows],
                      measured_development_improvement=improved if args.phase == "development" and not shard else None,
                      paired_shard={"case": args.case, "seed": args.seed} if shard else None,
                      development_pair_outcomes=[{"case": r["case"], "seed": r["seed"], "arm": r["method"],
                          "clean_goal": bool(r["success"]), "falls": bool(r["fell"]), "contacts": bool(r["collision"]),
                          "verified_recoveries": r["verified_map_recoveries"]} for r in rows],
                      confirmation_required=True, hypothesis="LIO-verified coordinate recovery increases clean goals without increasing falls or contacts",
                      baseline_lio_mode="shadow_only_computation_charged", recovery_lio_mode="SE3_registration_and_validation_no_pose_fallback")
        if shard:
            result["scientific_status"] = "DEVELOPMENT_PAIR_NO_GLOBAL_VERDICT"
    except Exception as error:
        result["problems"].append(f"{type(error).__name__}: {error}")
    result["wall_seconds"] = time.monotonic()-began
    NAV.write_json(args.output/"campaign_result.json", result)
    print(json.dumps(result), flush=True)
    return 0 if result["status"] in ("PASS", "NEGATIVE") else 1


if __name__ == "__main__":
    raise SystemExit(main())
