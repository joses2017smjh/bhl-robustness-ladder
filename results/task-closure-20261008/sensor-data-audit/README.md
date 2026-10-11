# Retained sensor data audit — 2026-10-08

This is a read-only audit of nine published JSON artifacts, not a new sensor
capture, simulator run, training campaign, or perception accuracy result.
Every input's SHA-256 is recorded in [audit.json](audit.json).

Run from a repository checkout using Python's standard library:

```bash
python scripts/bench/sensor_dataset_audit.py --out /tmp/new-sensor-audit.json
python -m pytest -q tests/test_sensor_dataset_audit.py
```

The output path must be new. Supply explicit JSON paths before `--out` to audit
a subset. The program recognizes the existing inspection episode/trace and
maze episode-summary representations; it does not certify arbitrary new
dataset manifests. Its report schema is `bhl-sensor-dataset-audit-v1`.
An integrity failure writes its diagnostic report and exits with status 1.
`NOT_READY` alone is informational and does not cause a failure exit.

Nine artifacts pass logged timestamp, shape, range and declared-source
integrity checks. Seven retain sparse sensor snapshots: 36 LiDAR sector minima,
paired 8×8 idealized ray depths, and ten-float IMU feature packets. Their
snapshots are logged one second apart, while the declared exteroception rate
is 10 Hz. The other two artifacts retain 0.2-second controller/pose traces but
no framewise sensor measurements. Videos cannot recover the missing metric
arrays.

In `inspection-dropout035-s3-14.json`, the 265 logged snapshots comprise 237
fresh exteroception packets, 26 stale packets and two absent packets; the
maximum logged capture age is 0.4 seconds. These counts describe the sparse
saved snapshots, not all policy ticks, a packet loss probability, or measured
hardware latency. Absent packets in the `off` control arms of other files are
expected; zero-filled policy features are not raw sensor measurements.

All nine artifacts are **NOT_READY** for raw-stereo, scan-matching or calibrated
multisensor dataset experiments. The saved depth is ray-cast simulation depth,
not image correspondence; scans are pooled sectors, not the original ranges
and angles. Original RGB pairs, per-eye/per-ray timing, hardware calibration,
explicit hit masks, clock/receive timing and a full sensor stream are missing.
Inspection navigation declares oracle simulator localization. Zero observed
IMU age here is a simulation timestamp property, not a hardware requirement
already satisfied. Separate public probes/videos reuse episodes; do not pool
these files as independent trials.

[negative-controls.json](negative-controls.json) records six refused
in-memory corruptions of one original published snapshot: future time, stale
measurement hidden behind current receive time, absent packet marked fresh,
wrong depth dimensions, a nonfinite range, and relabeling ray depth as physical
stereo. The eleven targeted tests also check source hashes, summaries without
sensor arrays, refusal to overwrite an existing report, and a nonzero CLI exit
with a retained diagnostic report for corrupted data. None of these
corruptions is presented as measured perception performance.
