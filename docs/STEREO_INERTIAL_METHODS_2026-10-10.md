# Native stereo–IMU follow-up, 2026-10-10

This follow-up adds an actual ORB-SLAM3 `IMU_STEREO` comparison to the existing
stereo experiment. The pinned upstream estimator is
[`4452a3c4`](https://github.com/UZ-SLAMLab/ORB_SLAM3/tree/4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4).
The implementation follows upstream's [`TrackStereo` IMU interface](https://github.com/UZ-SLAMLab/ORB_SLAM3/blob/4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4/include/System.h)
and [inertial configuration parser](https://github.com/UZ-SLAMLab/ORB_SLAM3/blob/4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4/src/Tracking.cc).

`scripts/native/orb_runner.cc` now accepts `--stereo-inertial`. It passes original
camera pairs and timestamped accelerometer/gyroscope measurements to the native
estimator; the original stereo-only command remains compatible. The client checks
that the runtime explicitly enabled the requested mode and acknowledges every IMU
batch with the original sample count and first/last timestamps. A stereo-only
runtime cannot silently stand in for an inertial result.

The mounting transform is `T_imu_camera = inverse(T_B_I) @ T_B_C`, from declared
sensor calibration. IMU values use radians/second and specific force in m/s²,
including the accelerometer's gravity response. Each sample is supplied once,
with no pose-derived replacement, interpolation, or future samples. Missing IMU
coverage, gaps above 20 ms, conflicting repeated packet boundaries, nonfinite
measurements, and evaluator arrays in inference packets fail before inference.

`src/bhl_robust/research/native_orb_inertial.py` contains transport validation and
native YAML generation. `scripts/native/orb_inertial_replay.py` executes either
mode or a matched pair with original sensor-time wall pacing, preserves failed
tracking frames, and records source/input/runtime hashes. The diagnostic wrapper
build helper relinks against the unchanged native SLAM library and retains the
original feature thresholds.

```bash
PYTHONPATH=src python scripts/native/orb_inertial_replay.py \
  --manifest /path/to/capture/replay/manifest.json \
  --sequence ramp_step_native --runtime /path/to/runtime \
  --mode paired --output /new/output/directory
```

The predeclared engineering noise densities are gyro 0.001, acceleration 0.01,
gyro random walk 0.00001 and acceleration random walk 0.0001, at 200 Hz. These are
fixed estimator settings, not measured physical sensor calibrations. The existing
replay capture contains ideal simulated IMU; success would not establish hardware
robustness.

The native output reports `map_imu_initialized` separately from visual tracking.
`Tracking::OK` alone is insufficient to call inertial initialization successful.
Execution `PASS` means that the declared native interface ran to completion;
initialization failures remain scientific negatives. The initial diagnostic uses
previously consumed simulation scenes and cannot support held-out or generalization
claims. New navigation qualification requires a separately frozen campaign.

Focused verification: 36 transport/configuration and existing stereo compatibility
tests plus four navigation boundary tests passed with the shared compute allocation
and explicit local temporary paths. The actual native relink, 35-frame paired
smoke, and complete 900-frame paired replay all executed successfully. Execution
success does not change the negative inertial-initialization result below.

## Navigation implementation

`scripts/bench/stereo_inertial_navigation_campaign.py` adds a separate paired
navigation campaign using the existing body-attached stereo cameras, actual
MuJoCo gyroscope/accelerometer sensor values, frozen gait actor, native timing,
and raw LiDAR obstacle brake. A native visual `OK` pose is rejected by the
inertial arm until the native map reports successful IMU initialization. A map
frame change subsequently stops navigation. Four focused tests verify those
control boundaries and the fixed camera-to-IMU transform.

The frozen smoke design uses two fresh seeds (570000 and 570001), both estimator
arms, and 20 seconds per episode. The development design has 18 paired episodes:
three route fixtures, seeds 580000–580002, both arms, and 40 seconds per episode.
Development requires a matching source/input smoke receipt with actual inertial
initialization, accepted navigation poses, full episode horizons, and no falls,
contacts, or nonfinite state. A negative initialization smoke does not authorize
that development stage. The runtime can be fetched into worker-local storage
from its fixed release URL, bounded to 256 MiB and verified against SHA-256.

This comparison retains 640×480 ideal pinhole images at 5 Hz. Camera-rate,
physical exposure, and motion-blur ablations are not implemented here. Additional
initialization excitation or any threshold/noise tuning requires a new declared
experiment; the historical replay and its native feature thresholds are kept
fixed. The source dataset's original split labels are retained in receipts, but
reusing those scenes makes this follow-up diagnostic, including its originally
labelled test scene.

The [native runtime](https://github.com/joses2017smjh/bhl-robustness-ladder/releases/download/methods-campaign-evidence-20261010/orb-stereo-inertial-runtime-v1-20261010.tar.gz)
and [exact replay source freeze](https://github.com/joses2017smjh/bhl-robustness-ladder/releases/download/methods-campaign-evidence-20261010/stereo-inertial-source-v1-20261010.tar.gz)
were uploaded and independently downloaded for SHA-256 verification. Compact
[build, test, smoke, and publication receipts](../results/methods-campaign-20261010/stereo-inertial/)
remain in the repository.

## Completed native navigation smoke — job 21757768

All four fresh-seed simulation episodes completed their 20-second horizons with
runtime exit code **0**. The scientific readiness result is **NEGATIVE**:
neither stereo–IMU episode initialized. `gate_to_development` is false, and all
**18 planned development episodes remain unrun**. No parameters were tuned and no
replacement experiments were launched after that result.

| Native navigation arm | Completed episodes | Accepted native pose responses | IMU-initialized frames | Goals in smoke | Falls / contacts |
|---|---:|---:|---:|---:|---:|
| Stereo | 2/2 | 177/177 | Not applicable | 0/2 | 0 / 0 |
| Stereo–IMU | 2/2 | 0/178 | 0/178 | 0/2 | 0 / 0 |

The two inertial episodes retained `IMU_INITIALIZING_STOP` for all 89 native
responses each; their commanded velocity remained zero. All four episodes had
finite state and no map-frame changes. The baseline retained 89 and 88 native
responses, respectively, under the measured-latency delivery rule. These are
initialization smokes; the goal count within 20 seconds is not a success rate
for the unrun 40-second development campaign.

Execution completion and scientific qualification are separate here. The native
runtime completed cleanly, while the orchestration wrapper returned 1 with
`FAILED_SCIENTIFIC_READINESS_GATE` to prevent follow-on development after the
negative result. It did not report a native runtime exception.

The [reviewed episode summary](../results/methods-campaign-20261010/stereo-inertial/navigation-summary.json)
links the frozen source, native stream receipts, and raw output hashes. The
[raw navigation archive](https://github.com/joses2017smjh/bhl-robustness-ladder/releases/download/methods-campaign-evidence-20261010/stereo-inertial-nav-remote-v1--stereo-inertial-nav-smoke-20261010-21757768.tar.gz)
contains the camera images, IMU/ray measurements, traces, evaluator data, and native
logs. Its 147,993,306 bytes were published and independently downloaded for
verification; this review downloaded it again and confirmed SHA-256
`b4e4f47be2826412d48063bd254c0425767522d64a7489a299aab6cae134887c`.
The campaign, launch, runtime-log, and native-stream hashes were also checked
against their parent receipts. The [track ledger](../results/methods-campaign-20261010/stereo-inertial/LEDGER.md)
records the completed runs and the gated, unrun development stage.

## Complete native replay result

The exact same three previously consumed 150-frame sequences were replayed through
both native modes at their original 5 Hz cadence. No thresholds or noise settings
were tuned between the short smoke and the full replay.

| Original scene | Stereo tracked | Stereo–IMU tracked | Native IMU initialization | Stereo–IMU native p95 |
|---|---:|---:|---:|---:|
| Textured boxes | 0/150 | 0/150 | 0/150 | 30.08 ms |
| Thin posts | 0/150 | 0/150 | 0/150 | 33.26 ms |
| Ramp/step | 122/150 | 0/150 | 0/150 | 29.97 ms |

Each inertial replay consumed 5,961 original timestamped IMU measurements. Native
logs contain no missing-IMU warnings. The first two scenes never exceed the
unchanged initializer requirement of more than 500 features (maxima 463 and 427).
The ramp/step scene exceeds that feature threshold on 11 frames, and the native
estimator logs insufficient acceleration on all 11 initialization attempts.
Upstream requires a change of at least 0.5 m/s² between successive preintegration
mean accelerations unless its fast-initialization option is enabled; this
experiment leaves that option disabled.

The current ideal, smooth replay therefore does **not** demonstrate that adding
IMU resolves stereo initialization. A justified next data-collection experiment
would predeclare visible texture and safe acceleration excitation, then measure
initialization latency, success rate, drift, and closed-loop outcomes on fresh
seeds. The independently implemented body-sensor navigation smoke also failed actual
inertial initialization, so its 18 planned development episodes remain unrun. These measurements are simulation development evidence, unsuitable
for a resume claim of improved stereo–IMU accuracy or hardware robustness.

The [full initialization analysis](../results/methods-campaign-20261010/stereo-inertial/initialization-analysis.json),
[execution receipt](../results/methods-campaign-20261010/stereo-inertial/execution.json),
and [raw replay evidence](https://github.com/joses2017smjh/bhl-robustness-ladder/releases/download/methods-campaign-evidence-20261010/stereo-inertial-replay-evidence-v1-20261010.tar.gz)
preserve both successes and failures.
