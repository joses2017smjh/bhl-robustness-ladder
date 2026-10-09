# Actual stereo-estimated pose in the learned-gait navigation loop

[native_stereo_navigation_campaign.py](../scripts/bench/native_stereo_navigation_campaign.py)
requires a successfully compiled, pinned ORB-SLAM3 stereo runtime. It shares the
physical gait, fixed external routes, initial-station registration, measured
latency delivery and independent outcome evaluator with the native LiDAR–IMU
navigation experiment. The comparison is **stereo pose plus LiDAR obstacle
braking versus LiDAR–IMU pose plus LiDAR obstacle braking**. It does not claim
stereo-only obstacle sensing or end-to-end learned navigation.

Two fixed cameras are attached directly to the actual BHL base, at
`(.10, +/-.06, .38)` metres. Original images are 640×480, the stereo baseline is
12 cm and vertical field of view is 60°. Both cameras preserve the robot's full
roll, pitch and yaw; no free viewer camera or horizon correction generates
their inputs. The attach operation verifies that actuator, joint, sensor and
base slots retain the original learned-gait layout. Own-robot geometry is hidden
from both cameras, while the physical contacts remain unchanged.

Original RGB pairs are captured at causal 5-Hz scan boundaries. MuJoCo cached
camera fields belong to the preintegration time `data.time-physics_dt`; that
original timestamp accompanies each pair. Render and PNG-encoding costs are
measured and charged alongside actual native client wall time before a response
becomes available to control. Costs of images discarded while an earlier
response is pending are accumulated into the next request. A response cannot
change a command, tracking status or map-reset status before its charged arrival.

Only PNG paths and camera timestamps enter actual `TrackStereo`. LiDAR arrays,
IMU arrays and evaluator poses never enter ORB-SLAM3. Raw native `T_W_C`, tracking
state and map ID are retained. Fixed camera/body/IMU calibration converts valid
camera poses into the shared planner's `T_W_I` convention. Missing tracking or
map IDs keep the planner pose null. After a native map change, the planner
permanently stops; the original native pose is still preserved for diagnosis.

The external route and nominal start station are declared before each episode.
The initial native origin is registered to that start convention without
reading evaluator poses. Stale, lost, unregistered and reset estimates stop
commands. The existing raw LiDAR brake stops on nearby endpoints or inadequate
scan support. Simulator poses are used exclusively to generate sensors and
evaluate goals, contact, tilt and falls. Goal stopping uses estimated pose.

The smoke contains two actual eight-second straight-route episodes, groups
370000 and 370001. Each must produce actual tracked native poses and finite
physics, while goals are not required in that short smoke. The development
cohort uses the same three straight/dogleg/occluder route cases and groups
380000–380002 as the LiDAR–IMU experiment: nine 40-second episodes with one
frozen DR-default-s0 actor. Progress, goals, contacts, falls, tracking failures,
native map changes, client/capture latency and stale/lost stops are retained.
Development may proceed only after a fresh smoke from the same frozen source,
model and native-runtime archive. All nine clean full-horizon goals are required
before proposing an independent confirmation cohort. This is a development
gate, not confirmation or a physical safety claim.

Visual texture is generated from the declared group seed plus 910000 before
rendering. The exact PNG and world XML hashes are retained. Texture alters
visual features, with unchanged route/collision geometry. The ORB feature recipe
is frozen at 1200 features, eight pyramid levels, 1.2 scale factor, and FAST
thresholds 20/7; failed initialization is reported instead of lowering thresholds
after inspecting results.

Each episode retains original stereo PNGs and their manifest, actual timed raw
LiDAR/IMU arrays, every native response, calibration/configuration/runtime
receipts, commands, charged response-arrival times and separate evaluator truth.
This supports independent drift and outcome review. Native algorithm runtime
construction alone supplies no navigation result. Cameras, rays and IMU are
simulated; no hardware accuracy, terrain traversal or NavGym-actor claim follows
from this experiment.

The [ORB-SLAM3 paper](https://arxiv.org/abs/2007.11898) describes stereo tracking
and multiple-map recovery. This experiment uses the authors'
[original implementation](https://github.com/UZ-SLAMLab/ORB_SLAM3) and records
native map changes because a recovered map can change the frame assumed by a
navigation controller. Permanently stopping after that change is this
experiment's declared integration policy; it is not a limitation claimed by
the paper, which supports map merging.
