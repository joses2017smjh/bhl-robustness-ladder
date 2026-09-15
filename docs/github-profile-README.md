# Jose Sanchez

MS Artificial Intelligence, Oregon State (2026). I measure robot learning
systems until the measurement is the result — including when that means
retracting my own numbers.

**Seeking:** robotics ML / simulation engineer, applied scientist (perception & control).

[Portfolio](https://jose-sanchez-portfolio-com.vercel.app) ·
[Email](mailto:sanchej7@oregonstate.edu) ·
Resume (PDF on the portfolio)

Python · PyTorch · Isaac Lab · MuJoCo · ONNX · computer vision · RL · Slurm/HPC

## Featured work

### [bhl-robustness-ladder](https://github.com/joses2017smjh/bhl-robustness-ladder)

Train a 3D-printed humanoid in Isaac Lab, score it in MuJoCo. 156 policies,
6,348 sim2sim episodes. Transfer inverts the training ranking (23% falls vs
0%). Four of thirteen findings retract earlier claims in the same repo. The
free-standing robot sorts 22/24 rigid garment proxies with no falls.

### [isaac-sim-pruning-workflow](https://github.com/joses2017smjh/isaac-sim-pruning-workflow)

UR5e pruning in Isaac Sim: two Blender trees, live RGB-D tracking, dual-ToF
sensing, recorded surrogate-release successes and failures.

### [Vision-Based Metric Depth Estimation for Robotic Pruning](https://github.com/joses2017smjh/Vision-Based-Metric-Depth-Estimation-for-Robotic-Pruning)

Metric depth for a pruning cut, served as an API — metres, not a pretty PNG.
DINOv2 RGB+D refinement; Torch vs ONNX agreement at 1.5e-5. Numbers and
latency are on the [portfolio](https://jose-sanchez-portfolio-com.vercel.app).

### [IsaacSimFolding](https://github.com/joses2017smjh/IsaacSimFolding)

Reproducing *Learning to Fold* in Isaac Sim on a cluster whose pinned renderer
segfaults — flow-matching VLA, RECAP+AWR, Thompson-sampled inference.

### [Agentic-Soccer-Match-Prediction-MCP](https://github.com/joses2017smjh/Agentic-Soccer-Match-Prediction-MCP)

Calibrated forecasts over MCP, with a GRPO staking policy and human approval.
World Cup 2026 holdout numbers are on the portfolio, not restated here.

## How I work

I would rather publish a retraction than a number I cannot replay. The
humanoid repo's stereo cameras looked 20° up for an entire rung because Isaac
Lab 3.0 reads `(x, y, z, w)` and the pose was written `(w, x, y, z)`. The
fix is in the tree; so is the voided table.

## Contact

sanchej7@oregonstate.edu · Oregon State University, Corvallis
