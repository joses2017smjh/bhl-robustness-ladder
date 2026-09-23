# Jose Sanchez

I build robot-learning simulations and perception systems, then test where
they fail. My work connects Isaac Lab, MuJoCo, PyTorch and reproducible HPC
experiments.

**Seeking robotics ML / simulation engineering roles.** Oregon State University.

[Portfolio](https://jose-sanchez-portfolio-com.vercel.app) ·
[Robotics résumé](https://jose-sanchez-portfolio-com.vercel.app/resumes/resume_robotics.pdf) ·
[Email](mailto:sanchej7@oregonstate.edu)

[![Humanoid completes an ordered inspection route; known waypoints, learned gait and sensor braking](https://raw.githubusercontent.com/joses2017smjh/bhl-robustness-ladder/main/docs/gifs/weekend-inspection.gif)](https://github.com/joses2017smjh/bhl-robustness-ladder#readme)

## Selected engineering work

### [Humanoid simulation and evaluation](https://github.com/joses2017smjh/bhl-robustness-ladder)

Versioned PPO tasks, cross-engine gait evaluation, sensor checks and Slurm
promotion gates. The repaired known-route Isaac task scores **379/384** final
first episodes. Separate MuJoCo missions demonstrate two-/three-robot
synchronization with explicit negative controls. Known waypoints are not
autonomous visual navigation.

### [Vision-guided robotic pruning](https://github.com/joses2017smjh/isaac-sim-pruning-workflow)

UR5e simulation with wrist RGB-D tracking, dual ToF and an independent capture
grader. A 20-second recording shows approach, gated rigid-spur release and
return; a failed tracking sequence shows motion stopping without release.
One known target, not learned tree recognition or wood fracture.

### [SPUR metric-depth service](https://github.com/joses2017smjh/spur-depth-service)

Synthetic orchard depth from PyTorch models through FastAPI and split ONNX
inference. Includes named-GPU latency measurements, numerical parity checks,
and the accuracy loss when predicted masks replace ground truth. Synthetic
validation is not field accuracy.

## How I work

I keep failures and retractions next to the successful demos. A completed
training job is not a completed robot task; every headline should lead to a
protocol, a result file and a stated limit.

Python · PyTorch · Isaac Lab/Sim · MuJoCo · ONNX Runtime · OpenCV · Slurm · Apptainer
