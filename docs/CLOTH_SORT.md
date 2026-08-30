# Sorting garments into baskets

## The stack question, which your folding repo already answered — for 5.1

`IsaacSimFolding` pins Isaac Sim **5.1** because 6.0 removed PhysX's
particle-based cloth, and works around 5.1's RTX segfault on this cluster by
rendering through the OpenUSD **Storm** rasterizer instead. The cost it records
is that Storm only evaluates `UsdPreviewSurface` while Omniverse assets ship
MDL, so robots come out flat white — and physics is **CPU-only, one env per
process, no `NUM_ENVS` to turn up**.

That last line is what rules out RL on that stack. This project's terrain rungs
train at 4,096 envs for 6,000 iterations. At one env per process the same sample
budget is roughly four thousand times the wall clock, which is not a scheduling
problem, it is a different order of magnitude.

**6.0 does not have that constraint.** It dropped *PhysX particle* cloth and
ships **Newton 1.2** in its place, with VBD and Style3D solvers, plus
`DeformableObjectCfg` and an official cloth manipulation task at
`isaaclab_tasks/manager_based/manipulation/lift_franka_soft/`. That task runs
`CoupledMJWarpVBDSolverCfg` — MuJoCo-Warp for the rigid bodies, VBD for the
cloth, one-way coupled — on GPU. And 6.0's RTX renderer works on this cluster;
it is the stack the nine v2 cells already target.

So cloth and rendered RGB are available together, on GPU, on one stack. The 5.1
pin was correct for the particle path and does not carry over.

**What is still unknown is throughput.** Cloth is expensive and the Franka
example is a single-arm tabletop scene. Whether a humanoid plus five garments
runs at 64 envs or at 1,024 decides whether this is an RL task or a scripted
demonstration, so it gets measured before anything is designed around it.

## The morphology constraint, and what it forces

No fingers. 4 Nm arms. Hands that do not adduct past 0.355 m. Measured static
holding capacity is 1.59 kg per arm — the payload was never the problem, the
grip was.

**A humanoid with no fingers cannot pick up a sock.** Not with a better policy,
not with more iterations: two flat hands cannot pinch compliant fabric with any
reliability, and the friction pinch that half-works on a rigid cube depends on
squeezing against a rigid opposite face that cloth does not have.

So the task is **sweeping, not pick-and-place**. The garments start on a table
at `GRASP_Z`-height, the baskets sit below the table's front edge, and the robot
pushes each garment off its own edge segment into the basket underneath. That is
a non-prehensile action this morphology can actually perform, it still requires
recognising which garment is which, and it still requires positioning the body —
which is the part vision has to earn.

Sorting **by pushing** is also how the capability reads honestly in a demo: it
does not imply a gripper the robot does not have.

## Layout

| | |
|---|---|
| garments | 5 — two socks, two shirts, one jacket |
| baskets | 3, one per type, below three edge segments |
| table | top at 0.30 m, the measured squat band |
| start | robot spawns 2.0 m away and walks to the table |
| sensing | head camera, RGB **and** depth, same pose |

Five garments and three baskets means at least one basket takes two items, so
the policy cannot succeed by memorising a one-to-one layout — it has to
condition on what it sees.

## Staging

Walking and sweeping are different problems and this project has already learned
what happens when a reward pays for both at once. Staged:

1. **approach** — reach a marked footprint in front of the table
2. **address** — square up to the correct edge segment for the garment in hand
3. **sweep** — push one garment off the edge
4. repeat for the remaining four

Success terminates the episode, as in the other v2 tasks.

## G-C1, before any of it trains

* Does a Newton cloth scene with this humanoid construct, reset and step on v60?
* What is the throughput at 16 / 64 / 256 / 1024 envs? The curve decides RL
  against scripted demonstration.
* Does RGB render in the same frame the cloth simulates, or does enabling
  cameras change the solver path?
