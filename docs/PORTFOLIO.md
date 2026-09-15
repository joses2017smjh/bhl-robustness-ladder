# Portfolio and GitHub presence — copy to paste

This file is the recruiter-facing pack for
[jose-sanchez-portfolio-com.vercel.app](https://jose-sanchez-portfolio-com.vercel.app)
and github.com/joses2017smjh. The research repo stays the technical proof.
Numbers below are already published in this repository or on the live site;
nothing here is invented.

**Target roles:** robotics ML / simulation engineer, applied scientist
(perception and control).
**Stack to lead with:** Python, PyTorch, Isaac Lab, MuJoCo, ONNX, RL, metric
depth, Slurm.

---

## 1 · Prioritized audit

GitHub profile has **no profile README** (`joses2017smjh/joses2017smjh` 404).
Pinned repos currently surface older work (`AgCV`, `HomeWorkoutApp`) above the
humanoid and pruning stacks. `bhl-robustness-ladder` had an empty topics list
and a description that stopped at push recovery.

The portfolio already has a clear visual language and measured project cards.
Gaps:

| issue | why it costs interviews |
|---|---|
| Hero says “computer vision, 3D perception, and LLM systems” | the strongest 2026 work is **robot learning + measurement**, not LLM systems |
| BHL card still says **89 policies** | the repo now reports **156**; recruiters who click through see a mismatch |
| BHL card omits cloth-sort and the retractions | those are the two most distinctive claims |
| FIG. 02 is `multi_race` only | the lab-floor + depth GIF and the free-standing sort are stronger 15 s demos |
| Coursework projects (PointNet, DoM) sit equal with shipped stacks | they should be one click below, not in the first fold |
| No “I am looking for X” line | a recruiter has to infer the role |

This repo's README was already technically credible. It did not say who wrote
it, what role they want, or walk a recruiter through four case studies in
under a minute.

---

## 2 · Pin these five (in this order)

1. **bhl-robustness-ladder** — this repo
2. **isaac-sim-pruning-workflow**
3. **Vision-Based-Metric-Depth-Estimation-for-Robotic-Pruning** (or `spur-depth-service` if that is the served API)
4. **IsaacSimFolding**
5. **Agentic-Soccer-Match-Prediction-MCP**

Unpin or leave unpinned: `AgCV`, `HomeWorkoutApp`, homework repos,
`Joseswebsite`.

Profile README: copy [github-profile-README.md](github-profile-README.md) into
a new public repo named `joses2017smjh`.

---

## 3 · Hero copy (replace the current opening)

**Eyebrow:** MS Artificial Intelligence, Oregon State · 2026

**H1:** I measure robot learning until the number survives a second simulator.

**Sub:** Isaac Lab in, MuJoCo out. Perception APIs that return metres. I am
looking for a robotics ML / simulation role.

**Primary CTA:** View the humanoid ladder → `#humanoid` / GitHub
**Secondary:** Resume · Email

Drop “LLM systems” from the first sentence. Soccer and MetaNaviT stay on the
page; they are not the lead.

---

## 4 · Humanoid Robustness Ladder — project card

**Replace the current body** (it still says 89 policies).

**Title:** Humanoid Robustness Ladder

**One line:** How much disturbance an 11.3 kg, 6 Nm humanoid can take before it
stops learning — scored in a simulator it never trained in.

**Problem.** Upstream ships flat-ground PPO with no curriculum and no scoring
path. A policy that only works in PhysX has learned the solver.

**Solution.** Train in Isaac Lab (4,096 envs). Export ONNX. Replay through
upstream's own controller in headless MuJoCo. Gates refuse a verdict without a
control.

**Contribution.** The measurement stack: sim2sim harness, 13 findings, four
retractions left in public, including a stereo rung whose cameras pointed 20°
up because Isaac Lab 3.0 reads quaternions `(x, y, z, w)`.

**Result.** 156 policies, 6,348 scored episodes. Highest training reward falls
23% in MuJoCo; the repo default falls 0%. At terrain `d = 1.0`, 22-DoF falls
11.7% against the biped's 37.8%. The free-standing robot sorts 22 of 24 rigid
garment proxies with no falls.

**Tech:** Python, Isaac Lab, MuJoCo, PPO, rsl-rl, ONNX, Warp, Slurm

**Links:** [GitHub](https://github.com/joses2017smjh/bhl-robustness-ladder) ·
[README](https://github.com/joses2017smjh/bhl-robustness-ladder#readme) ·
[Findings](https://github.com/joses2017smjh/bhl-robustness-ladder/blob/main/docs/FINDINGS.md)

**Hero visual:** `docs/gifs/multi_lab.gif` (lab floor + depth band). Secondary:
`docs/gifs/isaac/cloth_sort_free_base_shirt.gif`.

---

## 5 · GIF / demo shot lists (15–30 s)

### Humanoid ladder — recruiter loop

1. **Problem (0–6 s):** `multi_race.gif` — four policies, one shove; the
   un-randomized robot is already on the ground.
2. **Impressive flow (6–16 s):** `multi_lab.gif` — orange 22-DoF robot clears
   cable / threshold / ramp; depth waterfall along the bottom.
3. **Differentiator (16–22 s):** freeze on the red border when the push-trained
   policy falls; caption “same clock, not a composite.”
4. **Result (22–30 s):** `cloth_sort_free_base_shirt.gif` — two sweeps, shirt
   in the basket, robot still standing. Caption “22/24 rigid proxies, 0 falls.”

### Pruning depth (from the live site's FIG. 01)

1. Field camera on dormant wood (starting state).
2. RGB-D overlay with a cut in metres.
3. API response / Torch vs ONNX delta.
4. Latency number on V100.

Do not use `ice_pair.gif` as a win. The ice result is retracted.

### Cooperative lift — only as a failure case

`carry_cube_pov.gif`: 7.8 cm lift, then the 41 cm collapse. Caption it as a
negative result or skip it on the homepage.

---

## 6 · Exact GitHub repo metadata

```
Description:
Train a 3D-printed humanoid in Isaac Lab, score it in MuJoCo. 156 policies,
6,348 sim2sim episodes, 13 findings — four of them retractions.

Homepage:
https://jose-sanchez-portfolio-com.vercel.app

Topics:
reinforcement-learning, isaac-lab, mujoco, humanoid-robotics, sim-to-sim,
robotics, pytorch, computer-vision, domain-randomization, locomotion
```

---

## 7 · Website structure (keep the current visual system)

1. Hero (copy above) + one looping GIF (`multi_lab.gif`)
2. Featured: Humanoid ladder, pruning depth, pruning workflow
3. Also: IsaacSimFolding, Soccer MCP, MetaNaviT
4. Coursework (PointNet, DoM) under “Also”
5. Beyond / now / resume — unchanged
6. Footer: GitHub, LinkedIn, email, resume

Each featured card: problem · solution · contribution · result · tech ·
GitHub. No skill-percentage charts. No “innovative / scalable.”

---

## 8 · Recruiter-style quality check

| check | this pack |
|---|---|
| Role is stated | robotics ML / simulation, applied scientist |
| Strongest project in <5 s | `multi_lab.gif` + 23% vs 0% |
| Claims match the repo | 156 policies, 22/24 sort, ice/stereo retractions named |
| Failures shown | lift 7.8 cm, zero task success, four retractions |
| Contact | email + portfolio + GitHub |
| No invented metrics | all figures from FINDINGS.md or the live site |
