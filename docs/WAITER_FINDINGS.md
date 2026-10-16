# Two findings from the language-guided waiter

The task: a simulated Berkeley Humanoid Lite with grippers reads a sentence ("serve the blue tumbler to the right
table on the tray"), picks the named tumbler, puts it on a tray and carries the tray to the named table. A SmolVLA
policy maps the head-camera image, the sentence and the joint state to base-velocity, arm and gripper commands. The
full program, gates and limits are in [WAITER_PROGRAM.md](WAITER_PROGRAM.md); the ledger is `SLURM_JOBS.md`.

Scope for everything below: simulation only; the demonstrations come from a scripted expert that reads oracle
object poses; a stance lock holds the base while the robot stands; the instruction space is small (two colours,
two tables, six sentence templates); each learned result is one checkpoint from one training run. Numbers marked
"dev" are development measurements, not scored gates.

## 1. The first policy failed on 1.6% of its training data, not on the task

The first fine-tune scored **28/100** (bar 40). On dev layouts it finished the task 20 times in 23 once it held the
named tumbler, close to the scripted expert's 86/100, but it picked the named tumbler only 23 times in 40: chance.

Why the choice was not learned:
- **The choice happens early.** In the demonstrations the first arm has moved by frame 9 in every episode checked
  (60 of 60); from then on the joint state says which arm is working. Only the first ten or so frames need the
  colour in the sentence: about 1.6% of all frames. With uniform sampling, 40,000 training steps showed each of
  those frames about three times.
- **The choice is an exclusive-or.** Probes on the frozen vision-language features find the sentence's colour and
  "the red tumbler is on the left" each linearly readable (97-100%), but "the named tumbler is on the left" only at
  50-71% linearly (97-100% with a one-hidden-layer MLP). Neither feature alone says which arm to move.

The fix was to draw 30% of each training batch from the first 12 frames of episodes. On the same dev layouts:

| training | episodes | sampling | task successes (dev) | named tumbler among pick-ups (dev) |
|---|---|---|---|---|
| more demonstrations only | 644 | uniform | 30/64 | 95 of 189 |
| original demonstrations + early frames | 187 | 30% from the first 12 frames | 41/64 | 176 of 177 |
| both (phase 3b) | 644 | 30% from the first 12 frames | 48/64 | 192 of 192 |

The sampling change alone fixes the choice; the extra demonstrations add on top, most on a sentence template the
policy never trained on. **Scored (phase 3b): 75/100 with each layout's own instruction and 76/100 with the held-out
template** (bars 40 and 20), on 100 fresh layouts. Dev language controls: with the table word swapped, 0/64 episodes
succeed at the original table (58 carried the tray elsewhere); with the colour word swapped, the original tumbler
reaches the tray in 0/64.
Evidence: `results/waiter-20261005/phase3-gate/verdict.json`, `results/waiter-20261005/phase3b-gate/verdict.json`;
ledger entries "Phase 3b: why the tumbler choice is not learned" and "Ablation result (2026-10-11 11:08)".

## 2. RL on the policy's noise worked by steering two numbers

DSRL keeps the policy frozen and trains a small SAC agent to choose the initial noise of its flow-matching action
head. On the first policy it **passed: 63/100 against 23/100** for the same frozen policy with its usual noise, on
100 unseen layouts (exact one-sided McNemar p = 1.1e-8).

What it learned:
- **Two noise dimensions choose the tumbler.** With one random noise vector held for a whole episode, a logistic
  regression from the vector predicts which tumbler is taken in 83-85% of held-out episodes (52% by always guessing
  the commoner side). Two dimensions carry it: the noise on the left and right shoulder-roll actions.
- **The trained agent sets them from the sentence.** On dev layouts its policy took the named tumbler 48 times in
  48 (the base policy: 23 named, 17 the other, 8 none). Its remaining failures are in the carry and the set-down,
  where it does worse than the base policy.

On the retrained phase 3b policy, which already picks the named tumbler (192 of 192 on dev), the same method
**failed: 80/100 against 74/100** (p = 0.20); setting the noise to zero scores 82/100. The noise agent fixed the one
decision the supervised policy had not learned, and had nothing left to add once supervised training learned it.
Evidence: `results/waiter-20261005/phase4-gate/verdict_dsrl2.json`, `results/waiter-20261005/phase4b-gate/verdict.json`;
ledger entry "Constant-latent probe (2026-10-10 16:20)".

## What these results do not show

- Standing without the stance lock. With a free base the scripted expert completes 1/100 with the selected walking
  controller; the best lock-free pairing reached 76/100 against a bar of 80. As of 2026-10-15 a further attempt
  (a standing controller that observes its own drift) is training; see `SLURM_JOBS.md`.
- Anything about hardware: the camera is rendered, poses for the demonstrations are oracle, and no policy here has
  run on a robot.
- Breadth of language: two colours, two tables and six templates; the held-out template tests phrasing, not new
  objects or places.
