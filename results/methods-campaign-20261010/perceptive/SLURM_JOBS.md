# Perceptive training campaign — 2026-10-10

Declared before submission: one fresh GPU smoke (16 environments, 1 PPO iteration for teacher and each of four student arms, deployment parity and all four observation fault conditions), followed only after execution PASS by three independent training seeds in a GPU array capped at one running task. Full seed budget: 256 environments, 200 teacher PPO iterations, 100 iterations per student, 24 rollout steps, four update epochs. Maximum 8h/seed; 1 GPU, 8 CPUs, 64GB, no requeue.

A teacher must qualify on at least 80% of 256 first-episode 10s velocity trials before student training starts. This is an engineering development gate, not an independently validated performance threshold. All arms share fixed terrain levels and reset seeds. Evaluation changes dropout (0/50%) and delay (0/80ms), retaining the same development geometry. Paper performance is never substituted for measurements.

Warm start: original dr-default-s0 model_5999.pt SHA256 7fde8dc0ddee142974d9d640feee710496bbb2079e841bf0c7b0abe8fe79e7df. Students accept only 45 proprioceptive values plus causal body-relative measured point packets. Privileged height labels and true velocity are training-only.

Status: source QA 16/16 tests passed in allocated CPU step via stereo integration test batch; predeclared smoke awaiting freeze/submission. No training result yet.

Actual GPU smoke job **21757199** submitted 2026-10-10T20:38:06Z. Frozen archive SHA256 `1fd8ceb1997950147d167216dbac42b57663f4279288d6728ad7846a31970bca`; protocol SHA256 `24303034311f027cc5a441457d49f8eec2eaf352032a861f9b335762fac7c3e8`. Full seeds not submitted pending same-source smoke.

Before any GPU execution, cancelled pending smoke21757199: source review found Isaac deterministic terrain-column allocation would omit obstacles with4columns. Changed to5columns to realize exact40/20/20/20 proportions. V1 source and submission are retained; no model was trained or evaluated. New same-source smoke required under v2.

## Corrected protocol and actual smoke

V2 and V3 were never submitted. Review corrected the qualification metric: survival plus integrated gravity-aligned command-tracking error, rather than net displacement, now determines qualification. Historical freezes and exact differing source bytes remain retained. V4 uses three distinct serial seed jobs instead of a Slurm array.

Fresh V4 GPU smoke **21757582 PASS**, archive `aa2a33f65afcbe1046bab32b63ce4b4f187956e7efa2534d5e99af8b818b648f`, protocol `682e659289232a7b39b9fc1e6a6bd3ece8746157b62a63fa9da9b5387f8b4931`. It executed the teacher and all four student variants with deployment parity checks. Each student evaluated 64 smoke trials (16 environments × four dropout/delay conditions); these are excluded from development scores. The one-iteration teacher **did not qualify**.

The durable full-campaign controller is **21757622**, using source-matched smoke evidence and the frozen three-seed protocol. The first full seed was submitted as **21757635**. Full results remain pending. Targeted policy tests: 18 passed; strict three-seed collector tests: 10 passed.
