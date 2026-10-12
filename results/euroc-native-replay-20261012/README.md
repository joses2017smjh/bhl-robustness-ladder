# EuRoC replay evidence

See [the result report](../../docs/EUROC_NATIVE_REPLAY_RESULTS_2026-10-12.md).
`smoke/` and `full/` preserve complete derived native frame traces, native
settings, selection/calibration hash receipts and execution metadata. They
contain no original dataset imagery, measured sensor vectors, point clouds,
or ground-truth poses.

`operations/` contains byte-for-byte snapshots of the acquisition, identity
check, Slurm launcher, independent rescorer and figure generator used from the
campaign directory. These are execution records: the worker paths in the Slurm
wrapper and launch plan identify this specific HPC campaign. They are not a
portable install package. To reproduce elsewhere, use the existing
`euroc_native_validation.py prepare` and `run` commands documented in the
[adapter handoff](../../docs/EUROC_FOLLOWUP_2026-10-12.md), with the pinned
protocol/runtime, independent intake verification, and a fresh same-source
40-pair native smoke before full replay. Prior outputs are immutable.

The frozen scientific snapshot is reproducible from repository commit
`1e973fbcfa582849150bc1c0116c25184ba4a77c`:

```bash
git archive 1e973fbcfa582849150bc1c0116c25184ba4a77c \
  src scripts tests results/euroc-external-validation-20261011/protocol.json \
  > source.tar
```

Expected archive SHA256:
`3a6dbfe33afe4871df1e7ef1d2ecb6fcd77c6c9bd2cab30a86f81bcdafca5beb`.
The complete per-Python-file source fingerprint is inside each native result.
Acquisition/reference receipts identify the original public sources and hashes;
no original dataset bytes are included here.

The native runner retains `ate: UNMEASURED`. Independent verification confirms
execution, tracking/reset coverage and timing; it does not score pose accuracy.
Original EuRoC V1_01_easy reference orientation has a documented issue, and the
mirror's complete image-byte identity to the blocked ETH master is unverified.

The published SVG has whitespace at line ends removed; plotted data and shapes
are unchanged from the generated figure.
