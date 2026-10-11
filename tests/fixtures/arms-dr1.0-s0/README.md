This is the existing frozen `arms-dr1.0-s0` learned gait used by the scripted
cooperative physics tests: 22 joint actions from 75 observations, at 25 Hz.
The fixture supports regression comparisons and carries no new training or
task-success claim.

`policy.onnx` and `deploy.original.yaml` preserve the original export bytes.
`deploy.yaml` changes only the checkpoint path to `policy.onnx`; tests resolve
that path relative to this directory. Robot geometry comes from the pinned
recursive Berkeley Humanoid Lite assets submodule, rather than a shared cache.
`provenance.json` records the source run, exact hashes, transformation and
upstream revisions. `LICENSE-upstream.txt` preserves Berkeley Humanoid Lite's
copyright and MIT license notice for its robot/control code.
