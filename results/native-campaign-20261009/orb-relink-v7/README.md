# Actual native ORB runtime — genuine build PASS

**Slurm step 21739893.3** produced a real executable of ORB-SLAM3 pinned at
`4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4`, with actual version and complete
`ldd` dependency smoke passing inside the pinned Ubuntu SIF.

The preceding v6 step compiled every native source/object and linked the
ORB library, then exposed the missing search path for Ubuntu's private
`usr/lib/libgdal.so.30`. V7 verified all **17,248** original source/object,
header, library, build and package inputs (**688,358,561 bytes**) before
resuming the final link with both private library directories. Native source,
object and adapter bytes were retained; no thresholds/calibration changed.
Original failure, actual step identity, exact compiled-input inventory and
actual executable/runtime manifests are retained for independent audit.

- [Actual completion](actual-build/completion.json)
- [Actual runtime manifest](actual-build/runtime.json)
- [Actual dependency smoke](actual-build/ldd-runtime.txt)
- [Compiled input provenance](compiled-work-receipt.json)
- [Frozen source intake](intake.json)

Full native-build archive is durable at
`/nfs/stak/users/sanchej7/humanoid-native-20261009/orb-relink-v7/orb-native-relink-v7-20261009-21739893/outputs.tar.gz`,
SHA256 `40c0aa5e88b6de3b0f7bb7d4302b39df976a7221fbddee1535f44cb2881fa5f0`,
112,189,261 bytes. This page publishes compact receipts. Build PASS establishes
a genuine executable; trajectory/tracking/navigation outcomes require the
separate unchanged scientific campaigns.
