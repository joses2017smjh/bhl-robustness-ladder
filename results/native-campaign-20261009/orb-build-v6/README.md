# Native ORB build v6

Actual clean build execution: **Slurm step 21739893.2**, with two CPUs and
16 GiB on the already allocated dgx2-5 compute node. Separate backup batch
**21743261** is pending; continuation **21743355** depends on that backup.
The allocated step must never be described as execution of the backup batch.

Frozen v6 source archive SHA256:
`af21c73636082252084a84af909c447ee34b7e0004615dd517a64718e3514cb2`.
Durable freeze: `/nfs/stak/users/sanchej7/humanoid-native-20261009/orb-build-v6`.
Actual step outputs: `orb-native-build-v6-20261009-21739893` below that directory.

[Source-equivalence proof](source-equivalence.json) confirms that v6 adds only
the private sysroot multiarch include needed for OpenSSL's `opensslconf.h`.
The native adapters/build invocation and upstream estimator algorithms remain
unchanged. [The original allocated v5 failure](../orb-build-step-v1/failed-build/completion.json)
and its complete raw compiler log archive remain available. Failed/pending
builds have no estimated trajectory and establish no SLAM performance result.

The same immutable offline and stereo-navigation templates will be promoted
only after a genuine compiled executable, verified runtime/dependency hashes,
and actual version/ldd smoke pass. Scientific acceptance settings are not
changed in response to compiler failures.
