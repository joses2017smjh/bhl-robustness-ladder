# Sensor-method evidence collection — October 10, 2026

`scripts/bench/sensor_methods_collect.py` collects the predeclared **18 development cells, 2,700 camera frames and 54 native estimator runs**. It reports incomplete evidence explicitly and makes no navigation or physical-hardware claim.

The collector verifies the frozen source archive and protocol against their intake, checks all completed-job receipts, and hashes original sensor, evaluator-truth and native artifacts from each raw archive. It checks the pretrained model and actual CUDA-kernel execution evidence, both pinned native runtime receipts/binaries, all three estimator arms, 150 original frames per cell, native input boundaries and unchanged estimator configurations. A duplicate cell quarantines every copy; retries cannot be pooled selectively.

Raw archives may remain in their original job directories or in explicitly supplied publication-mirror directories. A mirror is accepted only when its complete bytes match the job's original completion checksum and publication lineage. This utility performs no downloads, credential lookup, submission, publication or rerun.

```bash
PYTHONPATH=src python scripts/bench/sensor_methods_collect.py \
  --campaign-dir /path/to/sensors-v1 \
  --job-dirs /path/to/sensors-v1/sensor-methods-run-*-20261010-* \
  --mirror-dir /path/to/verified-release-downloads \
  --output /path/to/new-sensor-summary.json
```

Collection is read-only apart from the new output file, which cannot overwrite an existing report. Run the archive verification in an allocated CPU job. Exit status `1` with a JSON `INCOMPLETE` report is expected while cells are missing or invalid. Missing raw archives cannot be replaced by a scheduler success flag or a summary JSON.

Analysis first averages frame metrics within each cell, retaining available-frame counts and unknown denominators. All three conditions—nominal, reduced texture and half LiDAR returns—remain within their original geometry group. Paired differences are averaged over the three conditions and bootstrapped over **six geometry groups**, never individual frames. Intervals are descriptive development estimates from previously used scene families. No global comparison is produced until all 18 cells validate; a metric absent in any required geometry group receives an unavailable estimate.

The report keeps native qualification counts, coverage and unresolved hazards alongside conditional error. It labels averages of per-cell p95 latencies as such; they are not a pooled p95 or a sensor-to-action latency. Fusion recall is measured per ground cell, and stereo obstacle recall is measured per pixel, not per obstacle instance.

Validation: **16 collector tests passed**, including changed native runtimes, missing frames, incorrect truth hashes, missing fusion arms, false CUDA claims, changed gates, visual contamination of the camera ablation, partial and duplicate cohorts, group-level uncertainty, corrupted mirrors and unsafe archives. The combined collector/dispatcher/sensor-map suite passed **36/36** in Slurm step `21756762.23`. A real empty-cohort run successfully verified the `sensors-v1` frozen source and runtime inputs and returned `INCOMPLETE`, 0/18 cells. Receipts are retained in [`results/methods-campaign-20261010/sensor-collector`](../results/methods-campaign-20261010/sensor-collector).
