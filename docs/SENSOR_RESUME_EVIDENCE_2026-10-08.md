# Sensor campaign evidence for robotics and AI applications

This campaign adds useful perception engineering evidence to the Humanoid project: calibrated stereo geometry, reproducible inference, sensor disagreement handling, timestamp-aware estimator interfaces, and 3D mapping. The implementation currently supports a **short simulated kinematic replay pilot**. Physical sensor accuracy, native SLAM trajectories, estimated-pose navigation, and terrain traversal need their own completed runs before they become resume results. Current execution status belongs in [the campaign record](SENSOR_CAMPAIGN_2026-10-08.md); this document does not promote a queued or failed run into an achievement.

## Official job requirements checked on 2026-10-08

The following are skill references, not a ranked application list or a conclusion that the applicant meets every qualification. The displayed pages do not expose reliable publication dates. Retrieval dates and crawler freshness below describe the evidence available during this review.

| Employer and exact posting | Requirements relevant to this work | Availability evidence and practical fit |
| --- | --- | --- |
| [Apptronik — Staff Perception Hardware Engineer](https://job-boards.greenhouse.io/apptronik/jobs/5988546004) | Camera/LiDAR/IMU synchronization, intrinsic and extrinsic calibration, stereo geometry, and instrumented validation of perception hardware. | Direct official page retrieved on 2026-10-08 displayed the role and application form; tool reported a crawl today. This is a senior hardware role requiring extensive shipped-system experience. Simulated geometry work supports related skills; physical integration and timing measurements remain a separate requirement. |
| [Neuralink — Software Engineer, Robotics](https://job-boards.greenhouse.io/neuralink/jobs/4205469003) | Reliable robot software, Linux and C/C++/Rust, hands-on robotics, and computer vision. | Direct official page retrieved on 2026-10-08 displayed the role and application form; tool reported a crawl today. The posting emphasizes software operating on actual robots and a history of shipping products. Python simulation tooling alone does not establish those qualifications. |
| [Saronic — Software Engineer, Generalist](https://jobs.ashbyhq.com/saronic/7054836e-c21d-4a59-985a-6da638218c1a) | Perception/navigation algorithms, performance and reliability, systematic validation, and familiarity with sensor fusion and SLAM. | Official posting content was available through the search index on 2026-10-08, with a crawl reported today. Ashby requires JavaScript for direct rendering; current application availability was not independently verified. Treat this as a requirements reference until the live application is checked. |

An earlier [Apptronik Senior Perception Learning Engineer URL](https://job-boards.greenhouse.io/apptronik/jobs/5678268004) still had indexed content, but direct retrieval redirected to the generic careers page. Its open status is **unverified**; cached search text is insufficient to recommend it as a current opening.

## Translate implementation into hiring evidence

The connection between the code and the requirements above is an assessment of transferable skills, not an employer endorsement.

| Campaign capability | Evidence a reviewer can inspect | Valuable next measurement |
| --- | --- | --- |
| Stereo depth and GPU inference | SGBM and pinned official C-Fast-FoundationStereo ONNX adapters; original calibrated RGB inputs; model hashes; common-mask error paired with native coverage; first-prediction CUDA profiling checks; separate cold and warm timing. | Completed GPU execution proof, independent depth error and obstacle pixel recall, and warm p95 latency on a named device. A provider being listed is insufficient proof that inference ran on the GPU. |
| Calibrated sensor fusion | Optical-axis depth convention, rigid transforms, projection with a Z-buffer, five fusion arms, shared nested dropout masks, and real extrinsic perturbations. Unknown predictions remain visible in missed-obstacle and coverage metrics. | False-free-space counts and missed-obstacle rates for each arm under identical stress conditions; quantify a reduction only when the baseline count is nonzero. |
| Estimated-pose interfaces | Original image and per-point scan timestamps, ORB-SLAM3/FAST-LIO2 export adapters, metric-scale ATE/RPE scoring, and rejection of stale, lost-tracking, mismatched-clock, or unaligned pose/scan inputs before policy inference. | Native estimator trajectories with complete tracking records, then fresh closed-loop goal, collision, and fall counts. Adapter readiness is not a SLAM result. |
| 3D terrain mapping | Endpoint elevation, local plane slope, residual roughness, explicit unknown cells, and independent vertical-ray height and hazard labels. | Height RMSE, hazard recall with unknown-cell counts, then paired traversal outcomes for terrain the locomotion controller supports. Geometric hazard labels do not establish friction or traversability. |

The pilot retains 50 ms scan point timing without motion compensation. Its ideal analytic IMU, pinhole rendering, deterministic texture, and simulated intensity placeholder must remain visible in any technical discussion. A later calibrated physical replay is the strongest step toward hardware-facing perception roles.

## Resume wording supported by completed code

The [completed independent pilot report](SENSOR_PILOT_RESULTS_2026-10-08.md)
now supports this measured project bullet:

- Built and validated a calibrated sensor replay benchmark on **72 simulated stereo pairs and 73,695 timed 3D LiDAR returns**, comparing SGBM and pretrained stereo across **five fusion methods and nine sensor conditions**, with verified CUDA execution and independent geometry labels.

Use one or two bullets suited to the role. The additional bullets below describe
implemented tools and interfaces; they do not assert improved robot behavior.

- Implemented a calibrated stereo replay benchmark for SGBM and a pinned pretrained Fast-FoundationStereo model, recording model provenance, native coverage, paired depth error, and cold/warm inference timing.
- Developed five stereo–LiDAR fusion baselines with shared dropout masks and extrinsic calibration stress tests, preserving unknown regions and measuring false free space against independent simulated geometry.
- Built timestamp-preserving stereo/LiDAR–IMU replay adapters and an estimated-pose navigation interface that rejects stale or lost-tracking inputs; added metric-scale trajectory scoring and independent 3D terrain evaluation.

Describe the environment as **simulation** wherever a reader could infer physical validation. Stored profiling evidence confirms CUDA-executed benchmarking for this pilot. Its depth accuracy/coverage/latency tradeoff is descriptive evidence from one held-out scene; the heuristic fusion gate misses its advancement target. Claim deployed SLAM or improved navigation only after native estimator and closed-loop outcomes exist.

## Numerical bullet templates after verified runs

Replace every bracket with a value from the completed report; omit a template when its required experiment has not run. Planned frame counts, protocol thresholds, upstream paper scores, and desired percentage improvements are not results.

The depth, fusion and terrain templates can now use the measured pilot report
with its simulation and single-scene limits. The pose/navigation template
remains unfilled because native estimator and closed-loop experiments have
not executed. Do not present the proposed 20% fusion target as achieved or the
92.57 ms p95 as real-time processing at 15 Hz.

- **Depth:** “Evaluated SGBM and pretrained Fast-FoundationStereo on [N] simulated stereo pairs across [G] held-out scene groups; measured [A] m versus [B] m common-mask RMSE, [C]% versus [D]% native coverage, and [E]/[F] near-obstacle pixels recalled.” Add “warm p95 [L] ms on [GPU]” only with its timing scope.
- **Fusion:** “Compared five fusion methods under [conditions]; confidence gating changed false-free-space events from [a]/[n] to [b]/[n], with obstacle recall [r1]% versus [r2]% and unknown coverage [u1]% versus [u2]%.” Use matched denominators and state whether events are pixels, frames, or episodes.
- **Pose/navigation:** “Measured [ATE] m ATE and [lost]/[frames] lost-tracking frames using [estimator] on [origin] replay; completed [goals]/[episodes] navigation goals with [collisions] collisions and [falls] falls.” The two clauses require their respective experiments; neither can be filled from readiness export.
- **Terrain:** “Evaluated 3D LiDAR elevation maps against independent simulated geometry: [H] m height RMSE over [observed]/[labelled] cells and [detected]/[hazards] geometric hazards detected.” Add traversal outcomes only after executed control trials.

Publish the raw counts, scene split, software/model hashes, device, and report alongside any resume percentage so an interviewer can reproduce the claim. The current pilot has too few independent held-out scene groups to support a general superiority claim.
