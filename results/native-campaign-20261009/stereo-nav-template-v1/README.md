# Frozen stereo navigation source template

This template pins the original stereo navigation source, frozen DR-default-s0 actor, exact two-episode smoke and nine-episode development protocol before native ORB results exist. It is not an executed campaign or a measured navigation result.

Archive SHA-256: `d3442e2c65be3febf7851a8012f6e497800f015a86083f04a0ada2b126f17302`. All 848 source/actor entries were independently extracted and verified. The native runtime input is deliberately absent; its SHA placeholder is filled only by the immutable post-build promotion helper after the actual pinned ORB build passes. Source, actor, routes, thresholds and cohorts remain unchanged. The promoted campaign submits a fresh smoke and holds development on that smoke.

The single durable continuation is owned by the ORB build controller and also handles offline replay; no separate stereo controller is submitted. Original body-attached stereo images and native responses will be retained by the eventual jobs.
