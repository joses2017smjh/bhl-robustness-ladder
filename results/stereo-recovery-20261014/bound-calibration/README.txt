Calibration of the predeclared ResumeBound (stereo_relocalization.ResumeBound), run before the
stereo recovery campaign on prior data only: processing-schedule v1 bounded_latest episodes
(seeds 750000/750001; collector-inputs under processing-schedule-v1-20261012).
  calib.py <v1 campaign dir> <out.npy>   truth displacement vs dead-reckoned applied commands
  joint_bound.py <out.npy>               joint exceedance of 0.15 m + 0.6 L / 10 deg + 0.4 Theta
  body_height_tilt.py <v1 campaign dir>  body height and tilt change over windows
  timeline.py <dir of extracted bounded stereo native logs>   v1 loss archetype timelines
Outputs: *.out (captured 2026-10-14). Evaluator truth is used only to set the bound, never online.
