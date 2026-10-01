# Mission 7 replay gate — interim findings (scratch, not a deliverable)

Gate = exact ten-fall replay (scripts/mission7_plate_stage.py::run/_episode): the 10 fallen episodes of
results/mission7-replay-smoke-20260921/fullroute/legacy-doors.json (validation Doors layouts 0,1,4,5,7,9,12,13,14,15),
bitwise physics on cn-c22, recorded body-frame command stream except where PlateStage.command overrides,
doors open at RECORDED activation times, window = original fall time, pass = tilt<0.78 in all 10.

Best variants 9/10:
- v2-align (21405537) L7: door +y, entry yaw err -1.02 rad (approach only 1.24 s), cross cmd [0.16..0.28, 0.25..0.11, 0.35],
  stalled at along -0.29 m (near edge -0.24; right foot on plate 50-87 N) ~1 s, tilt 0.09->0.31->0.88, fall 18.28 s (2.4 s after kick).
- v4-settle080 (21405541) L0: door +y, yaw err -1.49 (sideways), cmd [0.02,0.30,0] pure lateral, leading left foot on plate edge
  60-340 N at along -0.25..-0.30, uncommanded yaw -0.65..-0.99 rad/s, tilt 0.14->0.80 in 0.65 s, fall 26.60 s (1.92 s after kick).

Pooled 5 retained variants, 35 crossings: in-cross falls 7/21 sideways (|yaw err|>0.8) vs 0/14 forward; clears 2/21 vs 7/14.
All 7 in-cross falls 1.88-2.44 s after the kick. L7 falls in 5/6 cross-clear variants (incl. V2 21401689).
Post-stage (open-loop tail) falls: 5 of 12 falls; align variants hand back with yaw +1.5 rad vs recording
-> recorded body-frame commands applied ~82-90 deg off in world for 12-27 s.

Gate issues: L9/12/13 end 2.2-3.0 s after capture (never cross); guarded PASS reference never crossed the plate
(5 crossings end at along -0.14..-0.33; L5/L15 stuck in approach ~27 s); "crossings completed" counts 4 s timeouts.
Route gate criterion drift: code gate doors_16 = episodes>=16 (scripts/mission7_plate_safe.py:158), later read as 16/16 successes.
