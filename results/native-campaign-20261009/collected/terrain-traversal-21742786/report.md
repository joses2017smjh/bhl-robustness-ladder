# Actual sensor-informed terrain traversal

Scientific status: **NEGATIVE**.

Stopping short is a timeout even without falls. Qualification and confirmation are reported separately; shared actors/groups are paired clusters. All sensors and physics are simulated.

|Stage|Actor|Terrain|Group|Arm|Goal|Fall|Contact|Progress(m)|Height RMSE(m)|
|---|---|---|---:|---|---:|---:|---:|---:|---:|
|screen|dr-default-s0|flat|250000|baseline|1|0|0|5.017|0.0025|
|screen|dr-default-s0|flat|250001|baseline|1|0|0|5.024|0.0864|
|screen|dr-default-s0|small_steps|250000|baseline|1|0|0|4.995|0.0793|
|screen|dr-default-s0|small_steps|250001|baseline|0|0|1|4.671|0.0970|
|screen|dr-default-s0|ramp|250000|baseline|0|0|1|2.681|0.0811|
|screen|dr-default-s0|ramp|250001|baseline|0|0|1|3.947|0.0880|
|screen|dr-default-s1|flat|250000|baseline|0|0|1|2.295|0.1022|
|screen|dr-default-s1|flat|250001|baseline|0|0|1|2.574|0.0934|
|screen|dr-default-s1|small_steps|250000|baseline|0|0|1|2.391|0.0972|
|screen|dr-default-s1|small_steps|250001|baseline|0|0|1|3.409|0.0892|
|screen|dr-default-s1|ramp|250000|baseline|0|0|1|3.285|0.0943|
|screen|dr-default-s1|ramp|250001|baseline|0|0|1|2.861|0.0831|
|screen|dr-default-s2|flat|250000|baseline|0|0|1|3.559|0.0895|
|screen|dr-default-s2|flat|250001|baseline|0|0|1|3.268|0.0952|
|screen|dr-default-s2|small_steps|250000|baseline|0|0|1|4.490|0.0849|
|screen|dr-default-s2|small_steps|250001|baseline|0|0|1|3.308|0.0827|
|screen|dr-default-s2|ramp|250000|baseline|0|0|1|3.592|0.0810|
|screen|dr-default-s2|ramp|250001|baseline|0|0|1|3.402|0.0859|

Height error compares raw inferred maps with independent dense scene geometry after commands were computed. Wall-top/side aliasing is included; these values are not ground-only accuracy. Smoke never qualifies actors.
