"""scripts/waiter/waiter_clips.py and scripts/waiter/dsrl/clip_env.py (display only): the clips repeat the scored
gate episodes step for step, the scored-run match check, and that the recording env only adds rendering to
WaiterEnv. No simulator is started."""
import inspect
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts" / "bench"), str(REPO / "scripts" / "waiter")]

import stand_manip_gate  # noqa: E402
import turn_test  # noqa: E402
import waiter_clips  # noqa: E402


def _body(fn, stop_at_return, drop=()):
    """Source lines of fn's body, trailing comments and blank lines dropped, whitespace stripped."""
    out = []
    for line in inspect.getsource(fn).splitlines()[1:]:
        line = line.split("  #")[0].strip()
        for d in drop:
            line = line.replace(d, "")
        if not line:
            continue
        if stop_at_return and line.startswith("return"):
            break
        out.append(line)
    return out


def _in_order(ref, got):
    it = iter(got)
    return [line for line in ref if not any(line == g for g in it)]


def test_walk_run_repeats_run_command_step_for_step():
    ref = _body(turn_test.run_command, stop_at_return=True)
    missing = _in_order(ref, _body(waiter_clips.walk_run, stop_at_return=False))
    assert not missing, f"waiter_clips.walk_run no longer repeats turn_test.run_command: {missing}"
    assert len(ref) > 25


def test_stand_run_repeats_the_q4_episode_step_for_step():
    """Every line of stand_manip_gate.episode up to its first return (the fall) appears, in order, in stand_run."""
    ref = _body(stand_manip_gate.episode, stop_at_return=True)
    missing = _in_order(ref, _body(waiter_clips.stand_run, stop_at_return=False, drop=("SM.",)))
    assert not missing, f"waiter_clips.stand_run no longer repeats the Q4 episode: {missing}"
    assert len(ref) > 12 and waiter_clips.STAND_TRAJ in stand_manip_gate.TRAJECTORIES


def test_scored_match():
    q4 = {"episodes": [{"trajectory": "P4_pick_then_hook", "seed": 0, "fell_at_s": None, "disp_m": 0.0467,
                        "yaw_deg": 7.06, "pitch_deg": 3.13, "ok": False}]}
    same = {"fell_at_s": None, "disp_m": 0.0467, "yaw_deg": 7.06, "pitch_deg": 3.13}
    assert waiter_clips.scored_match("stand", q4, same, 0)["matches_scored_run"]
    assert not waiter_clips.scored_match("stand", q4, dict(same, disp_m=0.05), 0)["matches_scored_run"]
    assert not waiter_clips.scored_match("stand", q4, same, 1)["matches_scored_run"]        # no such episode
    v2x = {"walks": [{"seed": 10, "displacement_m": 0.322, "yaw_deg": -8.1}]}
    assert waiter_clips.scored_match("walk", v2x, {"displacement_m": 0.322, "yaw_deg": -8.1}, 10)["matches_scored_run"]
    assert not waiter_clips.scored_match("walk", v2x, {"displacement_m": 0.33, "yaw_deg": -8.1}, 10)["matches_scored_run"]


def test_recording_env_only_adds_rendering():
    src = (REPO / "scripts/waiter/dsrl/clip_env.py").read_text()
    assert "class RecordingWaiterEnv(WaiterEnv):" in src
    assert "obs, r, done, info = super().step(a)" in src and "return obs, r, done, info" in src
    assert "obs = super().reset(seed, instruction_id=instruction_id)" in src
    # its own renderers, never the env's head-camera renderer or its render_enabled switch (ChunkRunner owns that)
    assert "self._renderer" not in src and "render_enabled" not in src.split('"""', 2)[2]
    for line in src.splitlines():
        assert not line.strip().startswith(("self.d.", "self.model.")) or "=" not in line, line
