"""Mission 7 gate predicates shared by the runners and the replay-gate follow-up.

Two numbers the docs cite were coded more loosely than the docs read them
(verified 2026-10-01, results/solutions-20260930/verify-mission7/):

* Route gate.  docs/MISSION7_TASKS.md gates Doors and Transport at >= 16/16
  SUCCESSES on validation layouts 0-15.  mission7_plate_safe.py coded
  ``doors_16`` / ``transport_16`` as ``episodes >= 16``, so 21398514 read true
  beside 1/16 and 0/16 successes, and the route-handoff probe that a replay
  10/10 releases encoded no criterion at all.
* Replay-gate crossings.  The follow-up's ``crossings_completed`` counted every
  episode with a ``recorded`` entry after ``cross`` in its stage history.
  PlateStage appends that entry whether the crossing cleared the plate, timed
  out at 4.04 s or fell, so 21405537 and 21405541 read 7/7 against 4/7 and 1/7
  real clears.

The gate and crossing predicates are pure.  The trace helpers at the bottom
stream the raw episodes.json(.gz) line by line: the traces are 0.3-1 GB
uncompressed and do not fit a small job when json.load'ed, and only time, base
xy and tilt are needed from each sample.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

from bhl_robust.mission.layout import generate


# Route gate: the documented protocol is validation layouts 0-15, every one a
# success.  A smoke or partial run cannot reach 16 successes.
ROUTE_GATE_LAYOUTS = tuple(range(16))
ROUTE_GATE_SUCCESSES = 16
ROUTE_GATE_CRITERION = (
    "per stage: >= 16 successes over validation layouts 0-15, i.e. 16/16 (docs/MISSION7_TASKS.md "
    "gate table); a run over other or fewer layouts reports NOT_GATE_PROTOCOL and never passes; "
    "the route gate is doors_16 AND transport_16, each from its own run when the stages run separately")

# A real clear: the base gets this far past the plate centre along the door
# direction before the stage hands back.  0.35 m is the --cross-clear of the
# cross-clear gates (the plate disc is 0.24 m in radius); it is applied to every
# variant, including the fixed-1.2 s crossings that have no --cross-clear, so
# the gates stay comparable.
CLEAR_ALONG_M = .35
FALL_TILT = .78          # the Mission 7 fall predicate, unchanged
TIME_EPS_S = 1e-9        # history and sample times are the same runner.d.time floats
REPLAY_SPLIT = "validation"   # PlateStage replays reset DebugEnv(split="validation") to the source index
CLEAR_DEFINITION = (
    "real clears: staged crossings whose base reached along >= +0.35 m past the staged plate's centre "
    "(along the door direction) before the stage's hand-back, with no fall (tilt >= 0.78) at or "
    "before hand-back; correct- and wrong-side plates alike (correct_side is reported per crossing); "
    "counted per crossing from the raw trace by scripts/mission7_gates.py; "
    "null when the raw trace is missing, unreadable, partial (complete: false) or not the trace of "
    "this gate's result.json")


def route_stage_passed(summary):
    """Route-gate criterion for one stage: at least 16 successes.

    ``summary`` carries ``successes`` (mission7_plate_safe.summarize()).  Over
    the documented 16-layout protocol that is 16/16.
    """
    return int(summary["successes"]) >= ROUTE_GATE_SUCCESSES


def route_gate_block(rows, stages):
    """result.json gate block for a route-handoff probe run (pure).

    ``rows`` are the probe's per-episode summaries (``stage``, ``layout_index``,
    ``success``).  A stage passes only when its episodes are exactly validation
    layouts 0-15 and route_stage_passed() holds; a run over other or fewer
    layouts reads NOT_GATE_PROTOCOL, a stage this run did not cover NOT_RUN.
    ``doors_16`` / ``transport_16`` are true only on PASS.
    """
    block = {"criterion": ROUTE_GATE_CRITERION, "required_successes": ROUTE_GATE_SUCCESSES,
             "protocol_layouts": list(ROUTE_GATE_LAYOUTS), "stages": {}}
    for stage in ("doors", "transport"):
        if stage not in stages:
            entry = {"verdict": "NOT_RUN", "passed": False}
        else:
            stage_rows = [row for row in rows if row.get("stage") == stage]
            layouts = sorted(int(row["layout_index"]) for row in stage_rows)
            summary = {"episodes": len(stage_rows),
                       "successes": sum(bool(row["success"]) for row in stage_rows)}
            protocol = layouts == list(ROUTE_GATE_LAYOUTS)
            passed = protocol and route_stage_passed(summary)
            entry = {**summary, "layout_indices": layouts, "protocol_complete": protocol,
                     "verdict": "PASS" if passed else ("FAIL" if protocol else "NOT_GATE_PROTOCOL"),
                     "passed": passed}
        block["stages"][stage] = entry
        block[f"{stage}_16"] = entry["passed"]
    return block


def crossing_outcomes(row, layout=None):
    """Per-crossing outcomes of one replay-gate episode row (pure).

    ``row`` is one entry of a replay gate's episodes.json -- ``layout_index``,
    ``stage_history`` and ``samples`` (``time_s``, ``xy``, ``tilt``; other keys
    are ignored) -- as json.load returns it or as read_replay_episodes() streams
    it.  For every ``cross`` entry in the stage history:

    * start_s is that entry's time and handback_s the first later ``recorded``
      entry for the same door (None when the trace ends first).
    * along is the base xy minus the plate centre, layout.plate(door, side),
      projected on the door direction: the quantity PlateStage tests against
      --cross-clear.  The side is the latest earlier history entry for that
      door that carries one.
    * The window is the samples with start_s <= time_s <= handback_s (to the
      end of the trace without a hand-back).  along_start_m / along_end_m are
      its first and last samples (the last one after a fall too), along_max_m
      the maximum up to the first fall.  A window with no sample in it, or a
      row with no samples at all, raises ValueError: PlateStage records every
      replay step, so neither is a crossing that can be scored.
    * cleared: along >= 0.35 m at some window sample at or before the first
      fall.  fell_in_crossing: the episode's first fall (tilt >= 0.78) lies in
      the window.  fell_before_handback: that fall is at or before handback_s
      (any fall when there is no hand-back).
    * real_clear: cleared, handed back, and not fell_before_handback.

    These reproduce results/solutions-20260930/verify-mission7/verify_crossings.py
    operation for operation.  The row's own first_fall_s, replay_elapsed_s,
    maximum_tilt and layout_seed are checked when present, so a mis-parsed
    trace or a different layout raises ValueError instead of counting.
    """
    index = int(row["layout_index"])
    if layout is None:
        layout = generate(REPLAY_SPLIT, index)
    if "layout_seed" in row and int(row["layout_seed"]) != layout.seed:
        raise ValueError(f"layout {index}: row seed {row['layout_seed']} != geometry seed {layout.seed}")
    samples = row["samples"]
    if not samples:   # PlateStage writes at least one sample per row (replay_elapsed_s is samples[-1])
        raise ValueError(f"layout {index}: no samples, so neither the header nor a crossing can be checked")
    t = np.array([s["time_s"] for s in samples], dtype=float)
    xy = np.array([s["xy"] for s in samples], dtype=float).reshape(-1, 2)
    tilt = np.array([s["tilt"] for s in samples], dtype=float)
    hit = np.flatnonzero(tilt >= FALL_TILT)
    fall_s = float(t[hit[0]]) if hit.size else None
    if "first_fall_s" in row and row["first_fall_s"] != fall_s:
        raise ValueError(f"layout {index}: first fall {fall_s} != recorded first_fall_s {row['first_fall_s']}")
    if "replay_elapsed_s" in row and row["replay_elapsed_s"] != t[-1]:
        raise ValueError(f"layout {index}: last sample {t[-1]} != replay_elapsed_s {row['replay_elapsed_s']}")
    if "maximum_tilt" in row and row["maximum_tilt"] != tilt.max():
        raise ValueError(f"layout {index}: max tilt {tilt.max()} != maximum_tilt {row['maximum_tilt']}")
    history = row.get("stage_history") or []
    outcomes = []
    for i, entry in enumerate(history):
        if entry.get("phase") != "cross":
            continue
        door = int(entry["door"])
        t0 = float(entry["time_s"])
        t1 = next((float(g["time_s"]) for g in history[i + 1:]
                   if g.get("phase") == "recorded" and g.get("door") == door), None)
        side = next((int(g["side"]) for g in reversed(history[:i])
                     if g.get("door") == door and "side" in g), None)
        if side is None:
            raise ValueError(f"layout {index}: crossing at {t0} s has no staged side for door {door}")
        _, direction = layout.door(door)
        direction = np.asarray(direction, dtype=float)
        plate = np.asarray(layout.plate(door, side), dtype=float)
        window = t >= t0 - TIME_EPS_S
        if t1 is not None:
            window &= t <= t1 + TIME_EPS_S
        if not window.any():
            raise ValueError(f"layout {index}: no samples between the crossing at {t0} s and its hand-back ({t1} s)")
        fell_in = (fall_s is not None and fall_s >= t0 - TIME_EPS_S
                   and (t1 is None or fall_s <= t1 + TIME_EPS_S))
        # never empty: the fall is itself a window sample when fell_in
        upright = window & (t <= fall_s + TIME_EPS_S) if fell_in else window
        along = (xy[upright] - plate) @ direction
        along_all = (xy[window] - plate) @ direction
        cleared = bool(np.any(along >= CLEAR_ALONG_M))
        fell_before = fall_s is not None and (t1 is None or fall_s <= t1 + TIME_EPS_S)
        outcomes.append({
            "door": door, "side": side, "correct_side": side == layout.correct_sides[door],
            "start_s": t0, "handback_s": t1, "handed_back": t1 is not None,
            "along_start_m": float(along[0]), "along_max_m": float(along.max()),
            "along_end_m": float(along_all[-1]),
            "cleared": cleared, "first_fall_s": fall_s,
            "fell_in_crossing": fell_in, "fell_before_handback": fell_before,
            "real_clear": cleared and t1 is not None and not fell_before,
        })
    return outcomes


# mission7_overnight.write() serializes the trace with json.dumps(indent=2):
# the top-level keys sit at two spaces ("complete" first, then the "episodes"
# list), episodes open at four spaces and their keys sit at six, each sample
# opens at eight and its keys sit at ten, and the per-sample contact events and
# lists nest at twelve and deeper.  Exact indentation separates the three
# sample keys that are parsed from everything else.
_DEEP = " " * 11
_SAMPLE_KEY = " " * 10 + '"'
_TIME = " " * 10 + '"time_s": '
_XY = " " * 10 + '"xy": ['
_TILT = " " * 10 + '"tilt": '
_SAMPLE_OPEN = " " * 8 + "{"
_SAMPLE_CLOSE = " " * 8 + "}"
_SAMPLES_OPEN = " " * 6 + '"samples": ['
_SAMPLES_CLOSE = " " * 6 + "]"
_EPISODE_OPEN = " " * 4 + "{"
_EPISODE_CLOSE = " " * 4 + "}"
_COMPLETE = '  "complete": '
_EPISODES_OPEN = '  "episodes": ['
_EPISODES_CLOSE = "  ]"
_ENDS = {"top": "without an episode list in the indent=2 layout mission7_overnight.write() produces",
         "list": "inside its episode list", "after": "before its closing brace"}


def _number(text):
    return float(text.strip().rstrip(","))


def read_replay_episodes(path, require_complete=True):
    """Stream a replay gate's episodes.json(.gz) one lean episode row at a time.

    Each row holds the episode's keys other than ``samples`` (layout_index,
    stage_history, first_fall_s, ...) as written, and ``samples`` reduced to
    ``time_s`` / ``xy`` / ``tilt``.  The whole file must have the layout
    mission7_overnight.write() gives it, from the opening to the closing brace,
    or ValueError: an unindented, otherwise indented or truncated file, one cut
    between two episodes included, never reads as fewer episodes.  With
    ``require_complete`` (the default) the top-level ``complete`` flag must be
    true; PlateStage rewrites the trace with ``complete: false`` after every
    episode while the gate runs, and that is rejected on its second line,
    before any episode is read.  crossing_outcomes() then cross-checks the
    parsed samples against the episode's own header.
    """
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    complete = None
    with opener(path, "rt") as stream:
        state = "top"
        head, samples, sample = [], [], None
        for line in stream:
            if state == "samples":
                if line.startswith(_DEEP):
                    continue
                if line.startswith(_SAMPLE_KEY):
                    if sample is None:
                        raise ValueError(f"{path}: sample key outside a sample: {line.strip()[:60]}")
                    if line.startswith(_TIME):
                        sample["time_s"] = _number(line[len(_TIME):])
                    elif line.startswith(_XY):
                        # next(stream, "") so a trace cut inside the pair raises ValueError, not RuntimeError
                        sample["xy"] = [_number(next(stream, "")), _number(next(stream, ""))]
                        if next(stream, "").strip() not in ("]", "],"):
                            raise ValueError(f"{path}: xy is not a pair at sample {len(samples)}")
                    elif line.startswith(_TILT):
                        sample["tilt"] = _number(line[len(_TILT):])
                elif line.startswith(_SAMPLE_OPEN):
                    if sample is not None:
                        raise ValueError(f"{path}: sample {len(samples)} never closes")
                    sample = {}
                elif line.startswith(_SAMPLE_CLOSE):
                    if sample is None or len(sample) != 3:
                        raise ValueError(f"{path}: sample {len(samples)} lacks time_s/xy/tilt")
                    samples.append(sample)
                    sample = None
                elif line.startswith(_SAMPLES_CLOSE):
                    state = "tail"
                continue
            if state == "head":
                if line.startswith(_SAMPLES_OPEN):
                    state = "tail" if line.rstrip().endswith(("[]", "[],")) else "samples"
                elif line.startswith(_EPISODE_CLOSE):
                    raise ValueError(f"{path}: episode without samples")
                else:
                    head.append(line)
                continue
            if state == "tail":
                if line.startswith(_EPISODE_CLOSE):
                    text = "".join(head).rstrip().rstrip(",")
                    row = json.loads("{" + text + "}")
                    row["samples"] = samples
                    yield row
                    state = "list"
                else:
                    head.append(line)   # keys written after "samples", if any
                continue
            if state == "list":   # between two episodes: the next one opens or the list closes
                if line.rstrip() == _EPISODE_OPEN:
                    state, head, samples, sample = "head", [], [], None
                elif line.rstrip().rstrip(",") == _EPISODES_CLOSE:
                    state = "after"
                else:
                    raise ValueError(f"{path}: unexpected line between episodes: {line.strip()[:60]}")
                continue
            # top level: "top" before the episode list, "after" behind it, "end" past the closing brace
            if state == "end":
                if line.strip():
                    raise ValueError(f"{path}: content after the closing brace: {line.strip()[:60]}")
            elif line.startswith(_COMPLETE):
                complete = json.loads(line[len(_COMPLETE):].strip().rstrip(","))
                if require_complete and complete is not True:
                    raise ValueError(f'{path}: "complete": {json.dumps(complete)}, a partial trace (PlateStage '
                                     "marks it complete only after its last episode)")
            elif state == "top" and line.rstrip() == _EPISODES_OPEN:
                state = "list"
            elif state == "top" and line.rstrip().rstrip(",") == _EPISODES_OPEN + "]":
                state = "after"   # an empty episode list
            elif state == "after" and line.rstrip() == "}":
                state = "end"
        if state != "end":
            raise ValueError(f"{path}: trace ends {_ENDS.get(state, f'inside an episode ({state})')}")
        if require_complete and complete is None:
            raise ValueError(f'{path}: no top-level "complete" flag')


def replay_trace_path(gate_dir):
    """A replay gate's raw trace: episodes.json (before the follow-up gzips it)
    or episodes.json.gz; None when neither exists."""
    gate_dir = Path(gate_dir)
    for name in ("episodes.json", "episodes.json.gz"):
        if (gate_dir / name).is_file():
            return gate_dir / name
    return None


def replay_gate_crossings(path, episode_comparison=None):
    """Real clears over one replay gate's complete raw trace, with per-crossing outcomes.

    ``episode_comparison`` is the gate's result.json rows.  When given, the
    traced episodes, samples aside, must equal them one for one and in order
    (PlateStage writes both files from the same rows), so a trace from another
    run, or one that does not cover the result's episodes, raises ValueError
    instead of counting.
    """
    layouts = []
    for n, row in enumerate(read_replay_episodes(path)):
        if episode_comparison is not None:
            if n >= len(episode_comparison):
                raise ValueError(f"{path}: more traced episodes than result.json's {len(episode_comparison)}")
            expected = episode_comparison[n]
            if {key: value for key, value in row.items() if key != "samples"} != expected:
                raise ValueError(f"{path}: traced episode {n} (layout {row.get('layout_index')}) is not result.json's "
                                 f"episode_comparison[{n}] (layout {expected.get('layout_index')})")
        layouts.append({"layout_index": row["layout_index"], "crossings": crossing_outcomes(row)})
    if episode_comparison is not None and len(layouts) != len(episode_comparison):
        raise ValueError(f"{path}: {len(layouts)} traced episodes, result.json has {len(episode_comparison)}")
    crossings = [c for layout in layouts for c in layout["crossings"]]
    return {"trace": str(path), "episodes": len(layouts), "crossings": len(crossings),
            "real_clears": sum(c["real_clear"] for c in crossings),
            "definition": CLEAR_DEFINITION, "layouts": layouts}
