#!/usr/bin/env python3
"""Mission 7 learned crossing (workstream m7-platecross, 2026-10-02): selection, checks and scene probe.

Frozen design: SLURM_JOBS.md, 'User approval recorded 2026-10-02 14:15', item (B). Launcher:
slurm/repo20260923/gpu_platecross.sbatch (its header is the predeclaration of record).

    select        the SELECTION step: reads every seed's JSONs, writes <RES>/selection.json ONCE (O_EXCL) when
                  all three seeds are complete; prints one 'PLATECROSS SELECTION: ...' line. INCOMPLETE is
                  printed, never written. An existing selection.json is re-printed, never overwritten.
    check-env     a run's params/env.yaml against the parent's (arms-turngait-clock-s2): identical except the
                  training seed and the declared terrain fields, which must hold the declared numbers.
    write-record  the per-seed training record, written once from the fresh training log's evidence.
    check-record  the problems that keep a stored training record from letting a seed's gates run.
    check-export  the exported policy.onnx against the final checkpoint's actor (max |difference| on random inputs).
    smoke-standins  stand-in seed JSONs for the smoke's selection plumbing (in a -smoke directory only;
                  --joint-check makes stand-in seed 1 a SYNTHETIC v2 FAIL + QUALIFIED 0/60 seed).
    probe-scene   (Isaac, inside the container) builds the task and measures the plates in the imported terrain
                  mesh: count, shape, radius / side, height, positions, the static collider, the spawn
                  clearance, the 77-wide actor observation, feet standing 0.03 m higher on plates than on the floor
                  and no link below the floor; one 'PLATECROSS PROBE: PASS|FAIL' line + JSON.

Everything but probe-scene is Isaac-free (the unit tests import this file by path); the terrain module is
loaded by path too, so importing this file never imports bhl_robust.tasks (which needs Isaac's app).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
import traceback
from fractions import Fraction
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TERRAIN_PATH = REPO / "src/bhl_robust/tasks/platecross_terrain.py"

# The variant: v1 = PlateCross (2026-10-02, every tile plated; the default, unchanged); v2 = F1 "PlateCross v2"
# (2026-10-04, half of the tiles flat: the plate tile in columns 0-4 of the curriculum layout, a plane in 5-9).
# Chosen by the environment variable PLATECROSS_VARIANT (the v2 launcher exports it); probe-scene takes it from its
# --task. Every v1 path is unchanged.
VARIANTS = {"v1": {"task": "Velocity-BHL-Arms-PlateCross-v0", "prefix": "arms-platecross-clocks2",
                   "trained_on": "plates"},
            "v2": {"task": "Velocity-BHL-Arms-PlateCross2-v0", "prefix": "arms-platecross2-clocks2",
                   "trained_on": "half plates, half flat ground (F1)"}}
VARIANT = os.environ.get("PLATECROSS_VARIANT", "v1")
if VARIANT not in VARIANTS:
    raise SystemExit(f"PLATECROSS_VARIANT={VARIANT!r} is not one of {sorted(VARIANTS)}")
TASK_ID = VARIANTS[VARIANT]["task"]
PREFIX = VARIANTS[VARIANT]["prefix"]
PARENT_RUN = "arms-turngait-clock-s2"
PARENT_TASK = "Velocity-BHL-Arms-TurnGaitClock-v0"
PARENT_ITER = 5999
FT_ITERS = 3000
SEEDS = (0, 1, 2)
N_PUSH = 60
PLATE_STEPS = 15                 # probe-scene: zero-action steps (0.6 s) after placing robots on plates / the floor
OBS_POLICY, OBS_CRITIC = 77, 80

# The PREDECLARED RULE, frozen 2026-10-02 (the task text, verbatim; also in the launcher header).
FROZEN_RULE = ("the three fine-tuned final checkpoints go through the unchanged turn qualification (turn_test v2 + "
               "cpu_turn_qualify); the qualified seed with the lowest push-fall rate (tie: lowest seed index) is the "
               "SINGLE stage gait run on bench v2 under bench v2's rule (N = 41; >= 39/41 clears, 0 falls, >= "
               "11/10/9/7 per heading) with the generic stage-gait override; no qualified seed -> NEGATIVE (no bench).")
FROZEN_LABELS = (f"Labels: LEARNED gait (fine-tuned from clock-s2 on {VARIANTS[VARIANT]['trained_on']}); MuJoCo "
                 "gates; bench v2's 180-deg timing caveat applies.")
# How the launcher reads "qualified" (stated before any run): the frozen rule's "unchanged turn qualification
# (turn_test v2 + cpu_turn_qualify)" counted per seed as the R1 / v5 joint rule counts it
# (scripts/bench/turngait_r12_verdict.py seed_counts; SLURM_JOBS.md, the R1/R2 predeclaration).
READING = ("A seed is qualified iff it both PASSES turn_test v2 (its <run>.json verdict, under the unchanged v2 "
           "protocol) AND is QUALIFIED by cpu_turn_qualify's unchanged rule (its <run>__qualify.json verdict), as the "
           "R1 / v5 joint rule counts a seed (turngait_r12_verdict.seed_counts); a seed that fails v2 does not count "
           "even if cpu_turn_qualify says QUALIFIED. The push-fall rate is falls / n of the qualify JSON's push clause "
           "(n = 60). Every seed needs its training record, v2 JSON (with the unchanged v2 protocol's rule block) and "
           "qualify JSON, all for the same exported deploy.yaml; otherwise the selection is INCOMPLETE and nothing is "
           "written.")
NEGATIVE_LINE = "NEGATIVE: no qualified seed, no bench"
BENCH_NOTE = ("The bench v2 run of the selected gait is the COORDINATOR's later submission (it needs workstream A's "
              "generic stage-gait override); this launcher never builds or runs it.")
# Disclosed before any run (also in the launcher header).
SWING_DISCLOSURE = ("R1's feet_swing_height reward (unchanged) measures foot height in absolute world z "
                    "(gait_clock_mdp.feet_swing_height: flat plane, ground z = 0), so over a plate its swing-clearance "
                    "target is 0.03 m lower relative to the plate top; feet in contact are excluded, and it is the only "
                    "absolute-z term among R1's 19 reward and 2 termination terms")
DISCLOSED = ("clock-s2's straight-walk drift (R1 walk drift 19.8 / 10.8 / 2.4 deg on qualify walk seeds 10-12) is "
             "inherited and is the baseline; " + SWING_DISCLOSURE + "; bench v2's 180-deg timing caveat applies")
# turn_test.py --protocol v2 as the launcher runs it (its defaults): scripts/bench/turngait_r12_verdict.py V2_RULE,
# copied (a test checks the two are equal). A v2 JSON with any other rule block is not the unchanged v2.
V2_RULE = {"turn_min_deg": 150.0, "drift_max_deg": 15.0, "seconds": 6.0, "turn_seeds": [0, 1, 2], "turn_wz": 0.6,
           "turn_warm_s": 3.0, "walk_cmd": [0.35, 0.0, 0.0], "walk_warm_s": 1.0, "walk_seed": 0}
V2_VERDICTS = ("PASS", "FAIL")
QUALIFY_VERDICTS = ("QUALIFIED", "NOT QUALIFIED")


def _load_by_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def terrain():
    """platecross_terrain, loaded by path (never through bhl_robust.tasks, which imports Isaac)."""
    return sys.modules.get("platecross_terrain_by_path") or _load_by_path("platecross_terrain_by_path", TERRAIN_PATH)


def final_ckpt(iters: int = FT_ITERS) -> str:
    return f"model_{PARENT_ITER + iters - 1}.pt"


def run_name(seed: int, prefix: str = PREFIX, suffix: str = "") -> str:
    return f"{prefix}-s{seed}{suffix}"


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_once(path: Path, obj: dict, tag: str) -> bool:
    """Write JSON to `path` only if it does not exist (hard link from a temp file: atomic, never overwrites)."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{tag}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    try:
        os.link(tmp, path)
        return True
    except FileExistsError:
        return False
    finally:
        tmp.unlink()


# =============================================================================== the selection rule (pure)

def seed_qualified(v2_verdict, qualify_verdict) -> bool:
    """The unchanged turn qualification of one seed (turngait_r12_verdict.seed_counts without its training clause,
    which a complete row already carries): turn_test v2 PASS AND cpu_turn_qualify QUALIFIED. A seed that fails v2
    does not count even if cpu_turn_qualify says QUALIFIED."""
    return v2_verdict == "PASS" and qualify_verdict == "QUALIFIED"


def _row_problem(r) -> str:
    """Why a row cannot be judged ('' = it can): missing, incomplete, or without a valid v2 / qualify verdict."""
    if not isinstance(r, dict):
        return "no row"
    if not r.get("complete"):
        return r.get("why") or "incomplete"
    if r.get("v2_verdict") not in V2_VERDICTS or r.get("qualify_verdict") not in QUALIFY_VERDICTS:
        return f"no valid v2 / qualify verdict ({r.get('v2_verdict')!r} / {r.get('qualify_verdict')!r})"
    return ""


def decide(rows: dict) -> dict:
    """The frozen rule on per-seed rows {seed: {"complete": bool, "why": str, "v2_verdict": PASS | FAIL,
    "qualify_verdict": QUALIFIED | NOT QUALIFIED, "falls": int, "n": int}}.
    -> {"verdict": SELECTED | NEGATIVE | INCOMPLETE, "seed": int | None, "reason": str}.
    INCOMPLETE if any declared seed is missing, incomplete or without a valid v2 / qualify verdict; a seed is
    qualified iff v2 PASS AND QUALIFIED (seed_qualified); NEGATIVE if none is; else the qualified seed with the
    lowest falls / n, ties to the lowest seed index."""
    missing = [s for s in SEEDS if _row_problem(rows.get(s))]
    if missing:
        why = "; ".join(f"s{s}: {_row_problem(rows.get(s))}" for s in missing)
        return {"verdict": "INCOMPLETE", "seed": None, "reason": f"seed(s) {missing} incomplete ({why})"}
    extra = sorted(set(rows) - set(SEEDS))
    if extra:
        return {"verdict": "INCOMPLETE", "seed": None, "reason": f"undeclared seed(s) {extra}"}
    qualified = [s for s in SEEDS if seed_qualified(rows[s]["v2_verdict"], rows[s]["qualify_verdict"])]
    if not qualified:
        return {"verdict": "NEGATIVE", "seed": None, "reason": NEGATIVE_LINE}
    rate = {s: Fraction(int(rows[s]["falls"]), int(rows[s]["n"])) for s in qualified}
    best = min(qualified, key=lambda s: (rate[s], s))
    ties = [s for s in qualified if rate[s] == rate[best]]
    out = [s for s in SEEDS if s not in qualified]
    return {"verdict": "SELECTED", "seed": best,
            "reason": (f"qualified seeds (v2 PASS AND QUALIFIED) {qualified}; push-fall rates "
                       + ", ".join(f"s{s} {rows[s]['falls']}/{rows[s]['n']}" for s in qualified)
                       + f"; lowest s{best}" + (f" (tie with {[t for t in ties if t != best]}: lowest seed index)"
                                                 if len(ties) > 1 else "")
                       + ("; not qualified: " + ", ".join(f"s{s} (v2 {rows[s]['v2_verdict']}, "
                                                           f"{rows[s]['qualify_verdict']})" for s in out)
                          if out else ""))}


# =============================================================================== per-seed inputs

RECORD_FLAGS = ("final_ckpt_present", "task_id_in_log", "parent_loaded_in_log", "plates_marker_in_log",
                "feet_gait_in_log", "push_robot_in_log")


def record_problems(rec, run: str, final: str, run_dir: str | None = None) -> list[str]:
    """Reasons a stored training record does not let this seed's gates run (or count): [] = ok."""
    if not isinstance(rec, dict):
        return ["record-not-a-json-object"]
    why = []
    if rec.get("run") != run:
        why.append(f"record-run-not-{run}")
    if rec.get("task") != TASK_ID:
        why.append(f"record-task-not-{TASK_ID}")
    if rec.get("final_ckpt") != final:
        why.append(f"record-final_ckpt-not-{final}")
    if run_dir is not None and rec.get("run_dir") != run_dir:
        why.append("record-is-for-another-run-dir")
    for k in RECORD_FLAGS:
        if rec.get(k) is not True:
            why.append(f"record-{k}-not-true")
    g = rec.get("guard")
    if not isinstance(g, list):
        why.append("record-has-no-guard-list")
    elif g:
        why.append("record-guard:" + ",".join(str(x) for x in g))
    if not str(rec.get("env_check", "")).startswith("PLATECROSS ENV: OK"):
        why.append("record-env-check-not-ok")
    if not str(rec.get("recipe_check", "")).startswith("GAIT-CLOCK RECIPE: OK"):
        why.append("record-recipe-check-not-ok")
    return why


def _json(path: Path):
    with open(path) as f:
        d = json.load(f)
    if not isinstance(d, dict):
        raise ValueError("not a JSON object")
    return d


def read_seed(res: Path, seed: int, prefix: str = PREFIX, suffix: str = "", iters: int = FT_ITERS) -> dict:
    """One seed's row for `decide`, read from its three JSONs (never from exit codes)."""
    run = run_name(seed, prefix, suffix)
    final = final_ckpt(iters)
    row = {"seed": seed, "run": run, "complete": False, "why": "", "qualified": False, "v2_verdict": None,
           "qualify_verdict": None, "falls": None, "n": None}
    paths = {"training": res / "training" / f"{run}.json", "v2": res / "turn-test-v2" / f"{run}.json",
             "qualify": res / "qualify" / f"{run}__qualify.json"}
    row["inputs"] = {k: str(v) for k, v in paths.items()}
    docs = {}
    for k, p in paths.items():
        try:
            docs[k] = _json(p)
        except FileNotFoundError:
            row["why"] = f"no {k} JSON ({p.name})"
            return row
        except Exception as exc:  # noqa: BLE001
            row["why"] = f"unreadable {k} JSON ({type(exc).__name__})"
            return row
    rec, v2, q = docs["training"], docs["v2"], docs["qualify"]
    rp = record_problems(rec, run, final)
    if rp:
        row["why"] = "training record: " + " ".join(rp)
        return row
    run_dir = Path(rec["run_dir"])
    if not (run_dir / final).is_file():
        row["why"] = f"run dir has no {final}"
        return row
    deploy = str(run_dir / "exported" / "deploy.yaml")
    row.update(run_dir=str(run_dir), deploy=deploy)
    if v2.get("protocol") != "v2" or v2.get("verdict") not in V2_VERDICTS:
        row["why"] = "v2 JSON is not a turn_test v2 verdict"
        return row
    rule = v2.get("rule")
    off = (sorted(k for k, v in V2_RULE.items() if rule.get(k) != v) if isinstance(rule, dict)
           else ["no rule block"])
    if off:
        row["why"] = f"v2 JSON's rule is not the unchanged turn_test v2 protocol ({', '.join(off)})"
        return row
    if v2.get("deploy") != deploy:
        row["why"] = "v2 JSON is for another deploy.yaml"
        return row
    if q.get("run") != run:
        row["why"] = "qualify JSON is for another run"
        return row
    if q.get("deploy") != deploy:
        row["why"] = "qualify JSON is for another deploy.yaml"
        return row
    if q.get("verdict") not in QUALIFY_VERDICTS:
        row["why"] = f"qualify verdict {q.get('verdict')!r}"
        return row
    try:
        push = q["clauses"]["push"]
        falls, n = push["falls"], push["n"]
        if isinstance(falls, bool) or isinstance(n, bool) or int(falls) != falls or int(n) != n:
            raise ValueError("non-integer push counts")
        falls, n = int(falls), int(n)
    except Exception as exc:  # noqa: BLE001
        row["why"] = f"qualify JSON has no push falls / n ({type(exc).__name__})"
        return row
    if n != N_PUSH or not 0 <= falls <= n:
        row["why"] = f"push clause {falls}/{n} (need n = {N_PUSH})"
        return row
    row.update(complete=True, v2_verdict=v2["verdict"], qualify_verdict=q["verdict"],
               qualified=seed_qualified(v2["verdict"], q["verdict"]), falls=falls, n=n,
               qualify_detail=q.get("detail"))
    return row


def stage_gait(row: dict) -> dict:
    """What the coordinator's bench v2 submission needs about the selected export, with its provenance."""
    deploy = Path(row["deploy"])
    onnx = deploy.parent / "policy.onnx"
    text = deploy.read_text()
    return {"deploy_yaml": str(deploy), "policy_onnx": str(onnx),
            "sha256": {"deploy_yaml": sha256(deploy), "policy_onnx": sha256(onnx)},
            "num_observations_line": next((ln for ln in text.splitlines() if ln.startswith("num_observations:")), None),
            "gait_clock_block": any(ln.startswith("gait_clock:") for ln in text.splitlines())}


def selection(res: Path, prefix: str = PREFIX, suffix: str = "", iters: int = FT_ITERS) -> tuple[dict, dict]:
    rows = {s: read_seed(res, s, prefix, suffix, iters) for s in SEEDS}
    return decide(rows), rows


def selection_line(dec: dict, rows: dict) -> str:
    if dec["verdict"] == "SELECTED":
        r = rows[dec["seed"]]
        return (f"PLATECROSS SELECTION: SELECTED {r['run']} (seed {dec['seed']}; v2 {r['v2_verdict']}, "
                f"{r['qualify_verdict']}, push {r['falls']}/{r['n']}; {dec['reason']}) -> bench v2 stage gait "
                f"{r['deploy']} [LEARNED gait (fine-tuned from clock-s2 on plates); MuJoCo gates]")
    if dec["verdict"] == "NEGATIVE":
        per = ", ".join(f"s{s} v2 {rows[s]['v2_verdict']} / {rows[s]['qualify_verdict']} (push "
                        f"{rows[s]['falls']}/{rows[s]['n']})" for s in SEEDS)
        return f"PLATECROSS SELECTION: {NEGATIVE_LINE} (qualified = v2 PASS AND QUALIFIED; {per})"
    return f"PLATECROSS SELECTION: INCOMPLETE ({dec['reason']}); nothing written"


def cmd_select(a) -> int:
    res = Path(a.res)
    out = res / "selection.json"
    if out.exists():
        try:
            d = _json(out)
            print(f"selection already recorded, not overwritten: {out}")
            print(d.get("line", "PLATECROSS SELECTION: (no line in the recorded JSON)"))
        except Exception as exc:  # noqa: BLE001
            print(f"PLATECROSS SELECTION: an unreadable {out} exists ({exc!r}); not overwritten")
        return 0
    dec, rows = selection(res, a.prefix, a.suffix, a.iters)
    line = selection_line(dec, rows)
    if dec["verdict"] == "INCOMPLETE":
        print(line)
        return 0
    doc = {"verdict": dec["verdict"], "line": line, "reason": dec["reason"],
           "selected": None, "bench": NEGATIVE_LINE if dec["verdict"] == "NEGATIVE" else BENCH_NOTE,
           "per_seed": {f"s{s}": {k: rows[s].get(k) for k in ("run", "run_dir", "deploy", "v2_verdict",
                                                              "qualify_verdict", "qualified", "falls", "n",
                                                              "qualify_detail", "inputs")}
                        for s in SEEDS},
           "rule": FROZEN_RULE, "labels": FROZEN_LABELS, "reading": READING, "v2_rule": V2_RULE,
           "task": TASK_ID, "parent": f"{PARENT_RUN}/model_{PARENT_ITER}.pt (task {PARENT_TASK})",
           "final_ckpt": final_ckpt(a.iters),
           "disclosed": DISCLOSED,
           "written_by": os.environ.get("SLURM_JOB_ID", f"local{os.getpid()}"), "time": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    if dec["verdict"] == "SELECTED":
        r = rows[dec["seed"]]
        doc["selected"] = {"seed": dec["seed"], "run": r["run"], "run_dir": r["run_dir"],
                           "push": {"falls": r["falls"], "n": r["n"], "rate": round(r["falls"] / r["n"], 4)},
                           "v2_verdict": r.get("v2_verdict"), "qualify_verdict": r.get("qualify_verdict"),
                           **stage_gait(r)}
    if write_once(out, doc, os.environ.get("SLURM_JOB_ID", "local")):
        print(f"wrote {out}")
    else:
        print(f"selection already recorded by another job, not overwritten: {out}")
    print(line)
    return 0


# =============================================================================== env.yaml against the parent

# Paths where a PlateCross run's params/env.yaml may differ from arms-turngait-clock-s2's: the training seed
# and the declared terrain change; a smoke also runs fewer envs. Everything else must be identical.
ENV_ALLOWED = {("seed",), ("scene", "terrain", "terrain_type"), ("scene", "terrain", "terrain_generator"),
               ("scene", "terrain", "max_init_terrain_level"), ("scene", "terrain", "visual_material")}
ENV_ALLOWED_SMOKE = {("scene", "num_envs"), ("scene", "terrain", "num_envs")}


def env_diff(env: dict, parent: dict, smoke: bool = False) -> list[str]:
    allowed = ENV_ALLOWED | (ENV_ALLOWED_SMOKE if smoke else set())
    out: list[str] = []

    def walk(a, b, path):
        if path in allowed:
            return
        if isinstance(a, dict) and isinstance(b, dict):
            for k in sorted(set(a) | set(b), key=str):
                if k not in a or k not in b:
                    if path + (k,) not in allowed:
                        out.append(".".join(map(str, path + (k,))) + (" (new)" if k not in b else " (missing)"))
                else:
                    walk(a[k], b[k], path + (k,))
        elif a != b:
            out.append(".".join(map(str, path)) + f": {a!r} != parent {b!r}")

    walk(env, parent, ())
    return out


def _near(a, b, tol=1e-9) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


def terrain_problems(env: dict) -> list[str]:
    """The declared terrain, as recorded in a run's env.yaml."""
    pt = terrain()
    p: list[str] = []
    t = ((env.get("scene") or {}).get("terrain")) or {}
    if t.get("terrain_type") != "generator":
        p.append(f"terrain_type {t.get('terrain_type')!r} != 'generator'")
    if t.get("max_init_terrain_level", "absent") is not None:
        p.append(f"max_init_terrain_level {t.get('max_init_terrain_level', 'absent')!r} != None")
    if t.get("visual_material", "absent") is not None:
        p.append("visual_material is not None")
    g = t.get("terrain_generator") or {}
    if not str(g.get("class_type", "")).endswith(":TerrainGenerator"):
        p.append(f"terrain_generator class {g.get('class_type')!r}")
    want = {"border_width": pt.BORDER_M, "num_rows": pt.NUM_ROWS, "num_cols": pt.NUM_COLS, "seed": pt.TERRAIN_SEED}
    for k, v in want.items():
        if not _near(g.get(k), v):
            p.append(f"terrain_generator.{k} {g.get(k)!r} != {v}")
    want_curriculum = VARIANT == "v2"            # v2: the column layout by proportion (F1); v1: random, one sub-terrain
    if g.get("curriculum") is not want_curriculum or g.get("use_cache") is not False:
        p.append(f"curriculum / use_cache {g.get('curriculum')!r} / {g.get('use_cache')!r} (declared "
                 f"{want_curriculum} / False)")
    size = list(g.get("size") or [])
    if len(size) != 2 or not all(_near(x, pt.TILE_M) for x in size):
        p.append(f"tile size {size!r} != [{pt.TILE_M}, {pt.TILE_M}]")
    subs = g.get("sub_terrains") or {}
    want_subs = list(pt.V2_SUB_TERRAINS) if VARIANT == "v2" else ["plates"]
    if sorted(subs) != sorted(want_subs):        # which tiles hold plates is measured by probe-scene, not read here
        p.append(f"sub_terrains {sorted(subs)!r} != {sorted(want_subs)!r}")
    if VARIANT == "v2":
        f = subs.get("flat") or {}
        if f.get("function") != "isaaclab.terrains.trimesh.mesh_terrains:flat_terrain":
            p.append(f"flat function {f.get('function')!r}")
        if not _near(f.get("proportion"), pt.V2_PROPORTIONS[1]):
            p.append(f"flat.proportion {f.get('proportion')!r} != {pt.V2_PROPORTIONS[1]}")
    s = subs.get("plates") or {}
    if s.get("function") != "bhl_robust.tasks.platecross_terrain:plates_terrain":
        p.append(f"plates function {s.get('function')!r}")
    plate_share = pt.V2_PROPORTIONS[0] if VARIANT == "v2" else 1.0
    for k, v in (("proportion", plate_share), ("round_radius_m", pt.ROUND_RADIUS_M), ("square_side_m", pt.SQUARE_SIDE_M),
                 ("height_m", pt.PLATE_HEIGHT_M), ("pitch_m", pt.PITCH_M), ("disc_sections", pt.DISC_SECTIONS)):
        if not _near(s.get(k), v):
            p.append(f"plates.{k} {s.get(k)!r} != {v}")
    if s.get("clear_centre") is not True:
        p.append(f"plates.clear_centre {s.get('clear_centre')!r} != True")
    if s.get("flat_patch_sampling") is not None:
        p.append("plates.flat_patch_sampling is set")
    return p


def check_env(env: dict, parent: dict, smoke: bool = False) -> list[str]:
    return [f"differs from {PARENT_RUN}: {d}" for d in env_diff(env, parent, smoke)] + terrain_problems(env)


def load_env_yaml(path):
    sys.path.insert(0, str(REPO / "src"))
    from bhl_robust.eval.gait_clock import load_env_yaml as _load  # numpy / yaml only

    return _load(Path(path))


def cmd_check_env(a) -> int:
    try:
        probs = check_env(load_env_yaml(a.env_yaml), load_env_yaml(a.parent_env_yaml), a.smoke)
    except Exception as exc:  # noqa: BLE001
        probs = [f"check crashed: {exc!r}"]
    if probs:
        print(f"PLATECROSS ENV: FAIL ({'; '.join(probs[:12])}{' ...' if len(probs) > 12 else ''})")
    else:
        print(f"PLATECROSS ENV: OK {a.env_yaml} (= {PARENT_RUN}'s env.yaml + the declared terrain"
              f"{' + the smoke env count' if a.smoke else ''} and the training seed only)")
    return 0


# =============================================================================== training record

def cmd_write_record(a) -> int:
    b = lambda s: s == "true"  # noqa: E731
    rec = {"run": a.run, "task": a.task, "seed": a.seed, "final_ckpt": a.final_ckpt, "run_dir": a.run_dir,
           "job": a.job, "parent_run_dir": a.parent_dir, "parent_ckpt": a.parent_ckpt,
           "parent_ckpt_sha256": a.parent_sha256, "final_ckpt_present": b(a.ckpt_ok), "task_id_in_log": b(a.task_ok),
           "parent_loaded_in_log": b(a.parent_ok), "plates_marker_in_log": b(a.plates_ok),
           "feet_gait_in_log": b(a.gait_ok), "push_robot_in_log": b(a.push_ok), "guard": a.guard.split(),
           "env_check": a.env_check, "recipe_check": a.recipe_check,
           "clause": ("Training counts only if the run dir holds the final checkpoint and the fresh training log shows the "
                      "PlateCross task id, the parent checkpoint being loaded, the plates terrain marker, the feet_gait "
                      "term and the push event, with an empty guard list and both env.yaml checks OK."),
           "note": f"fine-tune of {PARENT_RUN}/model_{PARENT_ITER}.pt; written once from this job's fresh training log"}
    if write_once(Path(a.out), rec, a.job):
        print(f"wrote {a.out}")
    else:
        print(f"training record already exists, not overwritten: {a.out}")
    return 0


def cmd_check_record(a) -> int:
    try:
        why = record_problems(_json(Path(a.record)), a.run, a.final_ckpt, a.run_dir)
    except Exception as exc:  # noqa: BLE001
        why = [f"unreadable-training-record({type(exc).__name__})"]
    print(" ".join(why))
    return 0


# =============================================================================== the export is the final checkpoint

EXPORT_TOL = 1e-4


def export_actor_error(onnx_path, ckpt_path, n: int = 8, seed: int = 0) -> float:
    """max |ONNX(x) - actor(x)| over n random observations, the actor rebuilt from an rsl_rl checkpoint's
    model_state_dict (its Linear layers with ELU between them, as the runner's [256, 128, 128] ELU MLP; no
    observation normalizer, as empirical_normalization is False for this runner)."""
    import numpy as np
    import onnxruntime as ort
    import torch

    sd = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)["model_state_dict"]
    if any(k.startswith("actor_obs_normalizer.") for k in sd):
        raise ValueError("the checkpoint carries an actor observation normalizer; this check assumes none")
    layers = sorted({int(k.split(".")[1]) for k in sd if k.startswith("actor.") and k.endswith(".weight")})
    if not layers:
        raise ValueError("no actor.<i>.weight in the checkpoint")
    so = ort.SessionOptions()
    so.intra_op_num_threads = so.inter_op_num_threads = 1      # no thread-affinity noise on shared nodes
    sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    width = int(sd[f"actor.{layers[0]}.weight"].shape[1])
    x = np.random.default_rng(seed).normal(size=(n, width)).astype(np.float32)
    h = torch.from_numpy(x).double()
    for j, k in enumerate(layers):
        h = h @ sd[f"actor.{k}.weight"].double().T + sd[f"actor.{k}.bias"].double()
        if j < len(layers) - 1:
            h = torch.nn.functional.elu(h)
    out = np.concatenate([np.asarray(sess.run(None, {inp.name: x[i:i + 1]})[0], dtype=np.float64).reshape(1, -1)
                          for i in range(n)])
    return float(np.abs(out - h.numpy()).max())


def cmd_check_export(a) -> int:
    try:
        err = export_actor_error(a.onnx, a.ckpt)
        par = export_actor_error(a.onnx, a.parent)
    except Exception as exc:  # noqa: BLE001
        print(f"PLATECROSS EXPORT: FAIL (check crashed: {exc!r})")
        return 0
    ok = err <= EXPORT_TOL
    print(f"PLATECROSS EXPORT: {'OK' if ok else 'FAIL'} (max |policy.onnx - {Path(a.ckpt).name} actor| = {err:.2e}, "
          f"tolerance {EXPORT_TOL:g}; against the parent {Path(a.parent).name}: {par:.2e}, reported)")
    return 0


# =============================================================================== smoke stand-ins

def cmd_smoke_standins(a) -> int:
    """Three stand-in seeds for the selection plumbing, from REAL R1 qualify / v2 JSONs (arms-turngait-clock-s0/1/2:
    v2 PASS / NOT QUALIFIED 8/60, v2 FAIL / NOT QUALIFIED 8/60, v2 PASS / QUALIFIED 9/60), re-pointed at the smoke's
    own run and export. The rule must then select stand-in seed 2. --joint-check (SYNTHETIC): stand-in seed 1's
    qualify JSON is relabelled QUALIFIED 0/60 while its real v2 JSON stays FAIL; counting cpu_turn_qualify alone
    would select seed 1, the joint reading must still select seed 2. Refuses any destination outside a '-smoke'
    directory."""
    dest = Path(a.dest).resolve()
    if not any(part.endswith("-smoke") for part in dest.parts):
        print(f"SMOKE-STANDINS: FAIL (destination {dest} is not under a directory ending in -smoke)")
        return 1
    rec = _json(Path(a.training_record))
    run_dir = rec["run_dir"]
    deploy = str(Path(run_dir) / "exported" / "deploy.yaml")
    src = Path(a.source_res)
    docs = []
    for s in SEEDS:
        run = run_name(s, a.prefix, a.suffix)
        r = dict(rec, run=run, seed=s, standin=f"smoke stand-in: the smoke's own record, renamed to seed {s}")
        v2 = _json(src / "turn-test-v2" / f"arms-turngait-clock-s{s}.json")
        v2.update(deploy=deploy, standin=f"smoke stand-in: R1 arms-turngait-clock-s{s}'s v2 JSON, re-pointed")
        q = _json(src / "qualify" / f"arms-turngait-clock-s{s}__qualify.json")
        q.update(run=run, deploy=deploy, standin=f"smoke stand-in: R1 arms-turngait-clock-s{s}'s qualify JSON, re-pointed")
        if a.joint_check and s == 1:
            try:
                if v2.get("verdict") != "FAIL":
                    raise ValueError(f"R1 s1's real v2 verdict is {v2.get('verdict')!r}, not FAIL")
                q["verdict"] = "QUALIFIED"
                q["clauses"]["push"].update(falls=0, rate=0.0)
            except Exception as exc:  # noqa: BLE001
                print(f"SMOKE-STANDINS: FAIL (--joint-check: {exc!r})")
                return 1
            q["standin"] = ("SYNTHETIC smoke stand-in: R1 arms-turngait-clock-s1's qualify JSON relabelled QUALIFIED "
                            "0/60, its real v2 JSON (FAIL) kept: the joint reading must not select it")
        docs.append((run, r, v2, q))
    for sub in ("training", "turn-test-v2", "qualify"):
        (dest / sub).mkdir(parents=True, exist_ok=True)
    for run, r, v2, q in docs:
        for path, doc in ((dest / "training" / f"{run}.json", r), (dest / "turn-test-v2" / f"{run}.json", v2),
                          (dest / "qualify" / f"{run}__qualify.json", q)):
            path.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"SMOKE-STANDINS: wrote 3 stand-in seeds under {dest} (expected selection: seed 2"
          f"{'; SYNTHETIC s1 = v2 FAIL + QUALIFIED 0/60, never selected' if a.joint_check else ''})")
    return 0


# =============================================================================== the scene probe (Isaac)

def cmd_probe_scene(argv: list[str]) -> int:
    from isaaclab.app import AppLauncher

    ap = argparse.ArgumentParser(prog="platecross_select.py probe-scene")
    ap.add_argument("--task", default=TASK_ID)
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--out", required=True)
    AppLauncher.add_app_launcher_args(ap)
    args = ap.parse_args(argv)
    launcher = AppLauncher(args)
    res = {"verdict": "FAIL", "problems": [], "complete": False, "task": args.task, "num_envs": args.num_envs,
           "steps": args.steps, "seed": args.seed, "label": "scene check (no policy); exploration seed"}
    try:
        _probe_body(args, res)
    except BaseException as exc:  # noqa: BLE001
        res["problems"].append(f"probe crashed: {exc!r}")
        print("PROBE-ERROR\n" + traceback.format_exc(), flush=True)
    finally:
        res["verdict"] = "PASS" if res["complete"] and not res["problems"] else "FAIL"
        out = Path(args.out)
        tmp = out.with_name(f".{out.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(res, indent=2, default=str) + "\n")
        tmp.replace(out)
        f = (res.get("plates") or {}).get("found", {})
        print(f"PLATECROSS PROBE: {res['verdict']} (plates round {f.get('round')} square {f.get('square')} other "
              f"{f.get('other')}; problems: {'; '.join(res['problems'][:6]) or 'none'})", flush=True)
        launcher.app.close()
    return 0


def _probe_body(args, res: dict) -> None:
    import gymnasium as gym
    import numpy as np
    import torch

    import bhl_robust.tasks  # noqa: F401  (registers the ids; Isaac's app exists now)
    from pxr import Usd, UsdGeom, UsdPhysics

    def current_stage():
        try:
            import isaaclab.sim as sim_utils                  # Isaac Lab 2.3: the stage the scene was built in

            return sim_utils.get_current_stage()
        except (ImportError, AttributeError):
            import omni.usd

            return omni.usd.get_context().get_stage()

    pt = terrain()
    P = res["problems"]
    spec = gym.spec(args.task)
    cfg = spec.kwargs["env_cfg_entry_point"]()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    tg = cfg.scene.terrain.terrain_generator
    res["terrain_cfg"] = {"terrain_type": cfg.scene.terrain.terrain_type, "size": list(tg.size),
                          "num_rows": tg.num_rows, "num_cols": tg.num_cols, "border_width": tg.border_width}
    env = gym.make(args.task, cfg=cfg)
    u = env.unwrapped
    env.reset()
    # 1. the actor / critic observation widths (R1's 77 / 80)
    dims = {k: list(u.observation_manager.group_obs_dim[k]) for k in ("policy", "critic")}
    res["obs_dims"] = dims
    if dims["policy"] != [OBS_POLICY] or dims["critic"] != [OBS_CRITIC]:
        P.append(f"observation widths {dims} (declared policy {OBS_POLICY}, critic {OBS_CRITIC})")
    # 2. the imported terrain mesh: one Mesh prim under /World/ground, a static triangle-mesh collider
    stage = current_stage()
    root = stage.GetPrimAtPath("/World/ground")
    meshes = [p for p in Usd.PrimRange(root) if p.IsA(UsdGeom.Mesh)] if root and root.IsValid() else []
    res["terrain_mesh_prims"] = [str(p.GetPath()) for p in meshes]
    if len(meshes) != 1:
        P.append(f"{len(meshes)} Mesh prims under /World/ground (expected 1)")
        return
    prim = meshes[0]
    mesh = UsdGeom.Mesh(prim)
    pts = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
    idx = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
    if counts.size == 0 or not (counts == 3).all():
        P.append("terrain mesh is not all triangles")
        return
    m = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()), dtype=np.float64)
    pts_w = pts @ m[:3, :3] + m[3, :3]
    chain, q = [], prim
    while q and q.IsValid() and str(q.GetPath()) != "/":
        chain.append(q)
        q = q.GetParent()
    coll = prim.HasAPI(UsdPhysics.CollisionAPI)
    enabled = bool(UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()) if coll else False
    rigid = [str(c.GetPath()) for c in chain if c.HasAPI(UsdPhysics.RigidBodyAPI)]
    res["collider"] = {"prim": str(prim.GetPath()), "collision_api": coll, "collision_enabled": enabled,
                       "rigid_body_api_on": rigid, "n_vertices": int(len(pts)), "n_triangles": int(len(counts)),
                       "static": coll and enabled and not rigid}
    if not (coll and enabled) or rigid:
        P.append(f"terrain mesh is not a static collider (CollisionAPI {coll}, enabled {enabled}, RigidBodyAPI on {rigid})")
    # 3. the plates in that mesh against the declared field
    found = pt.find_plates(pts_w, idx.reshape(-1, 3))
    if args.task == VARIANTS["v2"]["task"]:     # F1: the plate tile in the plate columns only, a plane elsewhere
        expected = pt.expected_world_plates_v2(tg.num_rows, tg.num_cols, tuple(tg.size), pt.PITCH_M, pt.CLEAR_CENTRE)
        res["variant"] = {"name": "v2", "plate_columns": pt.v2_plate_columns(tg.num_cols)}
    else:
        expected = pt.expected_world_plates(tg.num_rows, tg.num_cols, tuple(tg.size), pt.PITCH_M, pt.CLEAR_CENTRE)
    summary, probs = pt.check_scene_plates(found, expected)
    res["plates"] = summary
    P.extend(probs)
    # 4. robots spawn on tile centres, clear of every plate
    origins = u.scene.env_origins[:, :2].detach().cpu().numpy()
    centres = pt.tile_centres(tg.num_rows, tg.num_cols, tuple(tg.size))
    off = np.min(np.linalg.norm(origins[:, None, :] - centres[None, :, :], axis=2), axis=1)
    clear = pt.spawn_clearance(origins, found)
    res["spawn"] = {"max_origin_offset_from_a_tile_centre_m": float(off.max()), "n_tiles_used": int(len(np.unique(
        np.round(origins, 3), axis=0))), "min_plate_clearance_chebyshev_m": clear,
        "declared_clearance_m": pt.PITCH_M - 0.5 * pt.SQUARE_SIDE_M}
    if off.max() > 1e-3:
        P.append(f"an env origin is {off.max():.4f} m from every tile centre")
    if not clear >= pt.SPAWN_HALF_RANGE_M + 0.3:
        P.append(f"a plate footprint is {clear:.3f} m (Chebyshev) from a spawn origin (< {pt.SPAWN_HALF_RANGE_M + 0.3})")
    robot = u.scene["robot"]
    act = torch.zeros(u.num_envs, u.action_manager.total_action_dim, device=u.device)
    o = u.scene.env_origins
    feet, _ = robot.find_bodies(".*_ankle_roll")
    try:
        root_body = robot.find_bodies("base")[0]          # the termination's body_names="base"
    except Exception:  # noqa: BLE001
        root_body = [0]
    others = [b for b in range(robot.num_bodies) if b not in root_body]
    res["bodies"] = {"root": [robot.body_names[b] for b in root_body], "feet": [robot.body_names[b] for b in feet],
                     "n": int(robot.num_bodies)}
    # 5. the plates carry the feet (a static collider the feet stand on): 64 robots placed upright on plates (the
    #    round one at +1.2 / +1.2 m and the square one at +1.2 / 0 m from their tile centre, alternately), 64 on the
    #    floor at their tile centre; default joint pose, zero velocity, root 0.03 m higher on a plate (the BHL root
    #    frame is at ground level, its soles ~2 cm above it), then PLATE_STEPS zero-action steps. Feet on plates must
    #    stand 0.03 m (+/- 0.01) higher than feet on the floor, none of them below half that. Robots reset meanwhile
    #    (fallen) are left out; at least 3/4 must remain.
    n_place = min(64, u.num_envs // 2)
    plate_ids = torch.arange(0, n_place, device=u.device)
    floor_ids = torch.arange(n_place, 2 * n_place, device=u.device)
    ids = torch.cat([plate_ids, floor_ids])
    off = torch.tensor([[pt.PITCH_M, pt.PITCH_M, pt.PLATE_HEIGHT_M] if k % 2 == 0 else [pt.PITCH_M, 0.0, pt.PLATE_HEIGHT_M]
                        for k in range(n_place)], device=u.device, dtype=o.dtype)
    pose = robot.data.default_root_state[:, :7].clone()
    pose[plate_ids, :3] = o[plate_ids] + off
    pose[floor_ids, :3] = o[floor_ids]
    target = pose[plate_ids, :2].detach().cpu().numpy()
    centres = np.array([[p["x"], p["y"]] for p in found]) if found else np.zeros((0, 2))
    shapes_at = []
    for x, y in target:
        k = int(np.argmin(np.hypot(centres[:, 0] - x, centres[:, 1] - y))) if len(centres) else -1
        ok = k >= 0 and math.hypot(centres[k, 0] - x, centres[k, 1] - y) < 1e-3
        shapes_at.append(found[k]["shape"] if ok else "none")
    robot.write_root_pose_to_sim(pose[ids], env_ids=ids)
    robot.write_root_velocity_to_sim(torch.zeros(len(ids), 6, device=u.device), env_ids=ids)
    robot.write_joint_state_to_sim(robot.data.default_joint_pos[ids].clone(),
                                   torch.zeros_like(robot.data.default_joint_vel[ids]), env_ids=ids)
    before = u.episode_length_buf[ids].clone()
    for _ in range(PLATE_STEPS):
        env.step(act)
    kept = (u.episode_length_buf[ids] == before + PLATE_STEPS).detach().cpu().numpy()
    fz = (robot.data.body_pos_w[:, feet, 2] - o[:, 2:3]).detach().cpu().numpy()
    cs = u.scene.sensors["contact_forces"]
    cfeet, _ = cs.find_bodies(".*_ankle_roll")
    touch = (cs.data.net_forces_w[:, cfeet].norm(dim=-1) > 1.0).detach().cpu().numpy()
    kp, kf = kept[:n_place], kept[n_place:]
    pz, flz = fz[:n_place][kp], fz[n_place:2 * n_place][kf]
    drop = {"robots_on_plates": int(n_place), "robots_on_floor": int(n_place), "steps": PLATE_STEPS,
            "plate_shapes_under_robots": {s: shapes_at.count(s) for s in sorted(set(shapes_at))},
            "kept_on_plates": int(kp.sum()), "kept_on_floor": int(kf.sum())}
    if len(pz) and len(flz):
        step = float(np.median(pz) - np.median(flz))
        drop.update(median_foot_origin_z_plate_m=float(np.median(pz)), median_foot_origin_z_floor_m=float(np.median(flz)),
                    min_foot_origin_z_plate_m=float(pz.min()), plate_step_m=step,
                    feet_in_contact_plate=float(touch[:n_place][kp].mean()), feet_in_contact_floor=float(touch[n_place:2 * n_place][kf].mean()))
        if not 0.02 <= step <= 0.04:
            P.append(f"feet on plates stand {step:.4f} m above feet on the floor (declared plate height 0.03 m +/- 0.01)")
        if pz.min() < np.median(flz) + 0.5 * pt.PLATE_HEIGHT_M:
            P.append(f"a foot on a plate is at {pz.min():.4f} m, below half a plate above the floor feet ({np.median(flz):.4f} m)")
    if kp.sum() < 0.75 * n_place or kf.sum() < 0.75 * n_place:
        P.append(f"only {int(kp.sum())} / {int(kf.sum())} of {n_place} robots stayed up on plates / floor")
    if shapes_at.count("round") != (n_place + 1) // 2 or shapes_at.count("square") != n_place // 2:
        P.append(f"the placement points are not the declared plates: {drop['plate_shapes_under_robots']}")
    res["plate_drop"] = drop
    # 6. zero-action steps: finite observations and no link below the floor. The BHL root ("base") frame sits at
    #    ground level under the pelvis (MJCF base pos 0, pelvis box at +0.71 m), so it goes below 0 when the pelvis
    #    drops; every other link frame is inside its own geometry and must stay above -0.03 m.
    min_link, min_root, finite = float("inf"), float("inf"), True
    for _ in range(args.steps):
        obs, _, _, _, _ = env.step(act)
        finite = finite and bool(torch.isfinite(obs["policy"]).all())
        z = robot.data.body_pos_w[:, :, 2] - o[:, 2:3]
        min_link = min(min_link, float(z[:, others].min()))
        min_root = min(min_root, float(robot.data.root_pos_w[:, 2].sub(o[:, 2]).min()))
    res["rollout"] = {"zero_action_steps": args.steps, "finite_obs": finite, "min_link_origin_z_m (all but base)": min_link,
                      "min_root_frame_z_m (info: ground-level frame)": min_root}
    if not finite:
        P.append("non-finite policy observation during the zero-action steps")
    if not min_link > -0.03:
        P.append(f"a robot link went to {min_link:.3f} m (below the floor)")
    env.close()
    res["complete"] = True


# =============================================================================== CLI

def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["probe-scene"]:
        return cmd_probe_scene(argv[1:])
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--res", required=True)
    s.add_argument("--prefix", default=PREFIX)
    s.add_argument("--suffix", default="")
    s.add_argument("--iters", type=int, default=FT_ITERS)
    e = sub.add_parser("check-env")
    e.add_argument("--env-yaml", required=True)
    e.add_argument("--parent-env-yaml", required=True)
    e.add_argument("--smoke", action="store_true")
    w = sub.add_parser("write-record")
    for k in ("out", "run", "task", "final-ckpt", "run-dir", "job", "parent-dir", "parent-ckpt", "parent-sha256",
              "ckpt-ok", "task-ok", "parent-ok", "plates-ok", "gait-ok", "push-ok", "env-check", "recipe-check"):
        w.add_argument(f"--{k}", required=True)
    w.add_argument("--seed", type=int, required=True)
    w.add_argument("--guard", default="")
    c = sub.add_parser("check-record")
    for k in ("record", "run", "final-ckpt", "run-dir"):
        c.add_argument(f"--{k}", required=True)
    x = sub.add_parser("check-export")
    for k in ("onnx", "ckpt", "parent"):
        x.add_argument(f"--{k}", required=True)
    t = sub.add_parser("smoke-standins")
    for k in ("dest", "source-res", "training-record"):
        t.add_argument(f"--{k}", required=True)
    t.add_argument("--prefix", default=PREFIX)
    t.add_argument("--suffix", default="-smoke")
    t.add_argument("--joint-check", action="store_true")
    a = ap.parse_args(argv)
    return {"select": cmd_select, "check-env": cmd_check_env, "write-record": cmd_write_record,
            "check-record": cmd_check_record, "check-export": cmd_check_export,
            "smoke-standins": cmd_smoke_standins}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
