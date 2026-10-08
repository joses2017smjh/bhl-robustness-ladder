"""Turn a depth_bench.txt into JSON and check it against the published depth claims.

The published numbers (docs/REPORT.md "Validation first" / throughput table,
docs/FINDINGS.md section 5, commit b34b99a) have no surviving evidence file, so
this re-reads a fresh BENCH_OUT written by

  scripts/bench/depth_validate.py   CHECK/PASS/FAIL lines
  scripts/bench/depth_raycast.py    RESULT (+ DEPTH) lines, one per config

plus an optional ``NODE | key=value ...`` line the launcher writes first (host,
GPU, job, git HEAD), because env-steps/s is a property of the card as much as
of the code and the docs never named theirs.

Every claim is compared with a relative tolerance (default 10 %) and printed as
PASS, MISMATCH or NOT_MEASURED. ``measurement_complete`` is a separate question
-- did the bench produce everything depth_bench.sh is supposed to produce? --
and it alone decides the exit code unless --strict is given. Stdlib only, so
it runs anywhere (no Isaac import).

    python scripts/bench/depth_bench_summary.py results/.../depth_bench.txt \
        --json results/.../depth_bench_summary.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# ----------------------------------------------------------------- documented
# Values exactly as published. Rows are keyed (mode, res, envs); res 0 = no camera.
DOC_FINITE_FRACTION = 1.000          # "100% finite pixels"
DOC_MEAN_REL_ERR = 0.029             # "2.9% mean relative error"
DOC_MRE_PIXELS = 3136                # "over 3,136 pixels"
DOC_ROW_FLIPPED_ERR = 1.02           # "row-flipped hypothesis comes in at 102% error"
DOC_BEST_ORIENTATION = "as-returned"
DOC_DEPTH_COST_4096 = 0.016          # "1.6% of throughput at 4,096 envs"

DOC_ROWS = {
    # (mode, res, envs): (env-steps/s, ms/step, required-for-completeness)
    ("physics", 0, 2048): (13971, 146.6, True),
    ("depth", 32, 2048): (13989, 146.4, False),   # REPORT only; not in depth_bench.sh
    ("depth", 64, 2048): (13956, 146.7, True),
    ("physics", 0, 4096): (21844, 187.5, True),
    ("depth", 48, 4096): (21490, 190.6, True),
}

# ------------------------------------------------------------------- patterns
RE_NODE = re.compile(r"^NODE\s*\|\s*(.*)$")
RE_PITCH = re.compile(r"^CHECK\s*\|\s*camera pitch from cfg quaternion:\s*([-\d.]+) deg")
RE_MEASURE = re.compile(
    r"^CHECK\s*\|\s*(flat plane|generated terrain): camera height\s*([-\d.]+) m, finite\s*([\d.]+)")
RE_ORIENT = re.compile(
    r"^CHECK\s*\|\s*(as-returned|row-flipped)\s*: mean relative error\s*([\d.]+) over\s*(\d+) px")
RE_VERDICT = re.compile(
    r"^(PASS|FAIL)\s*\|\s*flat-ground depth matches closed form to\s*([\w.+-]+)\s*"
    r"\((\S+), tol\s*([\d.]+)\)")
RE_NONFINITE = re.compile(r"^FAIL\s*\|\s*non-finite depth on flat ground")
RE_DEPART = re.compile(r"^CHECK\s*\|\s*rough-ground departure from the flat model:\s*([\d.]+)")
RE_TERRAIN = re.compile(r"^(PASS|FAIL)\s*\|\s*terrain moves the depth image")
RE_SPREAD = re.compile(r"^CHECK\s*\|\s*across-env spread of mean depth on terrain:\s*([\w.+-]+) m")
RE_RESULT = re.compile(
    r"^RESULT\s*\|\s*(physics only|raycast depth (\d+)x(\d+))\s*\|\s*envs=\s*(\d+)\s*\|"
    r"\s*([\d.]+) env-steps/s\s*\|\s*([\d.]+) ms/step")
RE_DEPTH = re.compile(
    r"^DEPTH\s*\|\s*tensor\s*(\([^)]*\))\s*\|\s*finite=([\d.]+)"
    r"(?:\s*\|\s*range \[([-\w.+]+),\s*([-\w.+]+)\] m)?")
RE_STEP = re.compile(r"^STEP\s*\|\s*(.*)$")


def _kv(text: str) -> dict:
    """``a=1 b=two words`` style -> dict; values run to the next ``key=``."""
    out, parts = {}, re.split(r"\s+(?=[A-Za-z_][\w.-]*=)", text.strip())
    for p in parts:
        if "=" in p:
            k, v = p.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _num(s: str | None):
    try:
        return float(s) if s is not None else None
    except ValueError:
        return None


def parse(text: str) -> dict:
    d: dict = {"node": {}, "validate": {}, "orientations": {}, "rows": {},
               "steps": [], "warnings": [], "unparsed_result_lines": []}
    v = d["validate"]
    last_row_key = None
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        if m := RE_NODE.match(line):
            d["node"].update(_kv(m.group(1)))
        elif m := RE_STEP.match(line):
            d["steps"].append(_kv(m.group(1)))
        elif m := RE_PITCH.match(line):
            v["camera_pitch_deg"] = float(m.group(1))
        elif m := RE_MEASURE.match(line):
            key = "flat" if m.group(1) == "flat plane" else "terrain"
            if f"{key}_finite_fraction" in v:
                d["warnings"].append(f"duplicate '{m.group(1)}' measurement; keeping the last")
            v[f"{key}_camera_height_m"] = float(m.group(2))
            v[f"{key}_finite_fraction"] = float(m.group(3))
        elif m := RE_ORIENT.match(line):
            d["orientations"][m.group(1)] = {"mean_relative_error": float(m.group(2)),
                                             "pixels": int(m.group(3))}
        elif m := RE_VERDICT.match(line):
            v["validate_verdict"] = m.group(1)
            v["best_error"] = _num(m.group(2))
            v["best_orientation"] = m.group(3)
            v["validate_tol"] = float(m.group(4))
        elif RE_NONFINITE.match(line):
            v["flat_nonfinite_fail"] = True
        elif m := RE_DEPART.match(line):
            v["rough_departure"] = float(m.group(1))
        elif m := RE_TERRAIN.match(line):
            v["terrain_moves_verdict"] = m.group(1)
        elif m := RE_SPREAD.match(line):
            v["terrain_env_spread_m"] = _num(m.group(1))
        elif m := RE_RESULT.match(line):
            if m.group(1) == "physics only":
                key = ("physics", 0, int(m.group(4)))
            else:
                if m.group(2) != m.group(3):
                    d["warnings"].append(f"non-square depth row ignored: {line}")
                    continue
                key = ("depth", int(m.group(2)), int(m.group(4)))
            name = _row_name(key)
            if name in d["rows"]:
                d["warnings"].append(f"duplicate row {name}; keeping the last")
            d["rows"][name] = {"mode": key[0], "res": key[1], "envs": key[2],
                               "env_steps_per_s": float(m.group(5)),
                               "ms_per_step": float(m.group(6))}
            last_row_key = name
        elif m := RE_DEPTH.match(line):
            if last_row_key is None or d["rows"][last_row_key]["mode"] != "depth":
                d["warnings"].append(f"DEPTH line without a preceding depth RESULT: {line}")
                continue
            d["rows"][last_row_key].update({
                "depth_tensor": m.group(1), "depth_finite_fraction": float(m.group(2)),
                "depth_range_m": [_num(m.group(3)), _num(m.group(4))]})
        elif line.startswith("RESULT"):
            d["unparsed_result_lines"].append(line)
    return d


def _row_name(key) -> str:
    mode, res, envs = key
    return f"physics_{envs}" if mode == "physics" else f"depth{res}x{res}_{envs}"


def _cmp(measured, documented, tol, *, lower_bound_only=False) -> str:
    if measured is None:
        return "NOT_MEASURED"
    if lower_bound_only:        # a fraction documented at 1.0 cannot be exceeded
        return "PASS" if measured >= documented * (1.0 - tol) else "MISMATCH"
    return "PASS" if abs(measured - documented) <= tol * abs(documented) else "MISMATCH"


def evaluate(d: dict, tol: float) -> dict:
    v, rows, claims = d["validate"], d["rows"], []

    def claim(name, measured, documented, verdict, **extra):
        numeric = isinstance(measured, (int, float)) and isinstance(documented, (int, float))
        rel = (measured - documented) / abs(documented) if numeric and documented else None
        claims.append({"claim": name, "measured": measured, "documented": documented,
                       "rel_diff": rel, "verdict": verdict, **extra})

    ff = v.get("flat_finite_fraction")
    claim("finite_fraction", ff, DOC_FINITE_FRACTION,
          _cmp(ff, DOC_FINITE_FRACTION, tol, lower_bound_only=True),
          source="depth_validate flat plane, all envs, 3-decimal print")
    mre = d["orientations"].get(DOC_BEST_ORIENTATION, {}).get("mean_relative_error")
    if mre is None and v.get("best_orientation") == DOC_BEST_ORIENTATION:
        mre = v.get("best_error")
    claim("mean_relative_error", mre, DOC_MEAN_REL_ERR, _cmp(mre, DOC_MEAN_REL_ERR, tol),
          pixels=d["orientations"].get(DOC_BEST_ORIENTATION, {}).get("pixels"),
          documented_pixels=DOC_MRE_PIXELS, validate_verdict=v.get("validate_verdict"))
    flip = d["orientations"].get("row-flipped", {}).get("mean_relative_error")
    claim("row_flipped_relative_error", flip, DOC_ROW_FLIPPED_ERR,
          _cmp(flip, DOC_ROW_FLIPPED_ERR, tol))
    bo = v.get("best_orientation")
    claim("best_orientation", bo, DOC_BEST_ORIENTATION,
          "NOT_MEASURED" if bo is None else ("PASS" if bo == DOC_BEST_ORIENTATION else "MISMATCH"))

    for key, (sps_doc, ms_doc, required) in DOC_ROWS.items():
        name = _row_name(key)
        r = rows.get(name, {})
        sps = r.get("env_steps_per_s")
        claim(f"env_steps_per_s[{name}]", sps, sps_doc, _cmp(sps, sps_doc, tol),
              ms_per_step=r.get("ms_per_step"), documented_ms_per_step=ms_doc,
              depth_finite_fraction=r.get("depth_finite_fraction"),
              required=required, hardware_dependent=True)

    phys, dep = rows.get("physics_4096", {}), rows.get("depth48x48_4096", {})
    cost = None
    if phys.get("env_steps_per_s") and dep.get("env_steps_per_s"):
        cost = 1.0 - dep["env_steps_per_s"] / phys["env_steps_per_s"]
    # 10 % of 1.6 % is +-0.16 pp, below the run-to-run noise of a 60-step
    # timing. The strict verdict is reported as asked; `cost_below_5pct` is the
    # qualitative form of the claim ("depth is nearly free").
    claim("depth_cost_4096", cost, DOC_DEPTH_COST_4096, _cmp(cost, DOC_DEPTH_COST_4096, tol),
          cost_below_5pct=None if cost is None else cost < 0.05,
          note="strict 10%-relative check on a 1.6 % difference; read qualitatively")

    # --- completeness: did the bench produce what depth_bench.sh promises? ---
    missing = []
    for k in ("flat_finite_fraction", "validate_verdict", "best_error"):
        if v.get(k) is None:
            missing.append(f"validate.{k}")
    if "rough_departure" not in v:
        missing.append("validate.rough_departure")
    for key, (_, _, required) in DOC_ROWS.items():
        name = _row_name(key)
        if required and name not in rows:
            missing.append(f"row {name}")
        if required and key[0] == "depth" and name in rows and \
                rows[name].get("depth_finite_fraction") is None:
            missing.append(f"DEPTH line for {name}")
    if d["unparsed_result_lines"]:
        missing.append(f"{len(d['unparsed_result_lines'])} unparsed RESULT line(s)")

    return {"claims": claims, "measurement_complete": not missing, "missing": missing,
            "tolerance": tol}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bench_txt", help="BENCH_OUT file written by depth_validate/depth_raycast")
    ap.add_argument("--json", default="", help="write the summary JSON here")
    ap.add_argument("--tol", type=float, default=0.10, help="relative tolerance per claim (0.10 = 10%%)")
    ap.add_argument("--strict", action="store_true",
                    help="exit 3 if any measured claim MISMATCHes (default: exit on completeness only)")
    args = ap.parse_args()

    p = Path(args.bench_txt)
    if not p.is_file():
        print(f"DEPTH-BENCH FAIL | no bench file at {p}")
        if args.json:
            Path(args.json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.json).write_text(json.dumps(
                {"bench_txt": str(p), "measurement_complete": False,
                 "missing": ["bench file"], "claims": []}, indent=2))
        return 1

    parsed = parse(p.read_text(errors="replace"))
    ev = evaluate(parsed, args.tol)
    out = {"bench_txt": str(p), **ev, "node": parsed["node"], "validate": parsed["validate"],
           "orientations": parsed["orientations"], "rows": parsed["rows"],
           "steps": parsed["steps"], "warnings": parsed["warnings"],
           "unparsed_result_lines": parsed["unparsed_result_lines"]}
    counts = {k: sum(c["verdict"] == k for c in ev["claims"])
              for k in ("PASS", "MISMATCH", "NOT_MEASURED")}
    out["claim_counts"] = counts
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(out, indent=2))

    gpu = parsed["node"].get("gpu", "unknown")
    print(f"NODE   | host={parsed['node'].get('host', '?')} gpu={gpu} "
          f"stack={parsed['node'].get('stack', '?')} head={parsed['node'].get('head', '?')}")
    for c in ev["claims"]:
        m, doc = c["measured"], c["documented"]
        ms = "NA" if m is None else (f"{m:.4f}" if isinstance(m, float) and abs(m) < 10 else
                                     (f"{m:,.0f}" if isinstance(m, float) else str(m)))
        ds = f"{doc:.4f}" if isinstance(doc, float) and abs(doc) < 10 else (
            f"{doc:,}" if isinstance(doc, (int, float)) else str(doc))
        rd = "" if c["rel_diff"] is None else f" ({c['rel_diff']:+.1%})"
        extra = ""
        if c["claim"].startswith("env_steps_per_s") and c.get("depth_finite_fraction") is not None:
            extra = f" depth_finite={c['depth_finite_fraction']:.3f}"
        if c["claim"] == "depth_cost_4096" and c.get("cost_below_5pct") is not None:
            extra = f" cost<5%={'yes' if c['cost_below_5pct'] else 'no'}"
        if c["claim"] == "mean_relative_error" and c.get("pixels") is not None:
            extra = f" px={c['pixels']} (doc {DOC_MRE_PIXELS})"
        print(f"{c['verdict']:12s} | {c['claim']:34s} measured {ms:>10s} vs documented {ds:>8s}{rd}{extra}")
    for w in parsed["warnings"]:
        print(f"WARN   | {w}")
    for s in parsed["steps"]:
        print(f"STEP   | {' '.join(f'{k}={v}' for k, v in s.items())}")

    verdicts = " ".join(f"{k}={v}" for k, v in counts.items())
    if not ev["measurement_complete"]:
        print(f"DEPTH-BENCH FAIL | measurement incomplete: {'; '.join(ev['missing'])} | "
              f"claims {verdicts} | gpu={gpu}")
        return 1
    print(f"DEPTH-BENCH PASS | measurement complete | claims {verdicts} | gpu={gpu}")
    if args.strict and counts["MISMATCH"]:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
