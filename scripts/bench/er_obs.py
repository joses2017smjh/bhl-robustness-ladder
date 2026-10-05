"""ER-OBS-1: Gemini Robotics-ER 2 as an EXTERNAL observer of the scripted cooperative lift-hold-place (design G).

    LEARNED gait (frozen arms-dr1.0-s0) + SCRIPTED arms + EXTERNAL VLM observer
    (Gemini Robotics-ER 2 preview, rendered RGB, offline)

Subcommands (launchers: slurm/repo20260923/gpu_er_obs1_frames.sbatch, cpu_er_obs1_calls.sbatch):

  frames   --mode smoke|run   run the UNCHANGED lift-hold-place harness through its frame_hook and write one
                              fixed-camera RGB frame, its segmentation (cube mask), the C1 frame (cube invisible) and
                              the ORACLE labels per simulated second; check every episode against a read-only rollout
                              of the same seed (bitwise) and every C1 frame against the cube-removed render. run =
                              seeds 300-319, 20 s (400 frames); smoke = throw-away seeds 900-901, shortened episodes.
  calls    --mode mock|run    one request per frame and variant (main, C1) with the ONE fixed prompt and schema
                              (documented ER 2 REST shape); paced (1.0 s gap); re-sent only after HTTP 429/503 or a
                              connection error with no response; every final outcome (failures included) published
                              once under an exclusive per-frame claim; run needs ER_OBS_PAID_TIER_CONFIRMED=1, the key
                              file, the expected prompt+schema and request sha256, and no label class below 20 frames;
                              mock never touches the network or the key and runs on a virtual clock.
  latency  --mode mock|run    50 synchronous calls (no cache, no re-sends; paced), wall time per call.
  verdict  accuracy|latency   pure verdicts from the JSONs, written once, line read back from the JSON.
  hashes                      the prompt+schema sha256 and the request template (no key, no image).
  keycheck                    the key-file guard only (prints OK or the refusal; never the key).
  smoke-verdict frames|calls  the launchers' smoke verdicts from their step statuses and outputs.

Rules, clauses as applied and seed evidence: bhl_robust.eval.er_obs (DESIGN_TEXT, RULE_TEXT, CLAUSES_AS_APPLIED,
SEED_EVIDENCE); every output repeats them with the label and the source sha256s.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]                          # the code root this script runs from (a worktree or the repo)
sys.path.insert(0, str(HERE))
if str(REPO / "src") not in sys.path:
    sys.path.insert(1, str(REPO / "src"))

import numpy as np                                                    # noqa: E402

from bhl_robust.eval import er_obs as E                               # noqa: E402
from bhl_robust.eval.er_obs import client as C                        # noqa: E402
from bhl_robust.eval.er_obs import verdict as V                       # noqa: E402

MAIN_REPO = Path("/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder")
SCORED_LIFT_PLACE = "results/scripted-carry-20260927/score_liftplace_crew2.json"
SOURCES = (
    "src/bhl_robust/eval/er_obs/__init__.py", "src/bhl_robust/eval/er_obs/labels.py",
    "src/bhl_robust/eval/er_obs/render.py", "src/bhl_robust/eval/er_obs/client.py",
    "src/bhl_robust/eval/er_obs/verdict.py", "scripts/bench/er_obs.py", "tests/test_er_obs.py",
    "slurm/repo20260923/gpu_er_obs1_frames.sbatch", "slurm/repo20260923/cpu_er_obs1_calls.sbatch",
    "src/bhl_robust/eval/scripted_carry.py", "src/bhl_robust/eval/coop_replay.py", "src/bhl_robust/eval/livery.py",
    "src/bhl_robust/eval/mjcf_assets.py", "src/bhl_robust/eval/multi_robot.py", "src/bhl_robust/eval/team_airlock.py",
    "scripts/bench/team_airlock.py",
)
FRAME_STEPS = 25                                 # policy steps per simulated second (0.04 s policy step)


# ------------------------------------------------------------------ provenance

def sha256(path) -> str | None:
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_info(root: Path) -> dict:
    def run(*a):
        try:
            return subprocess.run(["git", "-C", str(root), *a], capture_output=True, text=True,
                                  timeout=30).stdout.strip()
        except Exception:                                              # noqa: BLE001
            return None
    return {"head": run("rev-parse", "HEAD"), "status_porcelain_sources": run("status", "--porcelain", "--", *SOURCES)}


def loaded_module_sha256() -> dict:
    """sha256 of every imported module file under the code root or the upstream checkout."""
    code, upstream = REPO.resolve(), (REPO / "external/Berkeley-Humanoid-Lite").resolve()
    out = {}
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        if not f:
            continue
        p = Path(f)
        if not p.is_absolute() or not p.is_file():
            continue
        p = p.resolve()
        if str(p).startswith(str(upstream) + os.sep):
            out["upstream/" + str(p.relative_to(upstream))] = sha256(p)
        elif str(p).startswith(str(code) + os.sep):
            out[str(p.relative_to(code))] = sha256(p)
    return dict(sorted(out.items()))


def provenance(**extra) -> dict:
    return {
        "task": E.TASK, "label": E.LABEL, "design_text": E.DESIGN_TEXT, "rule_text": E.RULE_TEXT,
        "clauses_as_applied": list(E.CLAUSES_AS_APPLIED), "post_review_fixes": list(E.POST_REVIEW_FIXES),
        "tier_amendment": list(E.TIER_AMENDMENT),
        "seed_evidence": list(E.SEED_EVIDENCE),
        "code_root": str(REPO), "git": git_info(REPO),
        "source_sha256": {s: sha256(REPO / s) for s in SOURCES},
        "loaded_module_sha256": loaded_module_sha256(),
        "host": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **extra,
    }


def parse_seeds(raw: str) -> list[int]:
    out = []
    for part in raw.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def write_json_new(path: Path, obj) -> None:
    """Create a JSON file; refuse an existing one (outputs are never overwritten)."""
    V.write_once(path, obj)


# ------------------------------------------------------------------ frames

def frames_guard(args) -> list[int]:
    seeds = parse_seeds(args.seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise SystemExit("ER-OBS-1 FRAMES: REFUSED (empty or repeated seeds)")
    if any(s in E.RESERVED_SEEDS for s in seeds):
        raise SystemExit("ER-OBS-1 FRAMES: REFUSED (seeds 0-39 and 100-129 are reserved by other experiments)")
    if args.mode == "run":
        if seeds != list(E.SCORED_SEEDS) or args.seconds is not None:
            raise SystemExit("ER-OBS-1 FRAMES: REFUSED (run = exactly seeds 300-319, full 20 s episodes)")
    else:
        if any(s in E.SCORED_SEEDS for s in seeds) or not set(seeds) <= set(E.SMOKE_SEEDS):
            raise SystemExit("ER-OBS-1 FRAMES: REFUSED (smoke runs only the throw-away seeds 900-901)")
        if args.seconds is None or not 1.0 <= args.seconds < E.EPISODE_S:
            raise SystemExit("ER-OBS-1 FRAMES: REFUSED (smoke needs --seconds in [1, 20))")
    if args.out.exists():
        raise SystemExit(f"ER-OBS-1 FRAMES: REFUSED ({args.out} exists; outputs are never overwritten)")
    return seeds


def harness_check(sc, p, q, cfg, policy_path) -> dict:
    """The harness is the one LIFT_PLACE_RULE was scored with (2026-09-27): same rule, parameters, lowering,
    keyframes and gait checkpoint."""
    ref_path = REPO / SCORED_LIFT_PLACE
    ref = json.loads(ref_path.read_text())
    got = {"rule": sc.LIFT_PLACE_RULE, "params": sc.params_dict(p), "place_params": sc.place_params_dict(q),
           "keyframes_left": sc.KEYFRAMES_LEFT}
    got = json.loads(json.dumps(got))
    same = {k: got[k] == ref[k] for k in got}
    ck = sha256(policy_path)
    same["checkpoint_sha256"] = ck == ref["checkpoint_sha256"]
    same["physics_dt"] = float(cfg.physics_dt) == float(ref["physics_dt"])
    same["policy_dt"] = float(cfg.policy_dt) == float(ref["policy_dt"])
    return {"reference": SCORED_LIFT_PLACE, "reference_sha256": sha256(ref_path), "equal": same,
            "ok": all(same.values()), "checkpoint_sha256": ck}


def cmd_frames(args) -> int:
    seeds = frames_guard(args)
    import mujoco
    from omegaconf import OmegaConf
    from team_airlock import CpuPolicy
    from bhl_robust.eval import scripted_carry as sc
    from bhl_robust.eval.er_obs import render as R

    cfg = OmegaConf.load(args.deploy)
    if cfg.num_actions != 22 or cfg.num_joints != 22 or cfg.num_observations != 75:
        raise SystemExit("requires the full 22-DoF, 75-observation humanoid locomotion policy")
    if abs(1.0 / float(cfg.policy_dt) - FRAME_STEPS) > 1e-9:
        raise SystemExit(f"policy_dt {cfg.policy_dt} does not give {FRAME_STEPS} steps per second")
    policy = CpuPolicy(cfg.policy_checkpoint_path)
    p, q = sc.CarryParams(), sc.PlaceParams()                  # stock pads, frozen lift-place script
    rule = dict(sc.LIFT_PLACE_RULE) if args.mode == "run" else dict(sc.LIFT_PLACE_RULE, episode_s=float(args.seconds))
    model, slots, pairs = sc.build_carry(args.upstream, args.cache_dir, 1, p)      # crew 2 = one pair
    pair = pairs[0]
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    hc = harness_check(sc, p, q, cfg, cfg.policy_checkpoint_path)
    args.out.mkdir(parents=True)
    rd = R.FrameRenderer(model, pair.cube_geom)
    n_frames = int(round(rule["episode_s"] / float(cfg.policy_dt))) // FRAME_STEPS
    frames, episodes, problems = [], [], []
    if not hc["ok"]:
        problems.append(f"harness differs from the scored lift-place run: {hc['equal']}")
    t_all = time.time()
    for seed in seeds:
        t0 = time.time()
        hook = R.FrameHook(rd, model, pair, floor, seed, args.out, p.t_settle, FRAME_STEPS)
        ep = sc.run_place_episode(model, slots, pairs, cfg, policy, seed, p, q=q, frame_hook=hook, rule=rule)
        t1 = time.time()
        reader = R.QposReader(FRAME_STEPS)
        ep_ref = sc.run_place_episode(model, slots, pairs, cfg, policy, seed, p, q=q, frame_hook=reader, rule=rule)
        ident = R.episode_identity(ep, ep_ref, hook.qpos, reader.qpos)
        recs = hook.finalize()
        row = ep["pairs"][0]
        episodes.append({
            "seed": seed, "steps": ep["steps"], "failed": ep["failed"], "frames": len(recs),
            "identity_vs_read_only_rollout": ident,
            "rest_z_mirror": {"rest_z_m": hook.rest_z, "rest_step": hook.rest_step, "steps_checked": hook.lift_checked,
                              "max_abs_mismatch_m": hook.lift_mismatch_max,
                              "ok": hook.rest_z is not None and hook.lift_mismatch_max == 0.0},
            "lift_place_row": {k: row[k] for k in ("success", "first_failed_check", "lift_peak_m", "lift_hold_s",
                                                   "cube_floor_contact", "final_dz_m", "final_offset_xy_m",
                                                   "final_cube_tilt_rad", "final_robot_contact", "floor_contact_first")},
            "wall_s": {"rendered": round(t1 - t0, 1), "read_only": round(time.time() - t1, 1)}})
        if not ident["ok"]:
            problems.append(f"seed {seed}: rendered rollout differs from the read-only rollout {ident}")
        if not episodes[-1]["rest_z_mirror"]["ok"]:
            problems.append(f"seed {seed}: rest-height mirror mismatch {episodes[-1]['rest_z_mirror']}")
        if len(recs) != n_frames:
            problems.append(f"seed {seed}: {len(recs)} frames, {n_frames} expected (failed={ep['failed']})")
        for r in recs:
            if not r["c1_check"]["ok"]:
                problems.append(f"{r['id']}: C1 frame fails the C1 gates (cube geom in the C1 scene, or beyond the "
                                f"renderer's repeat noise of the cube-removed render) {r['c1_check']}")
            bad = {k: v for k, v in r["png_chunks"].items() if set(v) - {"IHDR", "IDAT", "IEND"}}
            if bad:
                problems.append(f"{r['id']}: PNG carries extra chunks {bad}")
        frames.extend(recs)
        print(json.dumps({"seed": seed, "frames": len(recs), "identity": ident["ok"],
                          "rest_mirror": episodes[-1]["rest_z_mirror"]["ok"],
                          "c1_ok": all(r["c1_check"]["ok"] for r in recs),
                          "c1_max_diff": max((r["c1_check"]["max_diff_vs_cube_removed"] for r in recs), default=None),
                          "labels_true": {b: sum(r["labels"][b] for r in recs) for b in E.BOOLEANS},
                          "cube_visible": sum(r["cube_visible"] for r in recs),
                          "lift_place_success": row["success"], "wall_s": episodes[-1]["wall_s"]}), flush=True)
    rd.close()
    counts = V.class_counts(frames)
    expected = len(seeds) * n_frames
    manifest = provenance(
        mode=args.mode, schema="er_obs1_frames_v1", seeds=seeds, episode_s=rule["episode_s"],
        frames_per_episode=n_frames, expected_frames=expected,
        image={"width": R.WIDTH, "height": R.HEIGHT, "format": "png", "channels": "RGB"},
        camera=R.CAMERA, render_flags=R.RENDER_FLAGS, gl=rd.gl,
        harness={"protocol": "lift_place", "crew": 2, "rule": rule, "params": sc.params_dict(p),
                 "place_params": sc.place_params_dict(q), "keyframes_left": sc.KEYFRAMES_LEFT,
                 "deploy": str(Path(args.deploy).resolve()), "deploy_sha256": sha256(args.deploy),
                 "checkpoint": str(cfg.policy_checkpoint_path),
                 "simulator": f"MuJoCo {mujoco.__version__}", "physics_dt": float(cfg.physics_dt),
                 "policy_dt": float(cfg.policy_dt), "frame_every_policy_steps": FRAME_STEPS},
        harness_check=hc, episodes=episodes, frames=frames,
        summary={"frames": len(frames), "expected_frames": expected, "class_counts": counts,
                 "classes_below_20": V.short_classes(counts),
                 "cube_visible_frames": sum(f["cube_visible"] for f in frames),
                 "body_z_disagreement_frames": sum(f["body_z_disagrees"] for f in frames),
                 "c1_ok_all": all(f["c1_check"]["ok"] for f in frames),
                 "c1_identical_frames": sum(f["c1_check"]["identical_to_cube_removed"] for f in frames),
                 "c1_max_diff_levels": max((f["c1_check"]["max_diff_vs_cube_removed"] for f in frames), default=None),
                 "c1_max_diff_px": max((f["c1_check"]["diff_px_vs_cube_removed"] for f in frames), default=None),
                 "removed_repeat_max_diff_levels": max((f["c1_check"]["removed_repeat_diff"]["max"] for f in frames),
                                                       default=None),
                 "removed_repeat_max_diff_px": max((f["c1_check"]["removed_repeat_diff"]["n_px"] for f in frames),
                                                   default=None),
                 "identity_all": all(e["identity_vs_read_only_rollout"]["ok"] for e in episodes),
                 "wall_s": round(time.time() - t_all, 1)},
        problems=problems, check="FRAMES_OK" if not problems and len(frames) == expected else "FRAMES_FAIL")
    write_json_new(args.out / "manifest.json", manifest)
    s = manifest["summary"]
    print(f"ER-OBS-1 FRAMES {args.mode}: {manifest['check']} | frames {s['frames']}/{expected} | class counts "
          + " ".join(f"{b}={c['true']}T/{c['false']}F" for b, c in counts.items())
          + f" | classes below 20: {s['classes_below_20'] or 'none'} | cube-visible {s['cube_visible_frames']}"
          + f" | body-z disagreements {s['body_z_disagreement_frames']} | C1 ok {s['c1_ok_all']} (exact-identical "
          + f"{s['c1_identical_frames']}/{s['frames']}, max diff {s['c1_max_diff_levels']} levels / "
          + f"{s['c1_max_diff_px']} px; repeat noise {s['removed_repeat_max_diff_levels']} / "
          + f"{s['removed_repeat_max_diff_px']} px)"
          + f" | identity {s['identity_all']} | GL {rd.gl.get('GL_RENDERER')} | {E.LABEL}", flush=True)
    for pr in problems:
        print("PROBLEM:", pr, flush=True)
    return 0 if manifest["check"] == "FRAMES_OK" else 2


# ------------------------------------------------------------------ manifest loading for the caller

def load_manifest(frames_dir: Path, *, need_run: bool) -> dict:
    path = frames_dir / "manifest.json"
    man = json.loads(path.read_text())
    if man.get("schema") != "er_obs1_frames_v1" or man.get("check") != "FRAMES_OK":
        raise SystemExit(f"ER-OBS-1: REFUSED ({path} is not a FRAMES_OK ER-OBS-1 manifest)")
    if need_run and (man.get("mode") != "run" or man.get("seeds") != list(E.SCORED_SEEDS)
                     or len(man["frames"]) != E.EXPECTED_RUN_FRAMES or man.get("episode_s") != E.EPISODE_S):
        raise SystemExit(f"ER-OBS-1: REFUSED ({path} is not the scored run: seeds 300-319, 20 s, 400 frames)")
    man["_path"], man["_sha256"] = str(path), sha256(path)
    return man


def refuse_short_classes(man: dict) -> None:
    """Paid modes: refuse before any request when a label class has fewer than 20 frames (the accuracy verdict would
    read INCOMPLETE whatever the answers). The manifest's summary.classes_below_20 must be empty and agree with the
    recount from its frames."""
    short = V.short_classes(V.class_counts(man["frames"]))
    summ = (man.get("summary") or {}).get("classes_below_20")
    if short or summ != short:
        raise SystemExit(f"ER-OBS-1: REFUSED ({man['_path']}: label classes below 20 frames: recount "
                         f"{short or 'none'}, summary.classes_below_20 {summ!r}; the accuracy verdict would be "
                         "INCOMPLETE whatever the answers, so no paid call is made)")


def frame_png(frames_dir: Path, rec: dict, variant: str) -> bytes:
    """The frame's PNG bytes, checked against the manifest sha256 and for metadata chunks (never sent if either
    fails)."""
    from bhl_robust.eval.er_obs.render import png_chunks
    name = "rgb" if variant == "main" else "c1"
    b = (frames_dir / rec["files"][name]).read_bytes()
    if hashlib.sha256(b).hexdigest() != rec["sha256"][name]:
        raise SystemExit(f"ER-OBS-1: REFUSED ({rec['id']}/{variant}: PNG sha256 differs from the manifest)")
    extra = set(png_chunks(b)) - {"IHDR", "IDAT", "IEND"}
    if extra:
        raise SystemExit(f"ER-OBS-1: REFUSED ({rec['id']}/{variant}: PNG carries chunks {sorted(extra)})")
    return b


def mask_of(frames_dir: Path, rec: dict) -> np.ndarray:
    from PIL import Image
    p = frames_dir / rec["files"]["cube_mask"]
    if sha256(p) != rec["sha256"]["cube_mask"]:
        raise SystemExit(f"ER-OBS-1: REFUSED ({rec['id']}: cube mask sha256 differs from the manifest)")
    return np.asarray(Image.open(p)) > 0


def _forbidden_roots(*dirs) -> list:
    return [REPO, MAIN_REPO, *[d for d in dirs if d is not None]]


def make_transport(args, man: dict, frames_dir: Path, pre_dir: Path | None, out_dir: Path):
    """(transport, secret, pacer): mock -> MockTransport on a virtual clock, network disabled, key never touched;
    run -> the hash, paid-tier and key guards (in that order, before the key is read), then HttpTransport paced in
    real time (client.CLOCK / client.SLEEP)."""
    if args.mode == "mock":
        C.disable_network()
        image_sha, recs = {}, []
        hw = [man["image"]["height"], man["image"]["width"]]
        for m in [man] + ([args._pre_man] if getattr(args, "_pre_man", None) else []):
            for rec in m["frames"]:
                r = dict(rec, image_hw=hw)
                image_sha[rec["sha256"]["rgb"]] = (r, "main")
                image_sha[rec["sha256"]["c1"]] = (r, "c1")
                recs.append(r)
        force = getattr(args, "_force", None) or {}
        vc = C.VirtualClock()
        return (C.MockTransport(recs, image_sha, seed=args.mock_seed, force=force, clock=vc), None,
                C.Pacer(clock=vc, sleep=vc.sleep))
    if args.expect_prompt_sha256 != C.PROMPT_SCHEMA_SHA256:
        raise SystemExit("ER-OBS-1: REFUSED (--expect-prompt-sha256 does not match the code's prompt+schema sha256 "
                         f"{C.PROMPT_SCHEMA_SHA256}; record it in the ledger first)")
    if args.expect_request_sha256 != C.REQUEST_SHA256:
        raise SystemExit("ER-OBS-1: REFUSED (--expect-request-sha256 does not match the code's request-template sha256 "
                         f"{C.REQUEST_SHA256}; record it in the ledger first)")
    if C.declared_tier() is None:
        raise SystemExit(f"ER-OBS-1: REFUSED (declare the key's tier: {C.PAID_TIER_ENV}=1 or {C.TIER_ENV}=paid for a "
                         f"PAID-tier key, {C.TIER_ENV}=free for a free-tier key (free-tier inputs are used to improve "
                         "Google's products), never both; the key must be API-restricted (unrestricted keys get HTTP "
                         "403); nothing was sent)")
    try:                                     # offline TLS check: requests verifies against certifi's CA bundle
        import certifi
        ca = Path(certifi.where())
        if not ca.is_file() or ca.stat().st_size == 0:
            raise OSError(str(ca))
    except (ImportError, OSError):
        raise SystemExit("ER-OBS-1: REFUSED (no CA bundle for TLS: certifi missing or empty)") from None
    try:
        kp = C.check_key_file(C.key_file_path(), _forbidden_roots(frames_dir, pre_dir, out_dir))
        key = C.read_key(kp)
    except C.KeyRefused as e:
        raise SystemExit(f"ER-OBS-1: REFUSED (API key file: {e})") from None
    return C.HttpTransport(key), key, C.Pacer()


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def owner_info() -> dict:
    return {"slurm_job_id": os.environ.get("SLURM_JOB_ID"), "host": socket.gethostname(), "pid": os.getpid(),
            "since": _now()}


def preflight_check(res: dict) -> dict:
    """The preflight's format check: which layer failed, if any. Passing needs HTTP 200, an output_text that passes
    the schema check, and a usage object (its thought-token counts are the evidence that thinking ran)."""
    f = res["final"]
    if f is None:
        return {"ok": False, "layer": "service unavailable (not a format problem)",
                "detail": f"{res['stop']}: only HTTP 429/503 or connection errors, no answer"}
    k, st = f.get("kind"), f.get("http_status")
    if k == "http_error":
        layer = ("request shape" if st in (400, 404, 405, 413, 415, 422)
                 else "key (unrestricted, wrong or not paid tier? the docs: unrestricted keys get 403)"
                 if st in (401, 403) else "service")
        return {"ok": False, "layer": layer, "detail": f"HTTP {st}: {f.get('error')}"}
    if k in ("bad_envelope", "no_answer"):
        return {"ok": False, "layer": "response shape", "detail": f.get("error")}
    if k in ("non_json", "schema_violation"):
        return {"ok": False, "layer": "answer format", "detail": f"{k}: {f.get('error') or f.get('text', '')[:200]}"}
    if k in ("timeout", "transport_error"):
        return {"ok": False, "layer": "transport", "detail": f"{k}: {f.get('error')}"}
    if k != "ok":
        return {"ok": False, "layer": "unknown", "detail": k}
    if not f.get("usage"):
        return {"ok": False, "layer": "response shape",
                "detail": "no usage object (no top-level field containing 'usage'): the thinking level cannot be shown"}
    return {"ok": True, "layer": None, "detail": None, "usage": f["usage"], "thought_tokens": f.get("thought_tokens")}


def preflight(args, transport, secret, pacer, pre_dir: Path) -> dict:
    """One call (under the call policy) on the first main frame of a SMOKE frames directory (throw-away seed); it must
    pass `preflight_check` before any scored or measured call. The raw response is kept (redacted, first 64 KB)."""
    pman = load_manifest(pre_dir, need_run=False)
    if pman.get("mode") != "smoke" or not set(pman["seeds"]) <= set(E.SMOKE_SEEDS):
        raise SystemExit(f"ER-OBS-1: REFUSED (preflight frames {pre_dir} are not a smoke run on seeds 900-901)")
    rec = pman["frames"][0]
    res = C.call_with_policy(transport, frame_png(pre_dir, rec, "main"), pacer, secret=secret,
                             raw_cap=C.RAW_CAP_PREFLIGHT)
    out = dict(res["final"] or {"kind": "no_final_outcome"})
    out.update(attempts=res["attempts"], stop=res["stop"], format_check=preflight_check(res), frame_id=rec["id"],
               frames_manifest=pman["_path"], frames_manifest_sha256=pman["_sha256"], at=_now())
    return out


def print_preflight(pf: dict) -> None:
    fc = pf["format_check"]
    print(f"ER-OBS-1 PREFLIGHT: {'PASS' if fc['ok'] else 'FAIL'}"
          + ("" if fc["ok"] else f" [{fc['layer']}: {fc['detail']}]")
          + f" | frame {pf['frame_id']} | kind {pf.get('kind')} | attempts {len(pf['attempts'])} | latency "
          + f"{pf.get('latency_s')} s | thought tokens {pf.get('thought_tokens') or 'not reported'}", flush=True)


def call_header(args, man: dict) -> dict:
    return {"model": C.MODEL, "endpoint": C.ENDPOINT, "prompt": C.PROMPT, "json_schema": C.SCHEMA,
            "prompt_text": C.TEXT, "prompt_schema_sha256": C.PROMPT_SCHEMA_SHA256, "request_sha256": C.REQUEST_SHA256,
            "request_template": C.request_template(), "call_policy": C.CALL_POLICY,
            "request_format_tested_live": False,
            "request_format_source": ("saved official ER 2 docs (2026-10-03): REST example in aidev_robotics-overview."
                                      "txt:211-235 and aidev_robotics-spatial.txt:44-64; answer field output_text"),
            "frames_manifest": man["_path"], "frames_manifest_sha256": man["_sha256"], "mode": args.mode,
            "tier": C.declared_tier() if args.mode == "run" else "mock"}


# ------------------------------------------------------------------ the per-frame cache (claims, attempts, outcomes)

def claim(path: Path, owner: dict) -> bool:
    """Claim a frame and variant: exclusive create (O_CREAT | O_EXCL). True iff this process now holds it."""
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as f:
        f.write(json.dumps(owner) + "\n")
    return True


def release(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def publish_once(path: Path, obj) -> None:
    """Publish a JSON file atomically and exclusively: the complete content under a temporary name, then a hard link
    to the final name (fails if it exists; never overwrites); an O_EXCL create where hard links are unsupported."""
    data = (json.dumps(obj, indent=1, allow_nan=False) + "\n").encode()
    tmp = path.with_name(f".{path.name}.tmp-{socket.gethostname()}-{os.getpid()}-{time.time_ns()}")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    try:
        os.link(tmp, path)
    except FileExistsError:
        raise FileExistsError(f"{path} exists; outcomes are final and never overwritten") from None
    except OSError:
        V.write_once(path, obj)
    finally:
        tmp.unlink(missing_ok=True)


def append_jsonl(path: Path, obj) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(obj, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def check_cached(path: Path, img_sha: str, mode: str) -> dict:
    old = json.loads(path.read_text())
    if (old.get("image_sha256") != img_sha or old.get("request_sha256") != C.REQUEST_SHA256
            or old.get("prompt_schema_sha256") != C.PROMPT_SCHEMA_SHA256 or old.get("mode") != mode):
        raise SystemExit(f"ER-OBS-1 CALLS: REFUSED ({path} belongs to a different frame or request)")
    return old


def _min_gap(records) -> dict:
    gaps = [a["gap_before_s"] for r in records if r is not None for a in (r.get("attempts") or [])
            if a.get("gap_before_s") is not None]
    return {"min_gap_before_s": min(gaps) if gaps else None, "requests_after_a_previous_one": len(gaps),
            "rule_min_gap_s": C.MIN_GAP_S}


def _thought_summary(records) -> dict:
    vals = sorted(r["thought_tokens"][0][1] for r in records
                  if r is not None and r.get("thought_tokens"))
    return {"records_with_a_thought_count": len(vals), "min": vals[0] if vals else None,
            "median": vals[len(vals) // 2] if vals else None, "max": vals[-1] if vals else None}


# ------------------------------------------------------------------ calls

MOCK_FORCED_MAIN = ("one_wrong", "http_500", "malformed_json", "http_429_then_ok", "schema_violation", "timeout",
                    "prose", "no_output", "fenced_json", "http_503_then_ok", "bad_body", "conn_error_then_ok",
                    "http_400")
MOCK_FORCED_C1 = ("c1_point_out_of_range", "malformed_json", "timeout", "http_429_then_ok")


def cmd_calls(args) -> int:
    run = args.mode == "run"
    man = load_manifest(args.frames, need_run=run)
    pre_dir = args.preflight_frames
    if run and pre_dir is None:
        raise SystemExit("ER-OBS-1: REFUSED (run mode needs --preflight-frames <a smoke frames directory>)")
    if run:
        refuse_short_classes(man)
    if not run and pre_dir is not None:
        args._pre_man = load_manifest(pre_dir, need_run=False)
    out = args.out
    calls_json = out / "calls.json"
    if calls_json.exists():
        raise SystemExit(f"ER-OBS-1 CALLS: REFUSED ({calls_json} exists; outputs are never overwritten)")
    if not run:
        # every mock kind is exercised on the first frames (deterministic), the rest follow the mock plan;
        # transport-level failures are separated by other kinds so the forced run never trips the stop rule
        fr = man["frames"]
        args._force = {(fr[i]["id"], "main"): k for i, k in enumerate(MOCK_FORCED_MAIN) if i < len(fr)}
        args._force.update({(fr[i]["id"], "c1"): k for i, k in enumerate(MOCK_FORCED_C1) if i < len(fr)})
        if pre_dir is not None:              # the mock preflight frame always gets a well-formed answer
            args._force.setdefault((args._pre_man["frames"][0]["id"], "main"), "correct")
    transport, secret, pacer = make_transport(args, man, args.frames, pre_dir, out)
    cache = out / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    head = call_header(args, man)
    owner = owner_info()
    pf = None
    if pre_dir is not None:
        pf = preflight(args, transport, secret, pacer, pre_dir)
        write_json_new(out / f"preflight-{time.strftime('%Y%m%dT%H%M%S')}-{time.time_ns()}.json",
                       {**head, **pf, "job": owner})
        print_preflight(pf)
        if not pf["format_check"]["ok"]:
            print(f"ER-OBS-1 CALLS: STOPPED BEFORE ANY SCORED CALL (preflight failed: {pf['format_check']['layer']}; "
                  "nothing was scored; the raw response is in the preflight JSON)", flush=True)
            return 5
        if not run:
            transport.attempts.clear()       # mock: the preflight frame's request counter starts afresh
    made = cached = 0
    consecutive = 0
    busy = []
    for rec in man["frames"]:
        for variant in E.VARIANTS:
            fid = f"{rec['id']}__{variant}"
            final_path = cache / f"{fid}.json"
            img_sha = rec["sha256"]["rgb" if variant == "main" else "c1"]
            if final_path.exists():
                check_cached(final_path, img_sha, args.mode)
                cached += 1
                continue
            claim_path = cache / f"{fid}.claim"
            if not claim(claim_path, owner):
                busy.append(f"{rec['id']}/{variant}")
                continue
            sending = False
            try:
                if final_path.exists():          # published by another job between the two checks
                    check_cached(final_path, img_sha, args.mode)
                    release(claim_path)
                    cached += 1
                    continue
                png = frame_png(args.frames, rec, variant)
                log_path = cache / f"{fid}.attempts.jsonl"
                prior = read_jsonl(log_path)
                sending = True
                res = C.call_with_policy(transport, png, pacer, secret=secret,
                                         on_attempt=lambda a, lp=log_path: append_jsonl(lp, {**a, "job": owner}))
                if res["final"] is None:
                    release(claim_path)          # no answer was ever received for this frame: a resume may re-send
                    print(f"ER-OBS-1 CALLS: STOPPED at {rec['id']}/{variant}: {res['stop']} (no answer, nothing scored "
                          f"for this frame; its attempts are in {log_path.name}; recorded outcomes stay final); "
                          "resubmit to resume", flush=True)
                    return 4
                out_rec = dict(res["final"])
                attempts = prior + [{**a, "job": owner} for a in res["attempts"]]
                out_rec.update(frame_id=rec["id"], variant=variant, image_sha256=img_sha, mode=args.mode,
                               request_sha256=C.REQUEST_SHA256, prompt_schema_sha256=C.PROMPT_SCHEMA_SHA256,
                               at=_now(), n_attempts=len(attempts), attempts=attempts, job=owner)
                publish_once(final_path, out_rec)
                release(claim_path)
            except BaseException:
                if not sending:
                    release(claim_path)          # nothing was sent for this frame
                raise                            # a request went out without a published outcome: the claim stays
            made += 1
            consecutive = consecutive + 1 if out_rec["kind"] in C.TRANSPORT_FAILURES else 0
            if consecutive >= C.ABORT_CONSECUTIVE:
                print(f"ER-OBS-1 CALLS: STOPPED after {consecutive} consecutive final transport-level failures "
                      f"(last: {out_rec['kind']} {out_rec.get('http_status')}); recorded outcomes stay final; "
                      f"resubmit to resume", flush=True)
                return 4
    records, orphans = [], []
    for rec in man["frames"]:
        for variant in E.VARIANTS:
            fid = f"{rec['id']}__{variant}"
            path = cache / f"{fid}.json"
            if path.exists():
                records.append(json.loads(path.read_text()))
            else:
                records.append(None)
                if (cache / f"{fid}.claim").exists():
                    orphans.append(f"{rec['id']}/{variant}")
    missing = sum(r is None for r in records)
    kinds = {}
    for r in records:
        if r is not None:
            kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    print(f"ER-OBS-1 CALLS {args.mode}: made {made}, cached {cached}, missing {missing}, kinds {kinds}", flush=True)
    if missing:
        if orphans:
            print(f"ER-OBS-1 CALLS: {len(orphans)} frame/variant(s) are claimed without an outcome (another calls job "
                  f"is running, or a job died mid-call; never re-sent automatically; see the launcher header): "
                  f"{orphans[:10]}", flush=True)
        print("ER-OBS-1 CALLS: INCOMPLETE (resubmit to resume; no calls.json, no verdict)", flush=True)
        return 3
    payload = provenance(**head, schema="er_obs1_calls_v2", preflight=pf, records=records, outcome_kinds=kinds,
                         pacing=_min_gap(records), thought_tokens_summary=_thought_summary(records),
                         re_sent_records=sum(1 for r in records if r.get("n_attempts", 1) > 1),
                         note=("MOCK: synthetic responses on a virtual clock, no network, no key; a pipeline check, "
                               "not a result" if not run else "scored calls"))
    try:
        write_json_new(calls_json, payload)
    except FileExistsError:
        print(f"ER-OBS-1 CALLS: {calls_json} was written by an overlapping calls job; this job stops (exit 3)",
              flush=True)
        return 3
    print(f"ER-OBS-1 CALLS {args.mode}: COMPLETE -> {calls_json} | pacing {payload['pacing']} | thought tokens "
          f"{payload['thought_tokens_summary']}", flush=True)
    return 0


# ------------------------------------------------------------------ latency

def cmd_latency(args) -> int:
    run = args.mode == "run"
    man = load_manifest(args.frames, need_run=run)
    pre_dir = args.preflight_frames
    if run and pre_dir is None:
        raise SystemExit("ER-OBS-1: REFUSED (latency mode needs --preflight-frames <a smoke frames directory>)")
    if run:
        refuse_short_classes(man)
        if C.declared_tier() == "free":
            raise SystemExit(f"ER-OBS-1 LATENCY: REFUSED ({C.TIER_ENV}=free: on the free tier 50 calls at a "
                             f"{C.MIN_GAP_S:.1f} s gap would measure the rate limit, not the model; the latency run "
                             "needs a paid key; nothing was sent)")
    out_json = args.out / "latency.json"
    if out_json.exists():
        raise SystemExit(f"ER-OBS-1 LATENCY: REFUSED ({out_json} exists; outputs are never overwritten)")
    frames = man["frames"]
    if run and len(frames) < E.LATENCY_CALLS:
        raise SystemExit("ER-OBS-1 LATENCY: REFUSED (fewer than 50 frames)")
    picks = [frames[i % len(frames)] for i in range(E.LATENCY_CALLS)]
    if not run:
        # two forced timeouts (+inf); every other call gets a synthetic wall time in [0.30, 1.50) s (virtual clock)
        args._force = {}
        if pre_dir is not None:              # the mock preflight frame always gets a well-formed answer
            args._pre_man = load_manifest(pre_dir, need_run=False)
            args._force[(args._pre_man["frames"][0]["id"], "main")] = "correct"
    transport, secret, pacer = make_transport(args, man, args.frames, pre_dir, args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    head = call_header(args, man)
    pf = None
    if pre_dir is not None:
        pf = preflight(args, transport, secret, pacer, pre_dir)
        write_json_new(args.out / f"preflight-{time.strftime('%Y%m%dT%H%M%S')}-{time.time_ns()}.json",
                       {**head, **pf, "job": owner_info()})
        print_preflight(pf)
        if not pf["format_check"]["ok"]:
            print(f"ER-OBS-1 LATENCY: STOPPED BEFORE ANY MEASURED CALL (preflight failed: "
                  f"{pf['format_check']['layer']})", flush=True)
            return 5
    records = []
    for i, rec in enumerate(picks):
        png = frame_png(args.frames, rec, "main")
        if not run:
            transport.force[(rec["id"], "main")] = "timeout" if i in (7, 31) else "correct"
            u = int(hashlib.sha256(f"lat:{args.mock_seed}:{i}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
            transport.next_latency = 0.30 + 1.20 * u
        gap = pacer.before_send()
        r = C.call_once(transport, png, secret=secret, clock=pacer.now)
        pacer.after()
        r.update(i=i, frame_id=rec["id"], gap_before_s=gap, at=_now())
        records.append(r)
        print(json.dumps({"i": i, "kind": r["kind"], "latency_s": r.get("latency_s"), "gap_before_s": gap}),
              flush=True)
    gaps = [r["gap_before_s"] for r in records if r.get("gap_before_s") is not None]
    payload = provenance(**head, schema="er_obs1_latency_v2", preflight=pf, records=records,
                         synchronous=True, re_sends=0,
                         pacing={"min_gap_before_s": min(gaps) if gaps else None, "rule_min_gap_s": C.MIN_GAP_S},
                         connection="requests.Session keep-alive" if run else "mock (virtual clock)",
                         note=("MOCK: synthetic latencies on a virtual clock and two forced timeouts; a pipeline "
                               "check, not a result" if not run else "measured calls"))
    write_json_new(out_json, payload)
    print(f"ER-OBS-1 LATENCY {args.mode}: {len(records)} calls -> {out_json}", flush=True)
    return 0


# ------------------------------------------------------------------ verdicts

def accuracy_inputs(frames_dir: Path, calls_path: Path) -> tuple:
    calls = json.loads(calls_path.read_text())
    man = load_manifest(frames_dir, need_run=calls.get("mode") == "run")
    if calls.get("frames_manifest_sha256") != man["_sha256"]:
        raise SystemExit("ER-OBS-1 VERDICT: REFUSED (calls.json was made on a different frames manifest)")
    if calls.get("prompt_schema_sha256") != C.PROMPT_SCHEMA_SHA256 or calls.get("request_sha256") != C.REQUEST_SHA256:
        raise SystemExit("ER-OBS-1 VERDICT: REFUSED (calls.json used a different prompt, schema or request)")
    outcomes = {}
    by_id = {f["id"]: f for f in man["frames"]}
    for r in calls["records"]:
        if r is None:
            continue
        f = by_id.get(r["frame_id"])
        img = f["sha256"]["rgb" if r["variant"] == "main" else "c1"] if f else None
        if f is None or r.get("image_sha256") != img:
            raise SystemExit(f"ER-OBS-1 VERDICT: REFUSED (record {r.get('frame_id')}/{r.get('variant')} does not "
                             "match the manifest)")
        outcomes[(r["frame_id"], r["variant"])] = r
    hits = {}
    for f in man["frames"]:
        r = outcomes.get((f["id"], "main"))
        if f["cube_visible"] and r is not None and r.get("kind") == "ok":
            hits[f["id"]] = V.point_hit(r["answer"]["cube_point"], mask_of(frames_dir, f))
    return man, calls, outcomes, hits


def cmd_verdict(args) -> int:
    if args.which == "accuracy":
        man, calls, outcomes, hits = accuracy_inputs(args.frames, args.calls)
        v = V.accuracy_verdict(man["frames"], outcomes, hits, expected_frames=man["expected_frames"])
        mock = calls.get("mode") != "run"
        inputs = {"frames_manifest": man["_path"], "frames_manifest_sha256": man["_sha256"],
                  "calls": str(args.calls), "calls_sha256": sha256(args.calls)}
    else:
        lat = json.loads(Path(args.latency).read_text())
        if lat.get("prompt_schema_sha256") != C.PROMPT_SCHEMA_SHA256 or lat.get("request_sha256") != C.REQUEST_SHA256:
            raise SystemExit("ER-OBS-1 VERDICT: REFUSED (latency.json used a different prompt, schema or request)")
        v = V.latency_verdict(lat["records"])
        mock = lat.get("mode") != "run"
        inputs = {"latency": str(args.latency), "latency_sha256": sha256(args.latency)}
    if mock:
        v["computed_verdict"] = v["verdict"]
        v["verdict"] = "PIPELINE_CHECK"
        v["verdict_label"] = f"PIPELINE_CHECK (MOCK, not a result; computed {v['computed_verdict']})"
    out = provenance(**v, which=args.which, mode="mock" if mock else "run", inputs=inputs,
                     prompt_schema_sha256=C.PROMPT_SCHEMA_SHA256, request_sha256=C.REQUEST_SHA256)
    try:
        write_json_new(args.out, out)
    except FileExistsError as e:
        raise SystemExit(f"ER-OBS-1 VERDICT: REFUSED ({e})") from None
    back = json.loads(Path(args.out).read_text())                       # the line is read back from the JSON
    print((V.accuracy_line(back) if args.which == "accuracy" else V.latency_line(back)) + f" | json={args.out}",
          flush=True)
    return 0


# ------------------------------------------------------------------ small commands

def cmd_hashes(args) -> int:
    print(json.dumps({"prompt_schema_sha256": C.PROMPT_SCHEMA_SHA256, "request_sha256": C.REQUEST_SHA256,
                      "model": C.MODEL, "endpoint": C.ENDPOINT, "prompt_text": C.TEXT,
                      "request_template": C.request_template()}, indent=1))
    return 0


def cmd_keycheck(args) -> int:
    try:
        kp = C.check_key_file(C.key_file_path(), _forbidden_roots(*args.forbid))
        C.read_key(kp)
    except C.KeyRefused as e:
        print(f"ER-OBS-1 KEY FILE: REFUSED ({e})", flush=True)
        return 1
    print(f"ER-OBS-1 KEY FILE: OK ({kp}: regular file, owner, mode 600, outside the repo and outputs; content not "
          "shown)", flush=True)
    return 0


def _read(path):
    if path is None:
        return None
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def _calls_smoke_checks(calls: dict, man: dict | None, va: dict | None, problems: list, checks: dict) -> None:
    """The mock calls smoke: every final kind occurred; each re-sendable kind was re-sent after the required wait and
    recorded; pacing held; the preflight recorded a passing format check with usage; failures are scored wrong on
    main and correct on C1 (recounted from the raw records); C1 booleans are scored without the point; the C1
    failure rate is reported."""
    kinds = calls.get("outcome_kinds", {})
    checks["outcome_kinds"] = kinds
    for k in ("ok", "non_json", "schema_violation", "no_answer", "bad_envelope", "http_error", "timeout"):
        if not kinds.get(k):
            problems.append(f"mock outcome kind {k} never occurred")
    recs = [r for r in calls.get("records", []) if r is not None]

    def resent(status, first_kind, min_wait):
        for r in recs:
            a = r.get("attempts") or []
            if (len(a) >= 2 and r.get("kind") == "ok" and (status is None or a[0].get("http_status") == status)
                    and (first_kind is None or a[0].get("kind") == first_kind)
                    and (a[1].get("gap_before_s") or 0.0) >= min_wait - 1e-6):
                return True
        return False
    for name, ok in (("HTTP 429 (Retry-After 2 s)", resent(429, None, 2.0)),
                     ("HTTP 503 (backoff 5 s)", resent(503, None, C.BACKOFF_S[0])),
                     ("connection error (backoff 5 s)", resent(None, "connection_error", C.BACKOFF_S[0]))):
        checks[f"re-sent after {name}"] = ok
        if not ok:
            problems.append(f"no recorded re-send after {name}")
    gap = _min_gap(recs)
    checks["pacing"] = gap
    if gap["min_gap_before_s"] is None or gap["min_gap_before_s"] < C.MIN_GAP_S - 1e-6:
        problems.append(f"pacing: min gap {gap['min_gap_before_s']} s < {C.MIN_GAP_S} s")
    pf = calls.get("preflight") or {}
    checks["preflight_format_check"] = pf.get("format_check")
    if not (pf.get("format_check") or {}).get("ok") or not pf.get("usage") or not pf.get("thought_tokens"):
        problems.append("the preflight did not record a passing format check with usage and thought tokens")
    if man is None or va is None:
        problems.append("no manifest or accuracy verdict to recount against")
        return
    lab = {f["id"]: f["labels"] for f in man["frames"]}
    for b in E.BOOLEANS:
        for variant, key in (("main", "main_balanced_accuracy"), ("c1", "c1_balanced_accuracy")):
            tp = tn = 0
            for r in recs:
                if r["variant"] != variant:
                    continue
                y = lab[r["frame_id"]][b]
                if variant == "main":                      # failed call -> wrong
                    p = r["answer"][b] if r["kind"] == "ok" else (not y)
                else:                                      # failed call -> correct; booleans without the point
                    p = r["booleans"][b] if isinstance(r.get("booleans"), dict) else y
                tp += bool(y and p)
                tn += bool((not y) and (not p))
            got = va[key][b]
            if (tp, tn) != (got["tp"], got["tn"]):
                problems.append(f"{variant} {b}: verdict counts {got['tp']}/{got['tn']} != recount {tp}/{tn}")
    c1 = [r for r in recs if r["variant"] == "c1"]
    c1_failed = [r for r in c1 if not isinstance(r.get("booleans"), dict)]
    c1_no_point = [r for r in c1 if r["kind"] != "ok" and isinstance(r.get("booleans"), dict)]
    checks["c1_failed"], checks["c1_scored_without_a_valid_point"] = len(c1_failed), len(c1_no_point)
    if not c1_failed:
        problems.append("no failed C1 call to check (scored correct)")
    if not c1_no_point:
        problems.append("no C1 answer with an invalid cube_point to check (booleans scored as given)")
    rate = (va.get("c1_failures") or {}).get("rate")
    if rate is None or not c1 or abs(rate - len(c1_failed) / len(c1)) > 1e-12:
        problems.append(f"C1 failure rate not reported or wrong ({rate})")
    failed = [r for r in recs if r["kind"] != "ok"]
    checks["failed_records"] = len(failed)
    if not failed:
        problems.append("no failed record to check")


def cmd_smoke_verdict(args) -> int:
    """frames: SMOKE_PASS iff every step exited 0 (unit tests included) and the frames manifest reads FRAMES_OK.
    calls: MOCK_SMOKE_PASS iff every step exited 0, calls.json is complete, every mock outcome kind occurred, the
    call policy held (pacing, re-sends recorded), failures were scored wrong on main and correct on C1, and both
    verdict JSONs are PIPELINE_CHECK."""
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"ER-OBS-1 SMOKE VERDICT: REFUSED ({out} exists)")
    status = _read(args.step_status)
    problems = []
    if not isinstance(status, dict) or not status:
        problems.append("no step_status")
    else:
        for k, val in status.items():
            if type(val) is not int or val != 0:
                problems.append(f"step {k} exit {val}")
        if "pytest" not in status:
            problems.append("unit tests not recorded")
    checks = {}
    if args.kind == "frames":
        man = _read(Path(args.frames) / "manifest.json")
        checks["manifest_check"] = None if man is None else man.get("check")
        if checks["manifest_check"] != "FRAMES_OK":
            problems.append(f"frames manifest reads {checks['manifest_check']}")
        if man is not None:
            checks["summary"] = man.get("summary")
            checks["gl"] = man.get("gl")
        verdict = "SMOKE_PASS" if not problems else "SMOKE_FAIL"
    else:
        calls = _read(Path(args.calls))
        va, vl = _read(args.verdict_accuracy), _read(args.verdict_latency)
        if calls is None:
            problems.append("no calls.json")
        else:
            _calls_smoke_checks(calls, _read(Path(args.frames) / "manifest.json"), va, problems, checks)
        for name, v in (("accuracy", va), ("latency", vl)):
            if v is None or v.get("verdict") != "PIPELINE_CHECK" or v.get("mode") != "mock":
                problems.append(f"{name} verdict JSON missing or not a mock PIPELINE_CHECK")
            else:
                checks[f"{name}_computed"] = v.get("computed_verdict")
        if vl is not None and (vl.get("failed_calls") != 2 or vl.get("calls") != E.LATENCY_CALLS):
            problems.append("latency mock: expected 50 calls with exactly the 2 forced timeouts")
        verdict = "MOCK_SMOKE_PASS" if not problems else "MOCK_SMOKE_FAIL"
    v = provenance(kind=args.kind, verdict=verdict, step_status=status, problems=problems, checks=checks,
                   note="pipeline check on throw-away seeds; never a result")
    write_json_new(out, v)
    print(f"ER-OBS-1 SMOKE ({args.kind}): {verdict}" + (f" | problems: {problems}" if problems else "")
          + f" | {out}", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("frames")
    f.add_argument("--mode", choices=("smoke", "run"), required=True)
    f.add_argument("--deploy", type=Path, required=True)
    f.add_argument("--upstream", type=Path, required=True)
    f.add_argument("--cache-dir", type=Path, required=True)
    f.add_argument("--out", type=Path, required=True)
    f.add_argument("--seeds", required=True)
    f.add_argument("--seconds", type=float, default=None)
    for name in ("calls", "latency"):
        c = sub.add_parser(name)
        c.add_argument("--mode", choices=("mock", "run"), required=True)
        c.add_argument("--frames", type=Path, required=True)
        c.add_argument("--out", type=Path, required=True)
        c.add_argument("--preflight-frames", type=Path, default=None)
        c.add_argument("--expect-prompt-sha256", default=None)
        c.add_argument("--expect-request-sha256", default=None)
        c.add_argument("--mock-seed", type=int, default=0)
    v = sub.add_parser("verdict")
    v.add_argument("which", choices=("accuracy", "latency"))
    v.add_argument("--frames", type=Path)
    v.add_argument("--calls", type=Path)
    v.add_argument("--latency", type=Path)
    v.add_argument("--out", type=Path, required=True)
    sub.add_parser("hashes")
    k = sub.add_parser("keycheck")
    k.add_argument("--forbid", type=Path, nargs="*", default=[])
    s = sub.add_parser("smoke-verdict")
    s.add_argument("kind", choices=("frames", "calls"))
    s.add_argument("--step-status", type=Path, required=True)
    s.add_argument("--frames", type=Path, required=True)
    s.add_argument("--calls", type=Path)
    s.add_argument("--verdict-accuracy", type=Path)
    s.add_argument("--verdict-latency", type=Path)
    s.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.cmd == "verdict":
        need = ("frames", "calls") if args.which == "accuracy" else ("latency",)
        if any(getattr(args, n) is None for n in need):
            ap.error(f"verdict {args.which} needs --" + " --".join(need))
    return {"frames": cmd_frames, "calls": cmd_calls, "latency": cmd_latency, "verdict": cmd_verdict,
            "hashes": cmd_hashes, "keycheck": cmd_keycheck, "smoke-verdict": cmd_smoke_verdict}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
