#!/usr/bin/env python3
"""Waiter phase 2: record SCRIPTED-expert demonstrations in the contract format (docs/WAITER_INTERFACES.md).

One compressed `.npz` per episode: image (T,224,224,3) uint8 (the SIMULATED head camera), state (T,30) f32,
action (T,15) f32 (the action the env APPLIED after observing step t, i.e. after its clipping), instruction str,
seed int, success bool, stage (T,) int8, plus template_id str and instruction_id int. `manifest.json` lists every
episode (seed, file, T, success, stage reached, template, sha256), failures included and flagged; phase 3 trains on
successes only by default. Also writes `templates.json` (the declared instruction templates, phase 3's format).

Seed ranges are guarded: demonstrations 3000-3999 (`--range demo`), or development 1000-1999 (`--range dev`). An
existing episode file is never overwritten (a rerun skips it), so a killed job resumes.

usage: record_demos.py --deploy <deploy.yaml> --upstream <BHL> --cache-dir <dir> --out <dir> --seeds 3000-3049
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE))

RANGES = {"demo": (3000, 3999), "dev": (1000, 1999)}


def parse_seeds(text):
    out = []
    for part in text.split(","):
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b or a) + 1))
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def record(env, expert_cls, params, seed: int, out: Path) -> dict:
    from bhl_robust.eval.waiter_env import STAGES
    obs = env.reset(seed)
    ex = expert_cls(env, params)
    ex.reset()
    imgs, states, acts, stages = [], [], [], []
    done, info = False, {}
    while not done:
        a = ex.act()
        imgs.append(obs["image"])
        states.append(obs["state"])
        obs, _, done, info = env.step(a)
        acts.append(info["applied_action"])
        stages.append(info["stage"])
    path = out / f"ep_{seed}.npz"
    tmp = out / f".ep_{seed}.tmp.npz"
    np.savez_compressed(tmp, image=np.stack(imgs).astype(np.uint8), state=np.stack(states).astype(np.float32),
                        action=np.stack(acts).astype(np.float32), instruction=np.array(env.instruction),
                        seed=np.array(seed, dtype=np.int64), success=np.array(bool(info["success"])),
                        stage=np.asarray(stages, dtype=np.int8), template_id=np.array(env.template_id),
                        instruction_id=np.array(int(env.template_id[1:]), dtype=np.int64))
    tmp.rename(path)
    return {"seed": seed, "file": path.name, "T": len(acts), "success": bool(info["success"]),
            "fell": bool(info["fell"]), "stage": int(info["stage"]), "stage_name": STAGES[int(info["stage"])],
            "t": float(info["t"]), "instruction": env.instruction, "template_id": env.template_id,
            "target_colour": env.layout.target_colour, "target_side": env.layout.target_side,
            "expert_notes": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in ex.notes.items()},
            "sha256": sha256(path)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deploy", required=True, type=Path)
    ap.add_argument("--upstream", required=True, type=Path)
    ap.add_argument("--cache-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--range", choices=sorted(RANGES), default="demo")
    ap.add_argument("--no-render", action="store_true", help="zero images (plumbing tests only)")
    ap.add_argument("--params", default="{}", help="JSON overrides of ExpertParams")
    a = ap.parse_args(argv)
    seeds = parse_seeds(a.seeds)
    lo, hi = RANGES[a.range]
    bad = [s for s in seeds if not lo <= s <= hi]
    if bad:
        raise SystemExit(f"RECORD: REFUSED seeds outside the {a.range} range {lo}-{hi}: {bad[:5]}")
    from bhl_robust.eval.waiter_env import SLOTS, TEMPLATES, WaiterEnv
    from expert import ExpertParams, WaiterExpert
    a.out.mkdir(parents=True, exist_ok=True)
    tj = a.out / "templates.json"
    if not tj.exists():
        tj.write_text(json.dumps({"templates": TEMPLATES, "slots": SLOTS, "held_out": []}, indent=1))
    params = ExpertParams(**json.loads(a.params))
    env = WaiterEnv(a.deploy, a.upstream, a.cache_dir, render=not a.no_render)
    rows = []
    for s in seeds:
        if (a.out / f"ep_{s}.npz").exists():
            print(f"RECORD: seed {s} exists, skipped", flush=True)
            continue
        t0 = time.time()
        r = record(env, WaiterExpert, params, s, a.out)
        r["wall_s"] = round(time.time() - t0, 1)
        rows.append(r)
        (a.out / f"ep_{s}.json").write_text(json.dumps(r, indent=1))
        print(f"RECORD: seed {s} success {r['success']} stage {r['stage_name']} T {r['T']} t {r['t']:.1f} "
              f"wall {r['wall_s']} s", flush=True)
    env.close()
    # the manifest is rebuilt from every per-episode JSON in the directory (several jobs may share it)
    eps = sorted((json.loads(p.read_text()) for p in a.out.glob("ep_*.json")), key=lambda r: r["seed"])
    man = {"format": "docs/WAITER_INTERFACES.md npz", "labels": ["SCRIPTED expert", "ORACLE poses",
           "SIMULATED head camera", "LEARNED legs (WBC)"], "deploy": str(a.deploy), "n": len(eps),
           "n_success": sum(e["success"] for e in eps), "episodes": eps}
    tmp = a.out / ".manifest.tmp.json"
    tmp.write_text(json.dumps(man, indent=1))
    tmp.rename(a.out / "manifest.json")
    print(f"RECORD: {len(rows)} new episodes; manifest {man['n_success']}/{man['n']} successful -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
