#!/usr/bin/env python3
"""Write SYNTHETIC contract-format episodes (+ manifest.json, templates.json) for phase 3 plumbing tests.

Modes:
  expert  roll the SYNTHETIC stub env (vla_stub_env.py) with its scripted stub expert and record exactly what the
          contract says phase 2 records: image, state, the action applied after observing step t, instruction,
          seed, success, stage. Gives a learnable (if trivial) signal.
  random  random images/state/actions of the right shapes and dtypes (pure shape test).

Every file carries ``synthetic=True``; seeds are made-up ids (default 900000+) outside every contract range.
Nothing produced here is waiter data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vla_stub_env import SYNTHETIC_TEMPLATES, StubExpert, StubWaiterEnv, templates_json  # noqa: E402
from waiter_vla_data import ACTION_DIM, IMAGE_SHAPE, STATE_DIM, seed_range_name  # noqa: E402


def record_expert(seed: int, template_id: str, noise: float, max_steps: int) -> dict:
    env = StubWaiterEnv(render=True)
    obs = env.reset(seed, instruction_id=template_id)
    expert = StubExpert(env, noise=noise, seed=seed)
    images, states, actions, stages = [], [], [], []
    info = {"success": False, "stage": 0}
    for _ in range(max_steps):
        a = expert.act()
        images.append(obs["image"])
        states.append(obs["state"])
        actions.append(a)
        stages.append(env.stage)  # stage at observation time t
        obs, _, done, info = env.step(a)
        if done:
            break
    return {"image": np.stack(images), "state": np.stack(states).astype(np.float32),
            "action": np.stack(actions).astype(np.float32), "stage": np.asarray(stages, dtype=np.int8),
            "instruction": env.instruction, "success": bool(info["success"])}


def record_random(seed: int, template_id: str, length: int) -> dict:
    rng = np.random.default_rng(seed)
    env = StubWaiterEnv(render=False)
    env.reset(seed, instruction_id=template_id)
    action = rng.standard_normal((length, ACTION_DIM)).astype(np.float32) * 0.3
    action[:, :3] = np.clip(action[:, :3], -0.3, 0.3)
    action[:, 13:15] = rng.uniform(0, 1, size=(length, 2))
    return {"image": rng.integers(0, 256, size=(length, *IMAGE_SHAPE), dtype=np.uint8),
            "state": rng.standard_normal((length, STATE_DIM)).astype(np.float32),
            "action": action, "stage": rng.integers(0, 4, size=length).astype(np.int8),
            "instruction": env.instruction, "success": True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=16)
    p.add_argument("--first-seed", type=int, default=900000)
    p.add_argument("--mode", choices=("expert", "random"), default="expert")
    p.add_argument("--noise", type=float, default=0.02, help="expert arm-target noise (rad)")
    p.add_argument("--random-length", type=int, default=40)
    p.add_argument("--max-steps", type=int, default=750)
    p.add_argument("--no-compress", action="store_true")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "manifest.json").exists():
        raise FileExistsError(f"{args.out}/manifest.json exists; refusing to overwrite")
    keys = sorted(SYNTHETIC_TEMPLATES)
    rows = []
    for i in range(args.episodes):
        seed = args.first_seed + i
        rng_name = seed_range_name(seed)
        if rng_name not in (None, "dev"):
            raise SystemExit(f"synthetic seed {seed} falls in contract range {rng_name}; pick made-up ids")
        template_id = keys[i % len(keys)]
        ep = (record_expert(seed, template_id, args.noise, args.max_steps) if args.mode == "expert"
              else record_random(seed, template_id, args.random_length))
        name = f"synthetic_{args.mode}_{seed}.npz"
        path = args.out / name
        save = np.savez if args.no_compress else np.savez_compressed
        tmp = args.out / f".{name}.tmp.npz"
        save(tmp, image=ep["image"], state=ep["state"], action=ep["action"], stage=ep["stage"],
             instruction=np.asarray(ep["instruction"]), seed=np.asarray(seed, dtype=np.int64),
             success=np.asarray(ep["success"]), synthetic=np.asarray(True), template_id=np.asarray(template_id))
        os.replace(tmp, path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append({"file": name, "seed": seed, "success": ep["success"], "sha256": digest,
                     "frames": int(ep["image"].shape[0]), "instruction": ep["instruction"],
                     "template_id": template_id, "synthetic": True})
        print(f"[synthetic] {name} T={ep['image'].shape[0]} success={ep['success']} template={template_id}", flush=True)
    (args.out / "templates.json").write_text(json.dumps(templates_json(), indent=2) + "\n")
    (args.out / "manifest.json").write_text(json.dumps({"synthetic": True, "mode": args.mode,
                                                         "episodes": rows}, indent=2) + "\n")
    n_ok = sum(r["success"] for r in rows)
    print(f"[synthetic] wrote {len(rows)} episodes ({n_ok} successful) to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
