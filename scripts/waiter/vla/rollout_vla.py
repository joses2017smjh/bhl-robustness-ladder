#!/usr/bin/env python3
"""Closed-loop evaluation of a fine-tuned waiter SmolVLA in a contract environment.

The policy sees exactly what the contract gives a VLA (head RGB, 30-D state, instruction; never ``oracle()``) and
emits 15-D actions at 12.5 Hz with action chunking: one inference returns ``chunk_size`` actions, the first
``n_action_steps`` are executed open-loop, then it re-plans from a fresh observation. Several envs run in lockstep
and every env that needs a new chunk on the same tick is served by ONE batched inference call.

Success is the environment's own ``info["success"]`` (the frozen contract definition implemented by phase 2:
tumbler upright on the tray, tray in the target zone, grippers open, robot upright, held 1.0 s); this script only
counts it, it never re-defines it. Episodes end on success, fall, the env's time limit (90 s since the 2026-10-09
contract amendment; 60 s before), or ``--max-steps``. The waiter env runs with its defaults, so the stance lock (a
declared simulation fixture) holds the base whenever the commanded base velocity is ~0; ``--free-base`` turns it off.

Seed hygiene: ``--purpose dev`` only accepts the dev range 1000-1999 (stub env: made-up ids outside every contract
range). ``--purpose scored`` only accepts 4000-4099, needs ``--gate`` (frozen gate file, sha256 recorded) and never
overwrites an existing result.

Envs run in-process (``--workers 0``) or one subprocess per env (``vla_env_worker.py``), optionally under another
interpreter (``--env-python``: the MuJoCo venv) since venv-lehome-train has no MuJoCo.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vla_env_worker as env_worker  # noqa: E402
import waiter_vla_data as wd  # noqa: E402

INFO_KEYS = ("success", "fell", "stage", "tumbler_on_tray", "tray_on_target", "t")


# ---------------------------------------------------------------------------------------------------------------
# Environment handles
# ---------------------------------------------------------------------------------------------------------------
class InProcessEnv:
    def __init__(self, kind, kwargs):
        self.env = env_worker.make_env(kind, kwargs)
        self.synthetic = bool(getattr(self.env, "synthetic", False))
        self._pending = None

    def reset(self, seed, instruction_id):
        return self.env.reset(seed, instruction_id=instruction_id)

    def step_async(self, action):
        self._pending = self.env.step(action)

    def step_wait(self):
        out, self._pending = self._pending, None
        return out

    def close(self):
        if hasattr(self.env, "close"):
            self.env.close()


class SubprocessEnv:
    def __init__(self, kind, kwargs, python, extra_env):
        env = dict(os.environ)
        env.update(extra_env)
        self.proc = subprocess.Popen([python, str(HERE / "vla_env_worker.py")], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, env=env)
        self._call("make", kind, kwargs)
        self.synthetic = bool(self._last["synthetic"])

    def _send(self, *msg):
        env_worker.send(self.proc.stdin, msg)

    def _wait(self):
        status, payload = env_worker.recv(self.proc.stdout)
        if status != "ok":
            raise RuntimeError(f"env worker failed:\n{payload}")
        self._last = payload
        return payload

    def _call(self, *msg):
        self._send(*msg)
        return self._wait()

    def reset(self, seed, instruction_id):
        return self._call("reset", seed, instruction_id)

    def step_async(self, action):
        self._send("step", np.asarray(action, dtype=np.float32))

    def step_wait(self):
        return self._wait()

    def close(self):
        try:
            self._call("close")
        except Exception:
            pass
        self.proc.wait(timeout=30)


# ---------------------------------------------------------------------------------------------------------------
def check_obs(obs) -> None:
    img, state = obs["image"], obs["state"]
    if not (isinstance(img, np.ndarray) and img.shape == wd.IMAGE_SHAPE and img.dtype == np.uint8):
        raise RuntimeError(f"contract violation: image {getattr(img, 'shape', None)} {getattr(img, 'dtype', None)}")
    state = np.asarray(state)
    if state.shape != (wd.STATE_DIM,) or not np.isfinite(state).all():
        raise RuntimeError(f"contract violation: state shape {state.shape} or non-finite")
    if not isinstance(obs["instruction"], str) or not obs["instruction"].strip():
        raise RuntimeError("contract violation: empty instruction")


def clip_action(a: np.ndarray) -> tuple[np.ndarray, bool]:
    """Contract ranges the env would apply anyway: base velocity limits, gripper closure in [0, 1]."""
    out = np.asarray(a, dtype=np.float32).copy()
    out[:3] = np.clip(out[:3], -wd.BASE_LIMITS, wd.BASE_LIMITS)
    out[13:15] = np.clip(out[13:15], 0.0, 1.0)
    return out, bool(np.any(out != a))


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def rate(rows):
    k = sum(bool(r["success"]) for r in rows)
    return {"episodes": len(rows), "successes": k, "success_rate": (k / len(rows)) if rows else None,
            "wilson95": wilson(k, len(rows)), "falls": sum(bool(r["fell"]) for r in rows)}


def percentiles(xs):
    if not xs:
        return None
    a = np.asarray(xs) * 1e3
    return {"n": len(xs), "mean_ms": float(a.mean()), "p50_ms": float(np.percentile(a, 50)),
            "p95_ms": float(np.percentile(a, 95)), "max_ms": float(a.max())}


def atomic_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def parse_instruction_ids(text):
    if not text:
        return [None]
    out = []
    for v in text.split(","):
        v = v.strip()
        out.append(int(v) if v.lstrip("-").isdigit() else v)
    return out


def check_seeds(seeds, purpose, env_kind):
    for s in seeds:
        name = wd.seed_range_name(s)
        if purpose == "dev":
            if env_kind == "stub":
                if name not in (None, "dev"):
                    raise SystemExit(f"stub (synthetic) rollouts must use made-up or dev seeds; {s} is {name}")
            elif name != "dev":
                raise SystemExit(f"--purpose dev only accepts seeds 1000-1999; {s} is {name}")
        elif purpose == "scored":
            if env_kind != "waiter" or name != "phase3_scored":
                raise SystemExit(f"--purpose scored only accepts the waiter env and seeds 4000-4099; {s} is {name}")


# ---------------------------------------------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True, help="checkpoint dir or best.json/latest.json")
    p.add_argument("--env", choices=("stub", "waiter"), required=True)
    p.add_argument("--deploy", help="waiter env: stamped deploy.yaml")
    p.add_argument("--upstream", help="waiter env: upstream repo path")
    p.add_argument("--cache-dir", help="waiter env: MJCF cache dir")
    p.add_argument("--fixed-base", action="store_true")
    p.add_argument("--free-base", action="store_true", help="waiter env: stance_lock=False (no stance lock)")
    p.add_argument("--seeds", required=True, help="e.g. 1000-1019 or 910000-910007 (stub)")
    p.add_argument("--instruction-ids", default="", help="comma list passed to reset(); each seed x each id")
    p.add_argument("--templates", type=Path, help="templates.json to label template / held-out per episode")
    p.add_argument("--counterfactual-slot", help="CONTROL: give the policy the instruction with this slot swapped "
                   "(e.g. side: left->right) while the env scores the true layout; needs --templates")
    p.add_argument("--purpose", choices=("dev", "scored"), default="dev")
    p.add_argument("--gate", type=Path, help="frozen gate file (required for --purpose scored)")
    p.add_argument("--out", type=Path, required=True, help="result JSON")
    p.add_argument("--n-action-steps", type=int, help="override the checkpoint's executed steps per chunk")
    p.add_argument("--flow-steps", type=int, help="override SmolVLA denoising steps (default from checkpoint: 10)")
    p.add_argument("--batch-envs", type=int, default=4, help="envs in lockstep (= max inference batch)")
    p.add_argument("--workers", type=int, default=1, help="1 = one subprocess per env; 0 = in-process")
    p.add_argument("--env-python", default=sys.executable, help="interpreter for env workers (MuJoCo venv)")
    p.add_argument("--max-steps", type=int, default=1135, help="safety cap; the env's own 90 s limit is 1125 steps")
    p.add_argument("--noise-seed", type=int, default=0)
    p.add_argument("--amp", choices=("none", "bf16"), default="none")
    p.add_argument("--device", default="cuda")
    args = p.parse_args(argv)

    seeds = sorted(wd.parse_seed_list(args.seeds))
    check_seeds(seeds, args.purpose, args.env)
    gate = None
    if args.purpose == "scored":
        if args.gate is None or not args.gate.is_file():
            raise SystemExit("--purpose scored requires an existing --gate file (frozen before evaluation)")
        gate = {"path": str(args.gate.resolve()), "sha256": hashlib.sha256(args.gate.read_bytes()).hexdigest()}
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    templates = wd.TemplateSet.load(args.templates) if args.templates else None
    if args.counterfactual_slot and templates is None:
        raise SystemExit("--counterfactual-slot needs --templates")
    jobs = [(s, iid) for s in seeds for iid in parse_instruction_ids(args.instruction_ids)]

    import torch

    from smolvla_io import load_finetuned, to_batch

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda without a CUDA allocation")
        torch.ones(1, device="cuda").add_(1)
        torch.cuda.synchronize()
    policy, pre, post, cfg, meta = load_finetuned(args.checkpoint, device=args.device,
                                                  n_action_steps=args.n_action_steps, num_steps=args.flow_steps)
    policy.eval()
    k_exec, chunk = cfg.n_action_steps, cfg.chunk_size

    env_kwargs = {"render": True, "fixed_base": args.fixed_base}
    if args.env == "waiter":
        env_kwargs["stance_lock"] = not args.free_base
    if args.env == "waiter":
        env_kwargs.update({"deploy": args.deploy, "upstream": args.upstream, "cache_dir": args.cache_dir})
    n_slots = max(1, min(args.batch_envs, len(jobs)))
    t0 = time.monotonic()
    if args.workers == 0:
        envs = [InProcessEnv(args.env, env_kwargs) for _ in range(n_slots)]
    else:
        extra = {"PYTHONPATH": os.environ.get("PYTHONPATH", "")}
        envs = [SubprocessEnv(args.env, env_kwargs, args.env_python, extra) for _ in range(n_slots)]
    env_start_s = time.monotonic() - t0
    synthetic = args.env == "stub" or any(e.synthetic for e in envs) or bool(meta.get("synthetic"))

    result = {
        "synthetic": synthetic, "env": args.env, "purpose": args.purpose, "gate": gate,
        "checkpoint": meta.get("checkpoint"), "checkpoint_step": meta.get("step"),
        "checkpoint_data_hash": meta.get("data_hash"), "checkpoint_trained_on_synthetic": meta.get("synthetic"),
        "chunk_size": chunk, "n_action_steps": k_exec, "flow_steps": cfg.num_steps, "control_hz": wd.CONTROL_HZ,
        "seeds": seeds, "instruction_ids": parse_instruction_ids(args.instruction_ids),
        "batch_envs": n_slots, "workers": args.workers, "env_python": args.env_python,
        "counterfactual_slot": args.counterfactual_slot,
        "stance_lock": env_kwargs.get("stance_lock"), "fixed_base": bool(args.fixed_base),
        "host": socket.gethostname(), "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "gpu": torch.cuda.get_device_name() if args.device == "cuda" else "cpu", "amp": args.amp,
        "success_source": "env info['success'] (frozen contract definition implemented by the env)",
        "episodes": [], "completed": False,
    }
    if synthetic:
        result["label"] = "SYNTHETIC plumbing run: stub env and/or synthetic training data; not a waiter result"
    if args.counterfactual_slot:
        result["control"] = (f"COUNTERFACTUAL: the policy was told the wrong '{args.counterfactual_slot}'; success "
                             "here means the policy ignored the instruction")

    def flow_noise(seed, iid, replan):
        tag = hashlib.sha256(f"{args.noise_seed}|{seed}|{iid}|{replan}".encode()).digest()[:8]
        g = torch.Generator().manual_seed(int.from_bytes(tag, "little") & ((1 << 63) - 1))
        return torch.randn(chunk, cfg.max_action_dim, generator=g)

    infer_s, infer_batch, env_step_s = [], [], []

    def infer(slots_needing):
        images = np.stack([s["obs"]["image"] for s in slots_needing])
        states = np.stack([np.asarray(s["obs"]["state"], dtype=np.float32) for s in slots_needing])
        tasks = [s["policy_instruction"] for s in slots_needing]
        noise = torch.stack([flow_noise(s["seed"], s["iid"], s["replans"]) for s in slots_needing]).to(args.device)
        if args.device == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        with torch.no_grad():
            proc = pre(to_batch(images, states, tasks, args.device))
            if args.amp == "bf16":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    actions = policy.predict_action_chunk(proc, noise=noise)
            else:
                actions = policy.predict_action_chunk(proc, noise=noise)
            actions = post(actions.float())
        if args.device == "cuda":
            torch.cuda.synchronize()
        infer_s.append(time.perf_counter() - t)
        infer_batch.append(len(slots_needing))
        actions = actions.cpu().numpy()
        if actions.shape != (len(slots_needing), chunk, wd.ACTION_DIM) or not np.isfinite(actions).all():
            raise RuntimeError(f"policy produced bad actions {actions.shape}")
        for s, a in zip(slots_needing, actions):
            s["queue"] = list(a[:k_exec])
            s["replans"] += 1

    def start(slot_env, job):
        seed, iid = job
        obs = slot_env.reset(seed, iid)
        check_obs(obs)
        template = held = None
        if templates is not None:
            template = templates.match(obs["instruction"])
            held = template in templates.held_out
        policy_instruction = obs["instruction"]
        if args.counterfactual_slot:
            policy_instruction = templates.counterfactual(obs["instruction"], args.counterfactual_slot)
        return {"env": slot_env, "seed": seed, "iid": iid, "obs": obs, "queue": [], "replans": 0, "steps": 0,
                "policy_instruction": policy_instruction,
                "ret": 0.0, "max_stage": 0, "clipped": 0, "stale": 0, "t_wall": time.monotonic(),
                "instruction": obs["instruction"], "template": template, "held_out_template": held,
                "reward_mismatch": 0}

    def finish(s, info, termination):
        row = {"seed": s["seed"], "instruction_id": s["iid"], "instruction": s["instruction"],
               "policy_instruction": s["policy_instruction"],
               "template_id": s["template"], "held_out_template": s["held_out_template"],
               "success": bool(info.get("success", False)), "fell": bool(info.get("fell", False)),
               "termination": termination, "steps": s["steps"], "sim_t": float(info.get("t", s["steps"] * wd.CONTROL_DT)),
               "final_stage": int(info.get("stage", -1)), "max_stage": s["max_stage"], "return": s["ret"],
               "replans": s["replans"], "clipped_action_frac": s["clipped"] / max(1, s["steps"]),
               "stale_frames": s["stale"], "reward_success_mismatch_steps": s["reward_mismatch"],
               "final_info": {k: info.get(k) for k in INFO_KEYS if k in info},
               "wall_s": time.monotonic() - s["t_wall"]}
        if s["steps"] > 5 and s["stale"] >= s["steps"] - 1:
            row["render_suspect"] = True  # the head camera never changed: likely a swallowed render failure
        result["episodes"].append(row)
        done_n = len(result["episodes"])
        print(f"[rollout] {done_n}/{len(jobs)} seed={row['seed']} iid={row['instruction_id']} success={row['success']} "
              f"term={termination} steps={row['steps']} stage<={row['max_stage']}", flush=True)
        write(final=False)

    def write(final):
        rows = result["episodes"]
        summary = {"all": rate(rows)}
        if templates is not None:
            summary["seen_templates"] = rate([r for r in rows if r["held_out_template"] is False])
            summary["held_out_templates"] = rate([r for r in rows if r["held_out_template"] is True])
        summary["terminations"] = {k: sum(r["termination"] == k for r in rows)
                                   for k in ("success", "fell", "env_done", "max_steps")}
        summary["render_suspect"] = sum(bool(r.get("render_suspect")) for r in rows)
        result["summary"] = summary
        warm = infer_s[1:] if len(infer_s) > 1 else infer_s
        result["latency"] = {
            "inference_call": percentiles(warm), "first_call_ms": infer_s[0] * 1e3 if infer_s else None,
            "mean_batch": float(np.mean(infer_batch)) if infer_batch else None,
            "amortised_per_control_step_ms": (float(np.mean(warm)) * 1e3 / k_exec) if warm else None,
            "control_period_ms": 1e3 / wd.CONTROL_HZ, "env_step_round": percentiles(env_step_s[1:]),
            "env_start_s": env_start_s,
            "note": "inference = preprocess + flow sampling + unnormalise, GPU-synchronised, batch of envs "
                    "needing a chunk on that tick; first call (warm-up) excluded from percentiles",
        }
        result["completed"] = final and len(rows) == len(jobs)
        result["wall_s"] = time.monotonic() - t0
        atomic_json(args.out, result)

    pending = list(jobs)
    slots = []
    for e in envs:
        if pending:
            slots.append(start(e, pending.pop(0)))
    try:
        while slots:
            need = [s for s in slots if not s["queue"]]
            if need:
                infer(need)
            prev_images = []
            t = time.perf_counter()
            for s in slots:
                a, clipped = clip_action(s["queue"].pop(0))
                s["clipped"] += int(clipped)
                s["env"].step_async(a)
                prev_images.append(s["obs"]["image"])
            outs = [s["env"].step_wait() for s in slots]
            env_step_s.append(time.perf_counter() - t)
            survivors = []
            for s, prev, (obs, reward, done, info) in zip(slots, prev_images, outs):
                check_obs(obs)
                s["steps"] += 1
                s["ret"] += float(reward)
                s["max_stage"] = max(s["max_stage"], int(info.get("stage", 0)))
                s["stale"] += int(np.array_equal(prev, obs["image"]))
                s["reward_mismatch"] += int((float(reward) == 1.0) != bool(info.get("success", False)))
                s["obs"] = obs
                termination = None
                if info.get("success"):
                    termination = "success"
                elif info.get("fell"):
                    termination = "fell"
                elif done:
                    termination = "env_done"
                elif s["steps"] >= args.max_steps:
                    termination = "max_steps"
                if termination is None:
                    survivors.append(s)
                    continue
                finish(s, info, termination)
                if pending:
                    survivors.append(start(s["env"], pending.pop(0)))
            slots = survivors
    finally:
        for e in envs:
            e.close()
    write(final=True)
    sm = result["summary"]["all"]
    print(f"[rollout] COMPLETE {sm['successes']}/{sm['episodes']} success; latency {result['latency']['inference_call']}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
