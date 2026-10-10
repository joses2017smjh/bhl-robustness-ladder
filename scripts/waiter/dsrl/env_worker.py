#!/usr/bin/env python3
"""One CPU env worker: runs a WaiterEnv-contract env and executes whole action chunks.

Launched by ``chunk_vec_env.ChunkVecEnv`` with ANY interpreter that can import the env (the MuJoCo/WBC
stack lives in ``Humanoid_Lite/venv``: numpy 1.26, mujoco 3.3.5, onnxruntime; the learner lives in
``venv-lehome-train``: lerobot, numpy 2.4). Messages are pickled builtins only (arrays travel as
``("__nd__", dtype, shape, bytes)``), so the two numpy versions never unpickle each other's objects.

Protocol (learner -> worker):  ("make", spec) | ("reset", seed, instruction_id) | ("chunk", actions)
                               | ("oracle",) | ("close",)
Replies: ("ok", payload) or ("error", traceback_text).
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import os
import sys
import traceback
from multiprocessing.connection import Client
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def enc(x):
    """numpy-free, version-independent encoding of nested obs/info structures."""
    if isinstance(x, np.ndarray):
        return ("__nd__", x.dtype.str, tuple(int(s) for s in x.shape), np.ascontiguousarray(x).tobytes())
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, dict):
        return {str(k): enc(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [enc(v) for v in x]
    return x


def dec(x):
    if isinstance(x, tuple) and len(x) == 4 and x[0] == "__nd__":
        return np.frombuffer(x[3], dtype=np.dtype(x[1])).reshape(x[2]).copy()
    if isinstance(x, dict):
        return {k: dec(v) for k, v in x.items()}
    if isinstance(x, list):
        return [dec(v) for v in x]
    return x


def make_env(spec: dict):
    for p in spec.get("sys_path", []):
        if p not in sys.path:
            sys.path.insert(0, p)
    kind = spec.get("kind", "stub")
    kwargs = dict(spec.get("kwargs", {}))
    if kind == "stub":
        if str(HERE) not in sys.path:
            sys.path.insert(0, str(HERE))
        from stub_env import StubWaiterEnv
        return StubWaiterEnv(**kwargs)
    if kind == "waiter":
        from bhl_robust.eval.waiter_env import WaiterEnv   # phase 2 (contract)
        if kwargs.get("cache_dir"):                        # one MJCF cache per worker process: 16 workers building the
            kwargs["cache_dir"] = f"{kwargs['cache_dir']}-{os.getpid()}"    # scene into one directory raced (smoke
        return WaiterEnv(**kwargs)                         # 21756753: ParseXML on a half-written file)
    if kind == "import":
        mod, cls = spec["target"].split(":")
        return getattr(importlib.import_module(mod), cls)(**kwargs)
    raise ValueError(f"unknown env kind {kind!r}")


class ChunkRunner:
    """Executes an (H, 15) chunk on one env, stopping at ``done``."""

    def __init__(self, env):
        self.env = env

    def reset(self, seed, instruction_id=None):
        if instruction_id is None:
            return self.env.reset(int(seed))
        return self.env.reset(int(seed), instruction_id=int(instruction_id))

    def chunk(self, actions):
        actions = np.asarray(actions, np.float32)
        total, n, done = 0.0, 0, False
        success = fell = False
        stage_max, info, obs = 0, {}, None
        for a in actions:
            obs, r, done, info = self.env.step(a)
            total += float(r)
            n += 1
            success |= bool(info.get("success", False))
            fell |= bool(info.get("fell", False))
            stage_max = max(stage_max, int(info.get("stage", 0)))
            if done:
                break
        return {"obs": obs, "reward": total, "done": bool(done), "success": success, "fell": fell,
                "truncated": bool(done and not success and not fell), "n": n, "stage_max": stage_max,
                "stage_last": int(info.get("stage", 0)), "t": float(info.get("t", 0.0)), "info": info}


def serve(address, worker_id, authkey):
    conn = Client(address, family="AF_UNIX", authkey=authkey)
    conn.send(("hello", worker_id, os.getpid(), sys.executable, np.__version__))
    runner = None
    while True:
        msg = conn.recv()
        cmd = msg[0]
        try:
            if cmd == "make":
                env = make_env(msg[1])
                runner = ChunkRunner(env)
                sig = str(inspect.signature(env.step))
                conn.send(("ok", {"env": type(env).__name__, "step_signature": sig,
                                  "toy": bool(getattr(env, "toy", False))}))
            elif cmd == "reset":
                conn.send(("ok", enc(runner.reset(msg[1], msg[2]))))
            elif cmd == "chunk":
                conn.send(("ok", enc(runner.chunk(dec(msg[1])))))
            elif cmd == "oracle":
                conn.send(("ok", enc(runner.env.oracle())))
            elif cmd == "close":
                conn.send(("ok", None))
                break
            else:
                conn.send(("error", f"unknown command {cmd!r}"))
        except Exception:
            conn.send(("error", traceback.format_exc()))
    conn.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--address", required=True)
    ap.add_argument("--worker-id", type=int, required=True)
    a = ap.parse_args()
    authkey = bytes.fromhex(os.environ["DSRL_WORKER_AUTHKEY"])
    serve(a.address, a.worker_id, authkey)


if __name__ == "__main__":
    main()
