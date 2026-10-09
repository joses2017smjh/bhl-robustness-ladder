#!/usr/bin/env python3
"""One waiter environment in its own process, driven over stdin/stdout by rollout_vla.py.

Why a process: the policy interpreter (venv-lehome-train: lerobot, torch, numpy 2.x) has no MuJoCo, while the
phase 2 WaiterEnv runs in the main venv (MuJoCo 3.3.5, numpy 1.26). Messages therefore never pickle numpy arrays
(numpy 1.x cannot unpickle numpy 2.x arrays); arrays travel as (dtype, shape, bytes). One process per env also
gives parallel envs for free.

Protocol: 8-byte little-endian length + pickle (protocol 4) of plain Python objects.
  -> ("make", kind, kwargs)          <- ("ok", {"synthetic": bool})
  -> ("reset", seed, instruction_id) <- ("ok", obs)
  -> ("step", action)                <- ("ok", (obs, reward, done, info))
  -> ("close",)                      <- ("ok", None)
Errors come back as ("error", traceback string); the worker then exits.
"""
from __future__ import annotations

import os
import pickle
import struct
import sys
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def encode(obj):
    if isinstance(obj, np.ndarray):
        return {"__nd__": True, "dtype": obj.dtype.str, "shape": list(obj.shape),
                "data": np.ascontiguousarray(obj).tobytes()}
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: encode(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(encode(v) for v in obj)
    return obj


def decode(obj):
    if isinstance(obj, dict):
        if obj.get("__nd__") is True:
            return np.frombuffer(obj["data"], dtype=np.dtype(obj["dtype"])).reshape(obj["shape"]).copy()
        return {k: decode(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(decode(v) for v in obj)
    return obj


def send(stream, obj) -> None:
    blob = pickle.dumps(encode(obj), protocol=4)
    stream.write(struct.pack("<Q", len(blob)))
    stream.write(blob)
    stream.flush()


def recv(stream):
    head = stream.read(8)
    if len(head) < 8:
        raise EOFError("peer closed the pipe")
    (n,) = struct.unpack("<Q", head)
    blob = stream.read(n)
    if len(blob) < n:
        raise EOFError("truncated message")
    return decode(pickle.loads(blob))


def make_env(kind: str, kwargs: dict):
    if kind == "stub":
        sys.path.insert(0, str(HERE))
        from stub_env import StubWaiterEnv

        return StubWaiterEnv(**kwargs)
    if kind == "waiter":
        from bhl_robust.eval.waiter_env import WaiterEnv  # phase 2 (contract API)

        return WaiterEnv(**kwargs)
    raise ValueError(f"unknown env kind {kind}")


def serve(inp, out) -> int:
    env = None
    while True:
        try:
            msg = recv(inp)
        except EOFError:
            return 0
        try:
            cmd = msg[0]
            if cmd == "make":
                env = make_env(msg[1], msg[2])
                send(out, ("ok", {"synthetic": bool(getattr(env, "synthetic", False))}))
            elif cmd == "reset":
                send(out, ("ok", env.reset(msg[1], instruction_id=msg[2])))
            elif cmd == "step":
                send(out, ("ok", tuple(env.step(msg[1]))))
            elif cmd == "close":
                if hasattr(env, "close"):
                    env.close()
                send(out, ("ok", None))
                return 0
            else:
                raise ValueError(f"unknown command {cmd!r}")
        except Exception:
            send(out, ("error", traceback.format_exc()))
            return 1


if __name__ == "__main__":
    # Keep the protocol stream private: anything the env prints (Python or C level) goes to stderr.
    proto_out = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    raise SystemExit(serve(sys.stdin.buffer, proto_out))
