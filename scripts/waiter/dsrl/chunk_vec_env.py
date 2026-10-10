"""Learner-side vector of WaiterEnv-contract envs stepped one ACTION CHUNK at a time.

Two backends with one API:
  * subprocess workers (``python=<interpreter>``): N CPU processes, each one env (``env_worker.py``);
    all workers step their chunks in parallel while the learner does GPU work (``step_async``).
  * in-process (``python=None``): sequential, for unit tests.

API: ``reset({i: (seed, instruction_id)}) -> {i: obs}``; ``step_async({i: (H, 15) array})``;
``step_wait() -> {i: result}`` where result = {obs, reward, done, success, fell, truncated, n,
stage_max, stage_last, t, info}.
"""
from __future__ import annotations

import os
import secrets
import subprocess
import sys
import tempfile
import time
from multiprocessing.connection import Listener
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from env_worker import ChunkRunner, dec, enc, make_env  # noqa: E402


class WorkerError(RuntimeError):
    pass


class ChunkVecEnv:
    def __init__(self, spec: dict, n: int, python: str | None = None, timeout_s: float = 900.0,
                 log_dir: str | None = None):
        self.spec, self.n, self.timeout_s = spec, int(n), float(timeout_s)
        self.inprocess = python is None
        self._pending: dict[int, object] = {}
        self.worker_info = []
        if self.inprocess:
            self.runners = [ChunkRunner(make_env(spec)) for _ in range(self.n)]
            self.worker_info = [{"backend": "inprocess", "python": sys.executable, "numpy": np.__version__}]
            return
        authkey = secrets.token_bytes(16)
        self._tmp = tempfile.mkdtemp(prefix="dsrl_vec_", dir=os.environ.get("TMPDIR", "/tmp"))
        address = os.path.join(self._tmp, "learner.sock")
        self.listener = Listener(address, family="AF_UNIX", authkey=authkey)
        env = dict(os.environ, DSRL_WORKER_AUTHKEY=authkey.hex())
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        self.procs = []
        for i in range(self.n):
            out = open(os.path.join(log_dir, f"worker{i}.log"), "ab") if log_dir else subprocess.DEVNULL
            self.procs.append(subprocess.Popen(
                [python, str(HERE / "env_worker.py"), "--address", address, "--worker-id", str(i)],
                env=env, stdout=out, stderr=subprocess.STDOUT if log_dir else subprocess.DEVNULL))
        conns = {}
        for _ in range(self.n):
            c = self.listener.accept()
            hello = c.recv()
            conns[int(hello[1])] = c
            self.worker_info.append({"worker": hello[1], "pid": hello[2], "python": hello[3], "numpy": hello[4]})
        self.conns = [conns[i] for i in range(self.n)]
        for c in self.conns:
            c.send(("make", spec))
        self.env_desc = [self._recv(i) for i in range(self.n)]

    # ---- low level
    def _recv(self, i):
        c = self.conns[i]
        deadline = time.time() + self.timeout_s
        while not c.poll(1.0):
            if self.procs[i].poll() is not None:
                raise WorkerError(f"worker {i} exited with code {self.procs[i].returncode}")
            if time.time() > deadline:
                raise WorkerError(f"worker {i} timed out after {self.timeout_s:.0f} s")
        status, payload = c.recv()
        if status != "ok":
            raise WorkerError(f"worker {i}: {payload}")
        return dec(payload)

    # ---- API
    def reset(self, todo: dict) -> dict:
        if self.inprocess:
            return {i: self.runners[i].reset(s, iid) for i, (s, iid) in todo.items()}
        for i, (s, iid) in todo.items():
            self.conns[i].send(("reset", int(s), None if iid is None else int(iid)))
        return {i: self._recv(i) for i in todo}

    def step_async(self, chunks: dict):
        if self._pending:
            raise RuntimeError("step_async called twice without step_wait")
        if self.inprocess:
            self._pending = dict(chunks)
            return
        for i, a in chunks.items():
            self.conns[i].send(("chunk", enc(np.asarray(a, np.float32))))
        self._pending = {i: None for i in chunks}

    def step_wait(self) -> dict:
        pending, self._pending = self._pending, {}
        if self.inprocess:
            return {i: self.runners[i].chunk(a) for i, a in pending.items()}
        return {i: self._recv(i) for i in pending}

    def oracle(self, i):
        if self.inprocess:
            return self.runners[i].env.oracle()
        self.conns[i].send(("oracle",))
        return self._recv(i)

    def close(self):
        if self.inprocess:
            return
        for i, c in enumerate(self.conns):
            try:
                c.send(("close",))
                self._recv(i)
            except Exception:
                pass
        for p in self.procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
        self.listener.close()
