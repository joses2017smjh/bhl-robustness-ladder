#!/usr/bin/env python3
"""ER-OBS-1-OW: the ER-OBS-1 observer check with OPEN-WEIGHT vision-language models on our own GPUs.

The Gemini arm (scripts/bench/er_obs.py, design G) asks Gemini Robotics-ER 2 to read four booleans and a cube point
from rendered frames of the scripted cooperative lift. This arm asks the same question, with the same frames
(er-obs1-data/frames-run, job 21544356), the same prompt text (client.TEXT, sha256 client.PROMPT_SCHEMA_SHA256), the
same answer parser (client.classify) and the same pure scoring functions (verdict.accuracy_verdict, latency_verdict),
of open-weight models run locally with the Hugging Face `image-text-to-text` pipeline. Only the caller changes; no
file of the Gemini arm is edited. A different question from the Gemini arm's (an open model, not Gemini), so it is
its own predeclared arm (SLURM_JOBS.md).

    probe     development only: a few THROW-AWAY smoke frames (seeds 900/901), raw outputs printed; never scored
    calls     --mode smoke (a smoke frames dir) | run (the scored frames-run): one generation per frame and variant
              (main, C1); calls.json written once
    latency   50 sequential generations on the first 50 scored main frames; latency.json written once
    verdict   accuracy | latency, from calls.json / latency.json, written once (smoke -> PIPELINE_CHECK)

Decoding is greedy (do_sample False), max_new_tokens MAX_NEW_TOKENS, bf16 weights, batch 1, one GPU. A failed or
malformed answer is scored exactly as in the Gemini arm (main: wrong; C1: correct). Labels: LEARNED gait (frozen
arms-dr1.0-s0) + SCRIPTED arms + OPEN-WEIGHT VLM observer (local GPU, rendered RGB, offline).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import socket
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))

from bhl_robust.eval import er_obs as E                      # noqa: E402
from bhl_robust.eval.er_obs import client as C               # noqa: E402
from bhl_robust.eval.er_obs import verdict as V              # noqa: E402

LABEL = ("LEARNED gait (frozen arms-dr1.0-s0) + SCRIPTED arms + OPEN-WEIGHT VLM observer (local GPU, rendered RGB, "
         "offline)")
ARM = "ER-OBS-1-OW"
MAX_NEW_TOKENS = 256
DTYPE = "bfloat16"
# name -> (Hugging Face repo, pinned revision): the commits downloaded 2026-10-04 (solutions-20260930/campaign-er-obs-ow/
# models.json) and frozen after the development probe (throw-away smoke frames only), before any scored frame.
MODELS = {
    "molmo2-er": ("allenai/Molmo2-ER", "dab22564403d2607855bb1fffb0721285b445081"),
    "qwen3-vl-8b": ("Qwen/Qwen3-VL-8B-Instruct", "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"),
}
LATENCY_CALLS = 50


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _cli():
    """scripts/bench/er_obs.py, loaded by path (its manifest, frame and mask helpers are reused unchanged)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("er_obs_cli_for_local", HERE / "er_obs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Observer:
    """One open-weight model, loaded as its model card does (AutoProcessor + AutoModelForImageTextToText, remote code
    allowed for Molmo2) from the PINNED local snapshot (no network); greedy, batch 1, one GPU. The timed span is the
    whole answer: chat template + image preprocessing + generation + decoding."""

    def __init__(self, name: str, revision: str | None = None):
        repo, pinned = MODELS[name]
        self.name, self.repo, self.revision = name, repo, revision or pinned
        if self.revision is None:
            raise SystemExit(f"ER-OBS-1-OW: REFUSED ({name} has no pinned revision)")
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch = torch
        self.path = snapshot_download(repo, revision=self.revision, local_files_only=True)
        self.processor = AutoProcessor.from_pretrained(self.path, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(self.path, trust_remote_code=True,
                                                                 dtype=getattr(torch, DTYPE)).to("cuda").eval()
        self.gpu = torch.cuda.get_device_name(0)

    def ask(self, png: bytes) -> tuple[str, float]:
        from PIL import Image
        img = Image.open(io.BytesIO(png)).convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": C.TEXT}]}]
        self.torch.cuda.synchronize()
        t0 = time.perf_counter()
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                    return_dict=True, return_tensors="pt").to("cuda")
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        new = out[:, inputs["input_ids"].shape[1]:]
        text = self.processor.batch_decode(new, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        self.torch.cuda.synchronize()
        return str(text), time.perf_counter() - t0


def classify_text(text: str, model_repo: str) -> dict:
    """The local answer through the Gemini arm's own parser: wrapped in the interactions response shape it reads."""
    body = json.dumps({"id": "local", "model": model_repo, "status": "completed", "output_text": text}).encode()
    return C.classify(200, body)


def provenance(**extra) -> dict:
    return {"arm": ARM, "label": LABEL, "prompt_text": C.TEXT, "prompt_schema_sha256": C.PROMPT_SCHEMA_SHA256,
            "decoding": {"do_sample": False, "max_new_tokens": MAX_NEW_TOKENS, "dtype": DTYPE, "batch": 1},
            "source_sha256": {p: sha256(REPO / p) for p in ("scripts/bench/er_obs_local.py",
                                                            "src/bhl_robust/eval/er_obs/client.py",
                                                            "src/bhl_robust/eval/er_obs/verdict.py",
                                                            "scripts/bench/er_obs.py")},
            "host": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **extra}


def cmd_probe(a) -> int:
    cli = _cli()
    man = cli.load_manifest(a.frames, need_run=False)
    if man.get("mode") != "smoke":
        raise SystemExit("ER-OBS-1-OW PROBE: REFUSED (probe runs on a SMOKE frames dir only, never the scored frames)")
    obs = Observer(a.model, a.revision)
    for rec in man["frames"][: a.n]:
        for variant in E.VARIANTS:
            text, dt = obs.ask(cli.frame_png(a.frames, rec, variant))
            r = classify_text(text, obs.repo)
            print(json.dumps({"frame": rec["id"], "variant": variant, "latency_s": round(dt, 3), "kind": r["kind"],
                              "booleans": r.get("booleans"), "labels": rec["labels"], "text": text[:400]}), flush=True)
    return 0


def cmd_calls(a) -> int:
    cli = _cli()
    run = a.mode == "run"
    man = cli.load_manifest(a.frames, need_run=run)
    if not run and man.get("mode") != "smoke":
        raise SystemExit("ER-OBS-1-OW CALLS: REFUSED (smoke mode needs a SMOKE frames dir)")
    if run:
        cli.refuse_short_classes(man)
    out = a.out / "calls.json"
    if out.exists():
        raise SystemExit(f"ER-OBS-1-OW CALLS: REFUSED ({out} exists; outputs are never overwritten)")
    a.out.mkdir(parents=True, exist_ok=True)
    obs = Observer(a.model, a.revision)
    records, kinds = [], {}
    for rec in man["frames"]:
        for variant in E.VARIANTS:
            png = cli.frame_png(a.frames, rec, variant)
            try:
                text, dt = obs.ask(png)
                r = classify_text(text, obs.repo)
                r["latency_s"] = round(dt, 4)
            except Exception as e:  # noqa: BLE001 -- a generation failure is a failed call, scored as such
                r = {"kind": "transport_error", "error": f"{type(e).__name__}: {e}"[:300], "booleans": None,
                     "latency_s": None}
            r.update(frame_id=rec["id"], variant=variant, image_sha256=hashlib.sha256(png).hexdigest(),
                     mode=a.mode, model=obs.repo, revision=obs.revision)
            records.append(r)
            kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
        print(f"  {rec['id']} done | kinds so far {kinds}", flush=True)
    payload = provenance(schema="er_obs1_ow_calls_v1", mode=a.mode, model=obs.repo, revision=obs.revision,
                         gpu=obs.gpu, frames_manifest=man["_path"], frames_manifest_sha256=man["_sha256"],
                         records=records, outcome_kinds=kinds)
    V.write_once(out, payload)
    print(f"ER-OBS-1-OW CALLS {a.mode} {a.model}: COMPLETE -> {out} | kinds {kinds}", flush=True)
    return 0


def cmd_latency(a) -> int:
    cli = _cli()
    man = cli.load_manifest(a.frames, need_run=True)
    out = a.out / "latency.json"
    if out.exists():
        raise SystemExit(f"ER-OBS-1-OW LATENCY: REFUSED ({out} exists)")
    a.out.mkdir(parents=True, exist_ok=True)
    obs = Observer(a.model, a.revision)
    obs.ask(cli.frame_png(a.frames, man["frames"][0], "main"))          # warm-up, not counted
    records = []
    for rec in man["frames"][:LATENCY_CALLS]:
        try:
            text, dt = obs.ask(cli.frame_png(a.frames, rec, "main"))
            r = classify_text(text, obs.repo)
            r["latency_s"] = round(dt, 4)
        except Exception as e:  # noqa: BLE001
            r = {"kind": "transport_error", "error": f"{type(e).__name__}: {e}"[:300], "latency_s": None}
        r.update(frame_id=rec["id"])
        records.append(r)
    V.write_once(out, provenance(schema="er_obs1_ow_latency_v1", mode="run", model=obs.repo, revision=obs.revision,
                                 gpu=obs.gpu, warmup_calls=1, records=records))
    print(f"ER-OBS-1-OW LATENCY {a.model}: COMPLETE -> {out}", flush=True)
    return 0


def cmd_verdict(a) -> int:
    cli = _cli()
    if a.which == "accuracy":
        calls = json.loads(a.calls.read_text())
        man = cli.load_manifest(a.frames, need_run=calls.get("mode") == "run")
        if calls.get("frames_manifest_sha256") != man["_sha256"] or calls.get("prompt_schema_sha256") != C.PROMPT_SCHEMA_SHA256:
            raise SystemExit("ER-OBS-1-OW VERDICT: REFUSED (calls.json used other frames or another prompt)")
        by_id = {f["id"]: f for f in man["frames"]}
        outcomes = {}
        for r in calls["records"]:
            f = by_id[r["frame_id"]]
            if r["image_sha256"] != f["sha256"]["rgb" if r["variant"] == "main" else "c1"]:
                raise SystemExit(f"ER-OBS-1-OW VERDICT: REFUSED ({r['frame_id']}/{r['variant']} image mismatch)")
            outcomes[(r["frame_id"], r["variant"])] = r
        hits = {f["id"]: V.point_hit(outcomes[(f["id"], "main")]["answer"]["cube_point"], cli.mask_of(a.frames, f))
                for f in man["frames"] if f["cube_visible"] and outcomes.get((f["id"], "main"), {}).get("kind") == "ok"}
        v = V.accuracy_verdict(man["frames"], outcomes, hits, expected_frames=man["expected_frames"])
        mode, model = calls["mode"], calls["model"]
        inputs = {"calls": str(a.calls), "calls_sha256": sha256(a.calls), "frames_manifest_sha256": man["_sha256"]}
    else:
        lat = json.loads(a.latency.read_text())
        v = V.latency_verdict(lat["records"])
        mode, model = lat["mode"], lat["model"]
        inputs = {"latency": str(a.latency), "latency_sha256": sha256(a.latency), "gpu": lat.get("gpu")}
    if mode != "run":
        v["computed_verdict"], v["verdict"] = v["verdict"], "PIPELINE_CHECK"
    V.write_once(a.out, provenance(**v, which=a.which, mode=mode, model=model, inputs=inputs))
    print(f"ER-OBS-1-OW {a.which.upper()} {model} ({mode}): {v['verdict']} -> {a.out}", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("probe", "calls", "latency"):
        p = sub.add_parser(name)
        p.add_argument("--model", choices=sorted(MODELS), required=True)
        p.add_argument("--revision", default=None, help="probe only: a revision before one is pinned")
        p.add_argument("--frames", type=Path, required=True)
        if name == "probe":
            p.add_argument("-n", type=int, default=3)
        else:
            p.add_argument("--out", type=Path, required=True)
        if name == "calls":
            p.add_argument("--mode", choices=("smoke", "run"), required=True)
    v = sub.add_parser("verdict")
    v.add_argument("which", choices=("accuracy", "latency"))
    v.add_argument("--frames", type=Path)
    v.add_argument("--calls", type=Path)
    v.add_argument("--latency", type=Path)
    v.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd != "probe" and getattr(a, "revision", None):
        raise SystemExit("ER-OBS-1-OW: REFUSED (--revision is for the development probe; runs use the pinned one)")
    return {"probe": cmd_probe, "calls": cmd_calls, "latency": cmd_latency, "verdict": cmd_verdict}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
