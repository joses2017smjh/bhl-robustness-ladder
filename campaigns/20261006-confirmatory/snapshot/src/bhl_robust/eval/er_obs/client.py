"""ER-OBS-1 caller: ONE fixed prompt + JSON schema, the API-key guard, the transports, the call policy (pacing and
re-sends) and the response check.

What is sent: the rendered PNG frame (inline, base64) and the fixed text (the prompt, then the JSON schema), with the
fixed generation config, in the documented Gemini Robotics-ER 2 REST shape (saved official docs,
solutions-20260930/campaign-m7-180-gemini/aidev_robotics-overview.txt:211-235 and aidev_robotics-spatial.txt:44-64):

    POST https://generativelanguage.googleapis.com/v1beta/interactions
    headers: x-goog-api-key: <key>, Content-Type: application/json
    {"model": "gemini-robotics-er-2-preview",
     "input": [{"type": "image", "data": "<base64 PNG>", "mime_type": "image/png"}, {"type": "text", "text": TEXT}],
     "generation_config": {"thinking_level": "low"}}

Nothing else: no repo data, no paths, no seeds, no user data. REQUEST AMENDMENT (2026-10-05, before any scored call):
the saved robotics REST example's shape was refused by the live service at two preflights on a throw-away smoke
frame (job 21564188: HTTP 400 "Unknown parameter 'thinking_config' at 'generation_config'"; job 21564286: HTTP 400
"The 'type' parameter is required at 'input'"). The body now follows the live Interactions API reference and its
image-understanding REST example: a typed input list and the flat `generation_config.thinking_level` (the form all
four SDK examples use); the answer is read from the model_output step's text (ANSWER_FIELD), since the REST JSON has
no top-level output_text. No response-
schema field is documented for this endpoint: the docs request a JSON format in the prompt ("adjust the requested JSON
schema in the prompt", aidev_robotics-agentic.txt:806-809), so the schema travels in the text part, and
PROMPT_SCHEMA_SHA256 covers that exact text and the schema. The answer is read as ANSWER_FIELD says. A preflight
call on a smoke frame must pass before any scored call, and it records the raw response and the token usage.

The key travels only in the `x-goog-api-key` header; it is read from the file named by $GEMINI_API_KEY_FILE (default
~/.config/bhl/gemini_api_key) after `check_key_file` passes (regular file, owned by this user, mode 600, outside the
repo and the outputs) and is never printed, logged, stored or put in argv or the environment. Every stored or printed
error string and response body goes through `redact`.

Call policy (`CALL_POLICY`, decided after the 2026-10-03 review and before any real call): at least MIN_GAP_S between
the end of one request and the start of the next; a request is re-sent ONLY after HTTP 429, HTTP 503 or a connection
error with no HTTP response (the model produced no answer), honouring Retry-After (or google.rpc.RetryInfo), at most
MAX_RESENDS times per frame and variant within one job; when the re-sends are used up, or the service asks for a wait
longer than MAX_WAIT_S, the caller stops with no final outcome for that frame (resume later). Every other outcome is
final at its first attempt.

MOCK mode (`MockTransport`) never touches the network (the CLI also installs `disable_network`) and never opens the
key file; it synthesizes interaction-shaped responses from the frame labels, including malformed JSON, schema
violations, prose refusals, missing answers, HTTP errors, re-sendable 429/503/connection failures and timeouts, and it
runs on a virtual clock (`VirtualClock`), so pacing and waits are exercised without sleeping.
"""

from __future__ import annotations

import base64
import email.utils
import hashlib
import json
import math
import os
import re
import stat
import time
from pathlib import Path

from bhl_robust.eval.er_obs import BOOLEANS

MODEL = "gemini-robotics-er-2-preview"
ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/interactions"
CONNECT_TIMEOUT_S = 10.0
TIMEOUT_S = 60.0                  # a call slower than this (wall) is a timeout, whatever it returned
ABORT_CONSECUTIVE = 5             # final transport-level failures in a row that stop the caller
MIN_GAP_S = 1.0                   # end of one request -> start of the next (every request)
MAX_RESENDS = 3                   # re-sends per frame and variant within one job, only after a RETRYABLE outcome
BACKOFF_S = (5.0, 15.0, 45.0)     # wait before re-send 1, 2, 3 when the response gives no Retry-After/RetryInfo
MAX_WAIT_S = 120.0                # a requested wait longer than this stops the caller (no final outcome; resume later)
RETRYABLE_HTTP = frozenset({429, 503})
RAW_CAP_CALL = 8192               # bytes of the (redacted) raw response body kept per call
RAW_CAP_PREFLIGHT = 65536         # ... and for the preflight (the one place the live response shape is seen first)
KEY_ENV = "GEMINI_API_KEY_FILE"
DEFAULT_KEY_FILE = "~/.config/bhl/gemini_api_key"
PAID_TIER_ENV = "ER_OBS_PAID_TIER_CONFIRMED"
TIER_ENV = "ER_OBS_TIER"          # tier amendment (2026-10-05): "paid" or "free", the operator's declaration
TIERS = ("paid", "free")

#: Clock and sleep of the real caller, resolved at call time (tests monkeypatch these module attributes; nothing in
#: the CLI or the launchers can change them, so a real run is always paced in real time).
CLOCK = time.monotonic
SLEEP = time.sleep

#: The ONE fixed prompt (frozen 2026-10-03, before any call).
PROMPT = (
    "This image is one frame from a fixed camera watching a physics simulation. Two humanoid robots (orange and "
    "black) stand side by side on a checkered floor. Between them is a small grey plinth (a box-shaped stand, "
    "0.19 m tall). A blue cube, 0.28 m on each side, starts on top of the plinth; the robots may squeeze it between "
    "their hands, lift it, hold it, lower it, release it or drop it.\n"
    "Report the state of the blue cube in this frame only, as JSON with these fields:\n"
    "- cube_point: one point on the blue cube as [y, x], each normalized to 0-1000 (y from the top edge of the "
    "image, x from the left edge).\n"
    "- lifted_clear: true if the cube is lifted clear: its lowest corner is at least 2 cm above every surface "
    "directly below it (the floor, and the plinth top when the cube is over the plinth), so it rests on neither.\n"
    "- on_floor: true if any part of the cube touches the floor.\n"
    "- robot_contact: true if the robots press on the cube: the total contact force from the robots on the cube is "
    "more than 1 newton.\n"
    "- seated_flat: true if the cube sits on the plinth: it touches the plinth with its centre over the plinth top, "
    "and one of its faces (any face) is flat down, within 8 degrees of level.\n"
    "Answer with the JSON object only."
)

#: The fixed answer schema (minimal keywords; the 0-1000 range is stated and checked by `validate_answer`).
SCHEMA = {
    "type": "object",
    "properties": {
        "cube_point": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2,
                       "description": "[y, x] of one point on the blue cube, each normalized to 0-1000"},
        "lifted_clear": {"type": "boolean"},
        "on_floor": {"type": "boolean"},
        "robot_contact": {"type": "boolean"},
        "seated_flat": {"type": "boolean"},
    },
    "required": ["cube_point", "lifted_clear", "on_floor", "robot_contact", "seated_flat"],
}

#: The exact text part sent: the prompt, then the schema as JSON (the documented way to ask ER 2 for a JSON format).
SCHEMA_LINE = "JSON schema of the answer: "
TEXT = PROMPT + "\n" + SCHEMA_LINE + json.dumps(SCHEMA, sort_keys=True, ensure_ascii=True)
THINKING_LEVEL = "low"
GENERATION_CONFIG = {"thinking_level": THINKING_LEVEL}     # flat, as the SDKs (request amendment 2026-10-05)
#: Left at the API's defaults (recorded): everything the documented REST example does not set.
API_DEFAULTS_NOTE = ("only generation_config.thinking_level is set (the documented examples' one config field, in the "
                     "SDKs' flat form: the live service refused the REST example's nested thinking_config); temperature, top_p, the output-token cap (documented output limit 65,536), "
                     "media resolution, safety settings, system instruction and interaction storage are not set (API "
                     "defaults); single-turn request; no tools")

CALL_POLICY = {
    "min_gap_s": MIN_GAP_S,
    "gap": "a request is sent no earlier than min_gap_s after the previous request ended (response, error or "
           "timeout): preflight, scored calls, re-sends and latency calls alike",
    "resend_only_on": "HTTP 429, HTTP 503, or a connection error with no HTTP response (requests ConnectionError, "
                      "ConnectTimeout included; a read timeout is final): the model produced no answer",
    "max_resends_per_frame_per_job": MAX_RESENDS,
    "wait_before_resend": "max(Retry-After header, google.rpc.RetryInfo retryDelay) when the response gives one, else "
                          "5, 15, 45 s for re-send 1, 2, 3; never less than min_gap_s",
    "max_wait_s": MAX_WAIT_S,
    "when_resends_are_used_up_or_wait_too_long": "the caller stops (exit 4) with NO final outcome for that frame "
                                                 "(a 429/503/connection error is never a scored outcome); "
                                                 "resubmitting resumes",
    "every_other_outcome": "final at its first attempt (cached; failures included)",
    "latency_mode": "no re-sends: a 429/503/connection error there is a failed call (+inf)",
    "attempts": "every attempt (re-sends included) is appended to the frame's attempts log and copied into its final "
                "record",
}


def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


#: Where the answer is read (request amendment 2026-10-05, the live Interactions API reference): the text items of
#: every "model_output" step, joined in order; a top-level output_text (the SDKs' convenience property, not in the
#: REST JSON) is read only when the response has no steps list.
ANSWER_FIELD = ("steps[type=model_output].content[type=text].text, joined in order; top-level output_text only when "
                "the response has no steps list")

#: sha256 of the exact text part the model reads (prompt + schema line) and the schema the answer is checked against.
PROMPT_SCHEMA_SHA256 = sha256_hex(canonical({"prompt_text": TEXT, "json_schema": SCHEMA}))


def build_request(png: bytes) -> dict:
    """The request body (documented REST shape): the PNG frame inline and the fixed text, with the fixed generation
    config. Nothing else."""
    return {"model": MODEL,
            "input": [{"type": "image", "data": base64.b64encode(png).decode("ascii"), "mime_type": "image/png"},
                      {"type": "text", "text": TEXT}],
            "generation_config": dict(GENERATION_CONFIG)}


def request_template() -> dict:
    """`build_request` with the image replaced by a placeholder, plus the endpoint, the answer field and the call
    policy (no key, no image)."""
    body = build_request(b"")
    body["input"][0]["data"] = "<base64 PNG frame>"
    return {"method": "POST", "endpoint": ENDPOINT, "headers": {"Content-Type": "application/json",
                                                                "x-goog-api-key": "<from the key file; never stored>"},
            "body": body, "answer_field": ANSWER_FIELD,
            "answer_json": "one JSON object, bare or in exactly one ``` / ```json code block",
            "connect_timeout_s": CONNECT_TIMEOUT_S, "wall_timeout_s": TIMEOUT_S, "call_policy": CALL_POLICY,
            "api_defaults": API_DEFAULTS_NOTE}


REQUEST_SHA256 = sha256_hex(canonical(request_template()))


# ------------------------------------------------------------------ key guard

class KeyRefused(Exception):
    """The key file failed a check. The message never contains the key."""


def key_file_path(environ=None) -> Path:
    env = os.environ if environ is None else environ
    raw = env.get(KEY_ENV) or DEFAULT_KEY_FILE
    return Path(os.path.expanduser(raw))


def paid_tier_confirmed(environ=None) -> bool:
    """The operator's confirmation that the key is a PAID-tier, API-restricted key (ER_OBS_PAID_TIER_CONFIRMED=1)."""
    env = os.environ if environ is None else environ
    return env.get(PAID_TIER_ENV) == "1"


def declared_tier(environ=None) -> str | None:
    """The operator's tier declaration (tier amendment, 2026-10-05): "paid" iff ER_OBS_PAID_TIER_CONFIRMED=1 or
    ER_OBS_TIER=paid, "free" iff ER_OBS_TIER=free; None (refuse) when neither is given, ER_OBS_TIER holds anything
    else, or the two contradict each other (ER_OBS_PAID_TIER_CONFIRMED=1 with ER_OBS_TIER=free)."""
    env = os.environ if environ is None else environ
    raw = env.get(TIER_ENV)
    if raw is not None and raw not in TIERS:
        return None
    if paid_tier_confirmed(env):
        return None if raw == "free" else "paid"
    return raw


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def check_key_file(path, forbidden_roots) -> Path:
    """Refuse unless `path` is an absolute path to a regular file (not a symlink) owned by this user, with mode
    exactly 600, non-empty, and neither it nor its resolved path is inside any of `forbidden_roots` (the code root,
    the main repo, the frames and the output directories). Returns the resolved path."""
    p = Path(path)
    if not p.is_absolute():
        raise KeyRefused(f"key file path must be absolute ({KEY_ENV} or the default {DEFAULT_KEY_FILE})")
    if not os.path.lexists(p):
        raise KeyRefused(f"key file missing: {p} (set {KEY_ENV} or create {DEFAULT_KEY_FILE} with mode 600)")
    lst = os.lstat(p)
    if stat.S_ISLNK(lst.st_mode):
        raise KeyRefused(f"key file must be a regular file, not a symlink: {p}")
    if not stat.S_ISREG(lst.st_mode):
        raise KeyRefused(f"key file is not a regular file: {p}")
    if lst.st_uid != os.getuid():
        raise KeyRefused(f"key file is not owned by this user: {p}")
    mode = stat.S_IMODE(lst.st_mode)
    if mode != 0o600:
        raise KeyRefused(f"key file mode is {mode:03o}, must be 600: {p}")
    if lst.st_size == 0 or lst.st_size > 4096:
        raise KeyRefused(f"key file is empty or implausibly large ({lst.st_size} bytes): {p}")
    real = p.resolve()
    for root in forbidden_roots:
        if root is None:
            continue
        r = Path(root)
        for cand in {r.absolute(), r.resolve()}:
            if _inside(p.absolute(), cand) or _inside(real, cand):
                raise KeyRefused(f"key file is inside {cand} (it must live outside the repo and the outputs)")
    return real


_KEY_CHARS = re.compile(r"^[A-Za-z0-9_\-.]{8,512}$")


def read_key(path) -> str:
    """The key, stripped; refused (without echoing anything from the file) unless it is one plausible token."""
    try:
        key = Path(path).read_bytes().decode("ascii").strip()
    except (OSError, UnicodeDecodeError):
        raise KeyRefused("key file unreadable or not ASCII") from None
    if not _KEY_CHARS.match(key):
        raise KeyRefused("key file content is not a single API-key token")
    return key


def redact(text, secret: str | None) -> str:
    s = "" if text is None else str(text)
    if secret:
        s = s.replace(secret, "[REDACTED]")
    return s


# ------------------------------------------------------------------ transports

class TransportTimeout(Exception):
    """The request went out and no complete response came within the timeout: a FINAL outcome (timeout)."""


class TransportError(Exception):
    """Any other transport failure: a FINAL outcome (transport_error)."""


class TransportConnectionError(TransportError):
    """A connection error with no HTTP response (the model produced no answer): the one re-sendable transport case."""


class HttpTransport:
    """POST to the interactions endpoint with requests (keep-alive session, so the measured latency is the service's,
    not a fresh TLS handshake per call). Returns (status, body, {"retry_after": header or None}). The key lives only
    in this object's header dict."""

    def __init__(self, key: str, endpoint: str = ENDPOINT):
        import requests
        self._requests = requests
        self._session = requests.Session()
        self._session.trust_env = False          # no proxy/netrc surprises from the environment
        self._headers = {"Content-Type": "application/json", "x-goog-api-key": key}
        self._key = key
        self.endpoint = endpoint

    def __repr__(self):
        return f"HttpTransport(endpoint={self.endpoint!r})"

    def __call__(self, body: bytes):
        rq = self._requests
        try:
            r = self._session.post(self.endpoint, data=body, headers=self._headers,
                                   timeout=(CONNECT_TIMEOUT_S, TIMEOUT_S), allow_redirects=False)
            return int(r.status_code), r.content, {"retry_after": r.headers.get("Retry-After")}
        except rq.exceptions.ConnectTimeout as e:            # never connected: nothing was sent
            raise TransportConnectionError(redact(f"{type(e).__name__}: {e}", self._key)[:300]) from None
        except rq.exceptions.Timeout as e:                   # sent; no response in time: final
            raise TransportTimeout(type(e).__name__) from None
        except rq.exceptions.ConnectionError as e:
            import urllib3
            if e.args and isinstance(e.args[0], urllib3.exceptions.ReadTimeoutError):
                raise TransportTimeout("ReadTimeoutError (body)") from None
            raise TransportConnectionError(redact(f"{type(e).__name__}: {e}", self._key)[:300]) from None
        except rq.exceptions.RequestException as e:
            raise TransportError(redact(f"{type(e).__name__}: {e}", self._key)[:300]) from None


def disable_network() -> None:
    """Make any socket connection raise (mock mode and tests): nothing can reach the network."""
    import socket

    def _guard(*_a, **_k):
        raise RuntimeError("network disabled (ER-OBS-1 mock mode)")

    socket.socket.connect = _guard
    socket.socket.connect_ex = _guard
    socket.create_connection = _guard
    socket.getaddrinfo = _guard


class VirtualClock:
    """A deterministic clock for mock mode and tests: `sleep` and `advance` move it; nothing waits."""

    def __init__(self, t: float = 1000.0):
        self.t = float(t)
        self.slept = []

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        s = max(0.0, float(s))
        self.slept.append(s)
        self.t += s

    def advance(self, s: float) -> None:
        self.t += max(0.0, float(s))


# ------------------------------------------------------------------ mock service

def _interaction(text: str | None = None, *, usage: bool = True, thought_tokens: int = 37) -> bytes:
    """An Interaction resource as the live REST API returns it (mock): a thought step, then the answer as one text
    item of a model_output step, plus the usage block (total_thought_tokens etc.)."""
    resp = {"id": "mock-interaction", "model": MODEL, "status": "completed", "object": "interaction",
            "steps": [{"type": "thought", "summary": []}]}
    if text is not None:
        resp["steps"].append({"type": "model_output", "content": [{"type": "text", "text": text}]})
    if usage:
        resp["usage"] = {"total_input_tokens": 1290, "total_output_tokens": 61,
                         "total_thought_tokens": thought_tokens, "total_tokens": 1290 + 61 + thought_tokens}
    return json.dumps(resp).encode()


def _err(code: int, status: str, message: str, retry_delay: str | None = None) -> bytes:
    err = {"code": code, "message": message, "status": status}
    if retry_delay is not None:
        err["details"] = [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry_delay}]
    return json.dumps({"error": err}).encode()


MOCK_KINDS = ("correct", "one_wrong", "fenced_json", "malformed_json", "prose", "schema_violation",
              "c1_point_out_of_range", "no_output", "no_usage", "bad_body", "http_500", "http_400",
              "http_429_then_ok", "http_503_then_ok", "conn_error_then_ok", "http_429_always", "http_429_long",
              "timeout")


def mock_plan(frame_id: str, variant: str, seed: int = 0) -> str:
    """Deterministic synthetic outcome per frame and variant: mostly well-formed answers, with failure kinds mixed in
    (every final failure counts as a failed call; the re-sendable kinds recover on their re-send)."""
    u = int(sha256_hex(f"{seed}:{frame_id}:{variant}".encode())[:8], 16) / 0xFFFFFFFF
    edges = ((0.72, "correct"), (0.80, "one_wrong"), (0.83, "malformed_json"), (0.86, "schema_violation"),
             (0.89, "prose"), (0.91, "no_output"), (0.93, "http_500"), (0.95, "http_429_then_ok"), (0.97, "timeout"),
             (1.01, "fenced_json"))
    return next(kind for edge, kind in edges if u < edge)


class MockTransport:
    """Synthetic interaction responses keyed by the image's sha256 (the frame is identified from the request body
    itself, as the real service would only see the image). Never touches the network or the key file. With a
    `clock` (VirtualClock) every request advances it by a synthetic latency."""

    def __init__(self, frames: list[dict], image_sha: dict, seed: int = 0, plan=mock_plan, force: dict | None = None,
                 clock: VirtualClock | None = None, latency_s: float = 0.25):
        # image_sha: sha256 of the PNG -> (frame record, variant)
        self.by_sha = dict(image_sha)
        self.seed = int(seed)
        self.plan = plan
        self.force = dict(force or {})           # (frame_id, variant) -> kind
        self.clock = clock
        self.latency_s = float(latency_s)
        self.next_latency = None                 # one-shot override (latency mode)
        self.attempts = {}                       # (frame_id, variant) -> requests seen
        self.calls = []

    def _answer(self, frame: dict, variant: str, kind: str) -> dict:
        labels = frame["labels"]
        if variant == "c1":
            # a chance-level observer: booleans from the hash, never from the labels
            bits = int(sha256_hex(f"c1:{self.seed}:{frame['id']}".encode())[:4], 16)
            ans = {b: bool(bits >> i & 1) for i, b in enumerate(BOOLEANS)}
            ans["cube_point"] = [500, 500]
        else:
            ans = {b: bool(labels[b]) for b in BOOLEANS}
            if kind == "one_wrong":
                i = int(sha256_hex(f"w:{frame['id']}".encode())[:2], 16) % len(BOOLEANS)
                ans[BOOLEANS[i]] = not ans[BOOLEANS[i]]
            rc = frame.get("mask_interior_rc")
            if rc is None:
                ans["cube_point"] = [500, 500]
            else:                                 # pixel centre -> normalized [y, x]
                ans["cube_point"] = [round((rc[0] + 0.5) * 1000 / frame["image_hw"][0]),
                                     round((rc[1] + 0.5) * 1000 / frame["image_hw"][1])]
        if kind == "c1_point_out_of_range":
            ans["cube_point"] = [-1, -1]          # an honest "no cube here" point: a schema violation of the point
        return ans

    def __call__(self, body: bytes):
        req = json.loads(body)
        items = req.get("input")
        if (set(req) != {"model", "input", "generation_config"} or req["model"] != MODEL
                or req["generation_config"] != GENERATION_CONFIG or not isinstance(items, list) or len(items) != 2
                or items[1] != {"type": "text", "text": TEXT}
                or set(items[0]) != {"type", "data", "mime_type"} or items[0]["type"] != "image"
                or items[0]["mime_type"] != "image/png"):
            raise TransportError("mock: request differs from the fixed model/text/config")
        png = base64.b64decode(items[0]["data"])
        frame, variant = self.by_sha[sha256_hex(png)]
        key = (frame["id"], variant)
        n = self.attempts[key] = self.attempts.get(key, 0) + 1
        kind = self.force.get(key) or self.plan(frame["id"], variant, self.seed)
        self.calls.append((frame["id"], variant, kind, n))
        if self.clock is not None:
            lat = self.next_latency if self.next_latency is not None else self.latency_s
            self.next_latency = None
            self.clock.advance(lat)
        if kind == "timeout":
            raise TransportTimeout("mock timeout")
        if kind == "http_500":
            return 500, _err(500, "INTERNAL", "mock internal error"), {}
        if kind == "http_400":
            return 400, _err(400, "INVALID_ARGUMENT", "mock invalid argument"), {}
        if kind == "http_429_always":
            return 429, _err(429, "RESOURCE_EXHAUSTED", "mock quota"), {"retry_after": "1"}
        if kind == "http_429_long":
            return 429, _err(429, "RESOURCE_EXHAUSTED", "mock quota", retry_delay="600s"), {}
        if n == 1 and kind == "http_429_then_ok":
            return 429, _err(429, "RESOURCE_EXHAUSTED", "mock quota", retry_delay="1.5s"), {"retry_after": "2"}
        if n == 1 and kind == "http_503_then_ok":
            return 503, _err(503, "UNAVAILABLE", "mock overloaded"), {}
        if n == 1 and kind == "conn_error_then_ok":
            raise TransportConnectionError("mock connection reset before any response")
        if kind == "bad_body":
            return 200, b"<html>mock gateway page</html>", {}
        if kind == "no_output":
            return 200, _interaction(None), {}
        if kind == "prose":
            return 200, _interaction("I'm sorry, but I can't help with that request."), {}
        if kind == "malformed_json":
            return 200, _interaction('{"cube_point": [500, 500], "lifted_clear": tru'), {}
        if kind == "schema_violation":
            return 200, _interaction('{"cube_point": [500], "lifted_clear": "yes"}'), {}
        text = json.dumps(self._answer(frame, variant, kind))
        if kind == "fenced_json":
            text = "```json\n" + text + "\n```"
        return 200, _interaction(text, usage=(kind != "no_usage")), {}


# ------------------------------------------------------------------ response check

def validate_answer(obj) -> dict:
    """The schema check: an object with cube_point = two finite numbers in [0, 1000] (int or float, not bool) and
    the four fields as JSON booleans. Extra keys are ignored. Raises ValueError otherwise."""
    if not isinstance(obj, dict):
        raise ValueError("answer is not a JSON object")
    pt = obj.get("cube_point")
    if not (isinstance(pt, list) and len(pt) == 2):
        raise ValueError("cube_point is not a 2-list")
    for v in pt:
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1000:
            raise ValueError("cube_point value not a finite number in [0, 1000]")
    out = {"cube_point": [float(pt[0]), float(pt[1])]}
    for b in BOOLEANS:
        if not isinstance(obj.get(b), bool):
            raise ValueError(f"{b} is not a JSON boolean")
        out[b] = obj[b]
    return out


def booleans_of(obj) -> dict | None:
    """The four booleans parsed on their own (C1 scoring), whatever cube_point holds: an object whose four fields are
    JSON booleans, else None."""
    if not isinstance(obj, dict) or not all(isinstance(obj.get(b), bool) for b in BOOLEANS):
        return None
    return {b: obj[b] for b in BOOLEANS}


_FENCE = re.compile(r"\A```[ \t]*(?:[Jj][Ss][Oo][Nn])?[ \t]*\r?\n(.*?)\r?\n[ \t]*```\Z", re.S)


def parse_answer_json(text: str):
    """The answer's JSON value: after trimming surrounding whitespace, either one JSON value, or exactly one markdown
    code block (an opening ``` or ```json line, the JSON value, a closing ``` line) with nothing outside it. Anything
    else raises ValueError (non-JSON)."""
    s = text.strip()
    m = _FENCE.match(s)
    if m:
        s = m.group(1)
    return json.loads(s)


def usage_of(resp: dict) -> dict | None:
    """Every top-level field whose name contains 'usage' (the usage field's name is not in the saved docs)."""
    u = {str(k): v for k, v in resp.items() if "usage" in str(k).lower()}
    return u or None


def thought_tokens(usage) -> list:
    """[path, value] for every integer under a key containing 'thought' (e.g. total_thought_tokens,
    thoughtsTokenCount): the evidence that thinking ran, and how much."""
    out = []

    def walk(o, path):
        if isinstance(o, dict):
            for k, v in o.items():
                p = f"{path}.{k}" if path else str(k)
                if "thought" in str(k).lower() and isinstance(v, int) and not isinstance(v, bool):
                    out.append([p, v])
                walk(v, p)
        elif isinstance(o, list):
            for i, v in enumerate(o):
                walk(v, f"{path}[{i}]")
    walk(usage, "")
    return out


def _body_text(body) -> str:
    if body is None:
        return ""
    if isinstance(body, bytes):
        return body.decode("utf-8", errors="replace")
    return str(body)


def answer_text(resp: dict) -> tuple:
    """(text, None), (None, None) when there is no answer, or (None, error) for a malformed envelope: see
    ANSWER_FIELD."""
    steps = resp.get("steps")
    if steps is None:
        text = resp.get("output_text")
        if text is not None and not isinstance(text, str):
            return None, "output_text is not a string"
        return text, None
    if not isinstance(steps, list):
        return None, "steps is not a list"
    texts = []
    for s in steps:
        if not isinstance(s, dict):
            return None, "a step is not an object"
        if s.get("type") != "model_output":
            continue
        content = s.get("content")
        if not isinstance(content, list):
            return None, "a model_output step has no content list"
        for c in content:
            if isinstance(c, dict) and c.get("type") == "text":
                if not isinstance(c.get("text"), str):
                    return None, "a model_output text item is not a string"
                texts.append(c["text"])
    return ("".join(texts) if texts else None), None


def classify(status: int | None, body: bytes | None, *, secret: str | None = None,
             raw_cap: int = RAW_CAP_CALL) -> dict:
    """Outcome of one response from its HTTP status and body: kind 'ok' (with 'answer') or a failure kind. Every
    record carries 'booleans' (the four booleans parsed on their own, or None) for C1 scoring, and the raw body
    (redacted, capped) for audit."""
    rec: dict = {"http_status": status, "booleans": None, "raw": redact(_body_text(body), secret)[:raw_cap],
                 "raw_bytes": len(body or b"")}
    if status != 200:
        msg = None
        try:
            msg = (json.loads(body or b"{}").get("error") or {}).get("message")
        except (ValueError, AttributeError):
            msg = None
        rec.update(kind="http_error", error=redact(msg, secret)[:300] if msg else None)
        return rec
    try:
        resp = json.loads(body)
    except (ValueError, TypeError):
        rec.update(kind="bad_envelope", error="response body is not JSON")
        return rec
    if not isinstance(resp, dict):
        rec.update(kind="bad_envelope", error="response body is not a JSON object")
        return rec
    rec["usage"] = usage_of(resp)
    rec["thought_tokens"] = thought_tokens(rec["usage"]) if rec["usage"] else []
    rec["response_keys"] = sorted(str(k) for k in resp)[:40]
    for k in ("id", "model", "status"):
        if isinstance(resp.get(k), str):
            rec[f"response_{k}"] = resp[k][:200]
    text, err = answer_text(resp)
    if err:
        rec.update(kind="bad_envelope", error=err)
        return rec
    if text is None:
        rec.update(kind="no_answer", error="no model_output text in the response (no answer, a block, or a response "
                                           "shape the API reference does not describe)")
        return rec
    rec["text"] = redact(text, secret)[:2000]
    try:
        obj = parse_answer_json(text)
    except ValueError:
        rec.update(kind="non_json")
        return rec
    rec["booleans"] = booleans_of(obj)
    try:
        rec["answer"] = validate_answer(obj)
    except ValueError as e:
        rec.update(kind="schema_violation", error=str(e))
        return rec
    rec["kind"] = "ok"
    return rec


TRANSPORT_FAILURES = frozenset({"http_error", "transport_error", "timeout"})


def _seconds(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) and f >= 0 else None


def retry_hint_s(meta: dict | None, body: bytes | None, *, wall=time.time) -> float | None:
    """The wait the service asks for: max of the Retry-After header (delta-seconds or an HTTP date) and
    google.rpc.RetryInfo's retryDelay ("30s") in the error body; None when neither is given or parseable."""
    hints = []
    ra = (meta or {}).get("retry_after")
    if ra is not None:
        s = _seconds(str(ra).strip())
        if s is None:
            try:
                dt = email.utils.parsedate_to_datetime(str(ra))
                s = max(0.0, dt.timestamp() - wall())
            except (TypeError, ValueError, IndexError, OverflowError):
                s = None
        if s is not None:
            hints.append(s)
    try:
        err = json.loads(body or b"{}").get("error") or {}
        for d in err.get("details") or []:
            if isinstance(d, dict) and str(d.get("@type", "")).endswith("google.rpc.RetryInfo"):
                rd = str(d.get("retryDelay", ""))
                s = _seconds(rd[:-1]) if rd.endswith("s") else None
                if s is not None:
                    hints.append(s)
    except (ValueError, AttributeError, TypeError):
        pass
    return max(hints) if hints else None


def call_once(transport, png: bytes, *, secret: str | None = None, clock=None, raw_cap: int = RAW_CAP_CALL) -> dict:
    """One request: wall time from just before the POST to the body received; over TIMEOUT_S is a timeout whatever
    came back (except a 429/503, which carries no answer and stays re-sendable). 'retryable' marks the outcomes the
    call policy may re-send (HTTP 429/503, a connection error with no HTTP response)."""
    clock = clock if clock is not None else CLOCK
    body = json.dumps(build_request(png)).encode("ascii")
    t0 = clock()
    try:
        status, content, meta = transport(body)
    except TransportTimeout as e:
        return {"kind": "timeout", "latency_s": None, "error": redact(e, secret)[:300], "retryable": False}
    except TransportConnectionError as e:
        return {"kind": "connection_error", "latency_s": None, "error": redact(e, secret)[:300], "retryable": True}
    except TransportError as e:
        return {"kind": "transport_error", "latency_s": round(clock() - t0, 4), "error": redact(e, secret)[:300],
                "retryable": False}
    wall = clock() - t0
    rec = classify(status, content, secret=secret, raw_cap=raw_cap)
    rec["latency_s"] = round(wall, 4)
    rec["retryable"] = status in RETRYABLE_HTTP
    if rec["retryable"]:
        rec["retry_hint_s"] = retry_hint_s(meta, content)
    elif wall > TIMEOUT_S:
        rec = {"kind": "timeout", "latency_s": round(wall, 4), "late_result_kind": rec.get("kind"),
               "http_status": status, "booleans": None, "retryable": False}
    return rec


class Pacer:
    """At least `gap` seconds between the end of one request and the start of the next. The clock and sleep are the
    module's CLOCK/SLEEP (resolved at call time) unless given (mock mode: a VirtualClock)."""

    def __init__(self, gap: float = MIN_GAP_S, clock=None, sleep=None):
        self.gap = float(gap)
        self._clock, self._sleep = clock, sleep
        self.last_end = None
        self.requests = 0

    def now(self) -> float:
        return (self._clock if self._clock is not None else CLOCK)()

    def _do_sleep(self, s: float) -> None:
        (self._sleep if self._sleep is not None else SLEEP)(s)

    def before_send(self, wait_s: float | None = None) -> float | None:
        """Sleep until max(gap, wait_s) has passed since the previous request ended; returns the gap actually left
        (None for the first request)."""
        need = self.gap if wait_s is None else max(self.gap, float(wait_s))
        if self.last_end is None:
            self.requests += 1
            return None
        due = self.last_end + need
        while True:
            now = self.now()
            if now >= due:
                break
            self._do_sleep(due - now)
        self.requests += 1
        return round(self.now() - self.last_end, 4)

    def after(self) -> None:
        self.last_end = self.now()


ATTEMPT_FIELDS = ("kind", "http_status", "latency_s", "error", "retry_hint_s", "retryable")


def call_with_policy(transport, png: bytes, pacer: Pacer, *, secret: str | None = None, resends: int = MAX_RESENDS,
                     on_attempt=None, raw_cap: int = RAW_CAP_CALL) -> dict:
    """One frame and variant under CALL_POLICY. Returns {"final": the final outcome or None, "attempts": [...],
    "stop": None or why the caller must stop}. final is None only when every attempt was re-sendable (HTTP 429/503 or
    a connection error with no response: no answer was produced) and the re-sends are used up or the requested wait
    exceeds MAX_WAIT_S."""
    attempts = []
    wait = None
    for i in range(resends + 1):
        gap = pacer.before_send(wait)
        rec = call_once(transport, png, secret=secret, clock=pacer.now, raw_cap=raw_cap)
        pacer.after()
        att = {k: rec.get(k) for k in ATTEMPT_FIELDS}
        att.update(attempt=i + 1, gap_before_s=gap, waited_for_s=wait, at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        attempts.append(att)
        if on_attempt is not None:
            on_attempt(att)
        if not rec.get("retryable"):
            return {"final": rec, "attempts": attempts, "stop": None}
        if i == resends:
            return {"final": None, "attempts": attempts,
                    "stop": f"{resends} re-sends used up (last: {rec['kind']} {rec.get('http_status')})"}
        hint = rec.get("retry_hint_s")
        wait = max(MIN_GAP_S, hint if hint is not None else BACKOFF_S[min(i, len(BACKOFF_S) - 1)])
        if wait > MAX_WAIT_S:
            return {"final": None, "attempts": attempts,
                    "stop": f"the service asks for a {wait:.0f} s wait (> {MAX_WAIT_S:.0f} s)"}
    raise AssertionError("unreachable")
