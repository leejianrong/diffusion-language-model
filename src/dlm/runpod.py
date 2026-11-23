"""Minimal RunPod REST v2 client (https://api.runpod.io/v2). Stdlib only.

The API key is resolved lazily by reference (an environment variable, else runpodctl's own
config) and is only ever placed in the Authorization header. It is never logged, never in an
exception message, and API error bodies are redacted before they surface, because a create
response echoes the container's variables back.
"""

import json
import os
import time
import tomllib
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol

API_BASE = "https://api.runpod.io/v2"
DEFAULT_CONFIG = Path.home() / ".runpod" / "config.toml"
SECRET_WORDS = ("key", "token", "secret", "password", "auth", "credential", "cookie")
Transport = Callable[[str, str, dict[str, str], bytes | None, float], tuple[int, bytes]]


class ApiError(RuntimeError):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"RunPod API {status}: {detail}")
        self.status = status


def redact(obj: Any) -> Any:
    """Replace the value of every secret-looking key, recursively."""
    if isinstance(obj, dict):
        return {
            k: "[REDACTED]" if any(w in str(k).lower() for w in SECRET_WORDS) else redact(v)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [redact(v) for v in obj]
    return obj


def load_api_key(
    environ: Mapping[str, str] | None = None,
    config: Path = DEFAULT_CONFIG,
) -> str:
    src = os.environ if environ is None else environ
    if src.get("RUNPOD_API_KEY"):
        return src["RUNPOD_API_KEY"]
    if config.exists():
        with config.open("rb") as f:
            key = tomllib.load(f).get("apikey")
        if key:
            return str(key)
    raise RuntimeError("no RunPod API key: set RUNPOD_API_KEY or configure runpodctl")


class _Readable(Protocol):
    def read(self, n: int = ..., /) -> bytes: ...


def read_bounded(
    stream: _Readable,
    *,
    secs: float,
    clock: Callable[[], float] = time.monotonic,
    max_bytes: int = 4_000_000,
    partial: bool = False,
) -> bytes:
    """Read a response under a TOTAL deadline.

    A socket timeout only bounds each recv, so an endpoint that keeps sending (a log stream) never
    trips it. `partial=True` returns what arrived instead of raising, for streaming endpoints.
    """
    end = clock() + secs
    buf = bytearray()
    read = getattr(stream, "read1", stream.read)
    while True:
        try:
            chunk = read(65536)
        except TimeoutError:
            if partial:
                return bytes(buf)
            raise
        if not chunk:
            return bytes(buf)
        buf += chunk
        if len(buf) >= max_bytes or clock() >= end:
            if partial:
                return bytes(buf)
            raise TimeoutError("response did not finish within the deadline")


def _urlopen(
    method: str, url: str, headers: dict[str, str], body: bytes | None, timeout: float
) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, read_bounded(r, secs=timeout)
    except urllib.error.HTTPError as e:
        return e.code, read_bounded(e, secs=timeout)


class Client:
    def __init__(
        self,
        key_fn: Callable[[], str] = load_api_key,
        transport: Transport = _urlopen,
        base: str = API_BASE,
        timeout: float = 30.0,
    ) -> None:
        self._key_fn, self._transport, self._base, self._timeout = key_fn, transport, base, timeout

    def _call(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        headers = {"Authorization": f"Bearer {self._key_fn()}", "Content-Type": "application/json"}
        data = json.dumps(body).encode() if body is not None else None
        status, raw = self._transport(method, self._base + path, headers, data, self._timeout)
        try:
            parsed: Any = json.loads(raw) if raw else None
        except ValueError:
            parsed = None
        if status >= 400:
            shown = json.dumps(redact(parsed))[:400] if parsed is not None else "(non-JSON body)"
            raise ApiError(status, shown)
        return parsed

    def list_pods(self) -> list[dict[str, Any]]:
        out = self._call("GET", "/pods")
        if isinstance(out, dict):
            out = out.get("pods") or out.get("items") or out.get("data") or []
        return list(out or [])

    def create_pod(self, body: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = self._call("POST", "/pods", body)
        return out

    def get_pod(self, pod_id: str) -> dict[str, Any]:
        out: dict[str, Any] = self._call("GET", f"/pods/{pod_id}")
        return out

    def terminate(self, pod_id: str) -> None:
        """Idempotent: a pod that is already gone (404) counts as terminated."""
        try:
            self._call("DELETE", f"/pods/{pod_id}")
        except ApiError as e:
            if e.status != 404:
                raise

    def logs_text(self, pod_id: str, secs: float = 8.0) -> str:
        """What the (possibly streaming) log endpoint sends in `secs` seconds, never longer."""
        req = urllib.request.Request(
            f"{self._base}/pods/{pod_id}/logs",
            headers={"Authorization": f"Bearer {self._key_fn()}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=min(5.0, secs)) as r:
                return read_bounded(r, secs=secs, max_bytes=200_000, partial=True).decode(
                    errors="replace"
                )
        except urllib.error.HTTPError as e:
            raise ApiError(e.code, "log request refused") from None

    def cpu_catalog(self) -> Any:
        return self._call("GET", "/catalog/cpus")
