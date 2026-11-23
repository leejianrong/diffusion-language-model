import json
from typing import Any

import pytest

from dlm.runpod import ApiError, Client, load_api_key, read_bounded, redact

SECRET = "rpa_SUPERSECRET123"


class Recorder:
    def __init__(self, status: int = 200, body: Any = None) -> None:
        self.status, self.body = status, body
        self.calls: list[tuple[str, str, dict[str, str], bytes | None]] = []

    def __call__(
        self, method: str, url: str, headers: dict[str, str], data: bytes | None, t: float
    ) -> tuple[int, bytes]:
        self.calls.append((method, url, headers, data))
        return self.status, json.dumps(self.body).encode() if self.body is not None else b""


def client(rec: Recorder) -> Client:
    return Client(key_fn=lambda: SECRET, transport=rec)


def test_redact_hides_secret_looking_keys_at_any_depth() -> None:
    out = redact(
        {"id": "p", "env": {"RUNPOD_API_KEY": SECRET, "MODE": "x"}, "list": [{"token": "t"}]}
    )
    assert out["env"] == {"RUNPOD_API_KEY": "[REDACTED]", "MODE": "x"}
    assert out["list"] == [{"token": "[REDACTED]"}] and out["id"] == "p"


def test_key_only_goes_in_the_authorization_header() -> None:
    rec = Recorder(body=[])
    client(rec).list_pods()
    _, url, headers, data = rec.calls[0]
    assert headers["Authorization"] == f"Bearer {SECRET}"
    assert SECRET not in url and data is None and url.startswith("https://api.runpod.io/v2/")


def test_error_bodies_are_redacted_and_never_contain_the_key() -> None:
    rec = Recorder(500, {"error": "boom", "echo": {"env": {"RUNPOD_API_KEY": SECRET}}})
    with pytest.raises(ApiError) as ei:
        client(rec).create_pod({"name": "x", "env": {"RUNPOD_API_KEY": SECRET}})
    assert SECRET not in str(ei.value) and "[REDACTED]" in str(ei.value) and ei.value.status == 500


def test_terminate_is_idempotent_for_a_missing_pod_but_not_for_other_errors() -> None:
    client(Recorder(404, {"error": "not found"})).terminate("gone")
    client(Recorder(204)).terminate("ok")
    with pytest.raises(ApiError):
        client(Recorder(403, {"error": "forbidden"})).terminate("x")


def test_list_pods_accepts_a_bare_list_or_a_wrapped_one() -> None:
    assert client(Recorder(body=[{"id": "a"}])).list_pods() == [{"id": "a"}]
    assert client(Recorder(body={"pods": [{"id": "b"}]})).list_pods() == [{"id": "b"}]


def test_api_key_comes_from_env_then_config_and_missing_is_an_error(tmp_path) -> None:  # type: ignore[no-untyped-def]
    cfg = tmp_path / "config.toml"
    cfg.write_text("apikey = 'from-config'\n")
    assert load_api_key({"RUNPOD_API_KEY": "from-env"}, cfg) == "from-env"
    assert load_api_key({}, cfg) == "from-config"
    with pytest.raises(RuntimeError, match="no RunPod API key"):
        load_api_key({}, tmp_path / "missing.toml")


class _Endless:
    """A log stream that never ends: every read returns data, so a socket timeout never fires."""

    def read(self, n: int = 65536, /) -> bytes:
        return b"x" * 10


class _Finite:
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks

    def read(self, n: int = 65536, /) -> bytes:
        return self.chunks.pop(0) if self.chunks else b""


def test_read_bounded_stops_an_endless_stream_at_the_deadline() -> None:
    ticks = iter(range(1000))
    with pytest.raises(TimeoutError):
        read_bounded(_Endless(), secs=5, clock=lambda: next(ticks))
    ticks = iter(range(1000))
    got = read_bounded(_Endless(), secs=5, clock=lambda: next(ticks), partial=True)
    assert got.startswith(b"xxxx") and 0 < len(got) < 1000


def test_read_bounded_returns_a_finite_body_whole() -> None:
    assert read_bounded(_Finite([b"ab", b"cd"]), secs=5) == b"abcd"
