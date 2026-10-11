"""Recovery replay policy for MCP ``annotations.idempotentHint``.

An explicit ``false`` means the remote side effect may have committed before
an auth/session/transport response was lost.  Hermes may heal the connection,
but must return an uncertain outcome instead of issuing the RPC again.
``true`` and a missing hint intentionally retain the historical one-retry
behaviour for compatibility.
"""

import json
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tools import mcp_tool
from tools import mcp_tool_handlers as handlers
from tools import mcp_tool_loop
from tools import mcp_tool_registration as registration


@pytest.fixture(autouse=True)
def _clean_policy_state():
    with patch.dict(mcp_tool._tool_idempotent_hints, {}, clear=True), \
         patch.dict(mcp_tool._server_error_counts, {}, clear=True), \
         patch.dict(mcp_tool._server_breaker_opened_at, {}, clear=True):
        yield


def _tool(name, hint=...):
    annotations = None if hint is ... else SimpleNamespace(idempotentHint=hint)
    return SimpleNamespace(
        name=name, description="", inputSchema={}, annotations=annotations)


def test_discovery_records_false_true_and_missing_idempotency_hints():
    tools = [_tool("create_order", False), _tool("get_order", True), _tool("legacy_tool")]

    registration._record_tool_trust_metadata("ride", {}, tools)

    assert mcp_tool._tool_idempotent_hints["ride"] == {
        "create_order": False,
        "get_order": True,
        "legacy_tool": None,
    }


def test_annotation_reader_supports_both_sdk_spellings_and_cache_dicts():
    assert registration._annotation_idempotent_hint(
        SimpleNamespace(annotations=SimpleNamespace(idempotentHint=False))) is False
    assert registration._annotation_idempotent_hint(
        SimpleNamespace(annotations=SimpleNamespace(idempotent_hint=True))) is True
    assert registration._annotation_idempotent_hint(
        SimpleNamespace(annotations={"idempotentHint": False})) is False
    assert registration._annotation_idempotent_hint(
        SimpleNamespace(annotations={"idempotent_hint": True})) is True
    assert registration._annotation_idempotent_hint(
        SimpleNamespace(annotations={"idempotentHint": "false"})) is None
    assert registration._annotation_idempotent_hint(
        SimpleNamespace(annotations=None)) is None


def test_schema_cache_preserves_explicit_hints_without_inventing_missing_false():
    server = SimpleNamespace(
        _tools=[_tool("create_order", False), _tool("get_order", True), _tool("legacy_tool")],
        initialize_result=None,
        session=SimpleNamespace(),
        _list_cache_meta={},
    )
    config = {"tools": {"resources": False, "prompts": False}}

    with patch("tools.mcp_schema_cache.write_cache_entry") as write:
        registration._write_schema_cache("ride", server, config, lambda _name: True)

    rows = {row["name"]: row for row in write.call_args.kwargs["tools"]}
    assert rows["create_order"]["annotations"]["idempotentHint"] is False
    assert rows["get_order"]["annotations"]["idempotentHint"] is True
    assert "idempotentHint" not in rows["legacy_tool"]["annotations"]


def _install_session_expiry_server(name):
    """A connected server whose reconnect event publishes a fresh session."""
    mcp_tool_loop._ensure_mcp_loop()
    server = MagicMock()
    server.name = name
    ready = threading.Event()
    ready.set()

    class _Ready:
        def is_set(self):
            return ready.is_set()

        def clear(self):
            ready.clear()

        def set(self):
            ready.set()

    reconnected = threading.Event()

    class _Reconnect:
        def set(self):
            reconnected.set()
            old_session = server.session
            fresh_session = MagicMock()
            fresh_session.call_tool = old_session.call_tool
            server.session = fresh_session
            ready.set()

    server._ready = _Ready()
    server._reconnect_event = _Reconnect()
    server.session = MagicMock()
    mcp_tool._servers[name] = server
    return server, reconnected


@pytest.mark.parametrize(
    "hint, expected_calls, expect_success",
    [(False, 1, False), (True, 2, True), (None, 2, True)],
    ids=["explicit-false", "explicit-true", "missing"],
)
def test_session_recovery_reconnects_but_only_replays_true_or_missing(
        monkeypatch, tmp_path, hint, expected_calls, expect_success):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    name, tool_name = f"ride-{hint}", "create_order"
    server, reconnected = _install_session_expiry_server(name)
    calls = {"count": 0}

    async def _call_tool(*_args, **_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("Invalid or expired session")
        result = MagicMock()
        result.is_error = False
        result.content = [MagicMock(type="text", text="ok")]
        result.structured_content = None
        return result

    server.session.call_tool = _call_tool
    if hint is not None:
        mcp_tool._tool_idempotent_hints.setdefault(name, {})[tool_name] = hint

    try:
        payload = json.loads(handlers._make_tool_handler(name, tool_name, 10.0)({"x": "y"}))
        assert reconnected.is_set()
        assert calls["count"] == expected_calls
        if expect_success:
            assert payload == {"result": "ok"}
        else:
            assert payload["outcome_uncertain"] is True
            assert payload["automatic_replay"] is False
            assert payload["connection_recovered"] is True
            assert "Query authoritative status" in payload["error"]
    finally:
        mcp_tool._servers.pop(name, None)


def test_operator_no_replay_survives_lazy_discovery_metadata_overwrite(
        monkeypatch, tmp_path):
    """An upstream manifest cannot undo a plugin's hard no-replay request."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    name, tool_name = "ride-operator-floor", "create_order"
    server, reconnected = _install_session_expiry_server(name)
    calls = {"count": 0}

    async def _call_tool(*_args, **_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("Invalid or expired session")
        result = MagicMock()
        result.is_error = False
        result.content = [MagicMock(type="text", text="duplicated")]
        result.structured_content = None
        return result

    server.session.call_tool = _call_tool
    # Simulate lazy discovery overwriting an earlier local policy pin with an
    # incorrect upstream annotation immediately before handler policy is read.
    mcp_tool._tool_idempotent_hints.setdefault(name, {})[tool_name] = True

    try:
        payload = json.loads(handlers._make_tool_handler(
            name, tool_name, 10.0, allow_replay=False
        )({"x": "y"}))
        assert reconnected.is_set()
        assert calls["count"] == 1
        assert payload["outcome_uncertain"] is True
        assert payload["automatic_replay"] is False
    finally:
        mcp_tool._servers.pop(name, None)


def test_handler_plaintext_transport_response_reconnects_without_replay(
        monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    name, tool_name = "ride-plaintext", "create_order"
    server, reconnected = _install_session_expiry_server(name)
    calls = {"count": 0}

    async def _call_tool(*_args, **_kwargs):
        calls["count"] += 1
        raise RuntimeError("Unexpected Content-Type text/plain from MCP response")

    server.session.call_tool = _call_tool
    mcp_tool._tool_idempotent_hints.setdefault(name, {})[tool_name] = False

    try:
        payload = json.loads(handlers._make_tool_handler(name, tool_name, 10.0)({"x": "y"}))
        assert reconnected.is_set()
        assert calls["count"] == 1
        assert payload["outcome_uncertain"] is True
        assert payload["automatic_replay"] is False
        assert payload["connection_recovered"] is True
    finally:
        mcp_tool._servers.pop(name, None)


def test_nonidempotent_auth_recovery_does_not_invoke_retry_call(monkeypatch):
    retry_calls = {"count": 0}
    server = SimpleNamespace(_reconnect_event=SimpleNamespace(set=lambda: None))

    def _retry():
        retry_calls["count"] += 1
        return '{"result":"duplicated"}'

    monkeypatch.setattr(handlers, "_is_auth_error", lambda _exc: True)
    monkeypatch.setattr(handlers._loop, "_run_on_mcp_loop", lambda *_a, **_kw: True)
    monkeypatch.setattr(handlers, "_lookup_reconnectable_server", lambda *_a, **_kw: server)
    reconnect = MagicMock(return_value=True)
    monkeypatch.setattr(handlers._loop, "_signal_reconnect_and_wait", reconnect)

    payload = json.loads(handlers._handle_auth_error_and_retry(
        "ride", RuntimeError("401"), _retry, "tools/call create_order",
        allow_replay=False, tool_name="create_order"))

    reconnect.assert_called_once()
    assert retry_calls["count"] == 0
    assert payload["outcome_uncertain"] is True
    assert payload["automatic_replay"] is False
    assert payload["connection_recovered"] is True


def test_nonidempotent_stdio_recovery_does_not_invoke_retry_call(monkeypatch):
    retry_calls = {"count": 0}
    server = SimpleNamespace(_reconnect_event=SimpleNamespace(set=lambda: None))

    def _retry():
        retry_calls["count"] += 1
        return '{"result":"duplicated"}'

    monkeypatch.setattr(handlers, "_lookup_reconnectable_server", lambda *_a, **_kw: server)
    monkeypatch.setattr(handlers, "_mcp_loop_running", lambda: True)
    reconnect = MagicMock(return_value=True)
    monkeypatch.setattr(handlers._loop, "_signal_reconnect_and_wait", reconnect)

    payload = json.loads(handlers._handle_stdio_child_exited_and_retry(
        "ride", handlers._StdioChildExited("dead"), _retry, "tools/call create_order",
        allow_replay=False, tool_name="create_order"))

    reconnect.assert_called_once()
    assert retry_calls["count"] == 0
    assert payload["outcome_uncertain"] is True
    assert payload["automatic_replay"] is False
    assert payload["connection_recovered"] is True


def test_nonidempotent_stdio_predispatch_death_allows_one_safe_attempt(monkeypatch):
    retry_calls = {"count": 0}
    server = SimpleNamespace(_reconnect_event=SimpleNamespace(set=lambda: None))

    def _retry():
        retry_calls["count"] += 1
        return '{"result":"placed once"}'

    monkeypatch.setattr(handlers, "_lookup_reconnectable_server", lambda *_a, **_kw: server)
    monkeypatch.setattr(handlers, "_mcp_loop_running", lambda: True)
    reconnect = MagicMock(return_value=True)
    monkeypatch.setattr(handlers._loop, "_signal_reconnect_and_wait", reconnect)

    payload = json.loads(handlers._handle_stdio_child_exited_and_retry(
        "ride", handlers._StdioChildExited("dead before call", dispatched=False),
        _retry, "tools/call create_order", allow_replay=False, tool_name="create_order"))

    reconnect.assert_called_once()
    assert retry_calls["count"] == 1
    assert payload == {"result": "placed once"}


def test_nonidempotent_stdio_safe_attempt_failure_is_not_replayed_again(monkeypatch):
    retry_calls = {"count": 0}
    server = SimpleNamespace(_reconnect_event=SimpleNamespace(set=lambda: None))

    def _retry():
        retry_calls["count"] += 1
        raise TimeoutError("response lost after the safe attempt was dispatched")

    monkeypatch.setattr(handlers, "_lookup_reconnectable_server", lambda *_a, **_kw: server)
    monkeypatch.setattr(handlers, "_mcp_loop_running", lambda: True)
    monkeypatch.setattr(handlers._loop, "_signal_reconnect_and_wait", MagicMock(return_value=True))

    payload = json.loads(handlers._handle_stdio_child_exited_and_retry(
        "ride", handlers._StdioChildExited("dead before call", dispatched=False),
        _retry, "tools/call create_order", allow_replay=False, tool_name="create_order"))

    assert retry_calls["count"] == 1
    assert payload["outcome_uncertain"] is True
    assert payload["automatic_replay"] is False


def test_nonidempotent_session_expiry_without_reconnector_is_still_uncertain(monkeypatch):
    retry = MagicMock(return_value='{"result":"duplicated"}')
    monkeypatch.setattr(handlers, "_lookup_reconnectable_server", lambda *_a, **_kw: None)

    payload = json.loads(handlers._handle_session_expired_and_retry(
        "ride", RuntimeError("Invalid or expired session"), retry,
        "tools/call create_order", allow_replay=False, tool_name="create_order"))

    retry.assert_not_called()
    assert payload["outcome_uncertain"] is True
    assert payload["automatic_replay"] is False
    assert payload["connection_recovered"] is False


def test_existing_reconnect_teardown_is_not_signalled_a_second_time(monkeypatch):
    retry = MagicMock(return_value='{"result":"duplicated"}')
    server = SimpleNamespace(_reconnect_event=SimpleNamespace(set=lambda: None))
    monkeypatch.setattr(handlers, "_lookup_reconnectable_server", lambda *_a, **_kw: server)
    monkeypatch.setattr(handlers, "_mcp_loop_running", lambda: True)
    reconnect = MagicMock(return_value=True)
    monkeypatch.setattr(handlers._loop, "_signal_reconnect_and_wait", reconnect)

    payload = json.loads(handlers._handle_non_idempotent_transport_without_replay(
        "ride", RuntimeError("MCP call was aborted by a reconnect teardown; retry on rebuilt session"),
        retry, "tools/call create_order", tool_name="create_order"))

    reconnect.assert_not_called()
    retry.assert_not_called()
    assert payload["outcome_uncertain"] is True
    assert payload["recovery_in_progress"] is True
    assert payload["connection_recovered"] is False


def test_httpx_transport_errors_and_http_5xx_are_ambiguous():
    httpx = pytest.importorskip("httpx")
    failures = [
        httpx.ReadTimeout(""),
        httpx.ConnectTimeout(""),
        httpx.ReadError(""),
        httpx.ConnectError(""),
    ]
    sdk_httpx = mcp_tool.sdk_httpx()
    if sdk_httpx is not None and sdk_httpx is not httpx:
        failures.extend([sdk_httpx.ReadTimeout(""), sdk_httpx.ConnectError("")])
    upstream_502 = RuntimeError("upstream failure")
    upstream_502.response = SimpleNamespace(status_code=502)
    failures.append(upstream_502)

    assert all(handlers._is_ambiguous_transport_failure(failure) for failure in failures)


@pytest.mark.parametrize(
    "failure",
    [
        TimeoutError("tool response timed out"),
        ConnectionResetError("peer reset after request write"),
        RuntimeError("Unexpected Content-Type text/plain from MCP response"),
    ],
    ids=["timeout", "connection-reset", "text-plain"],
)
def test_nonidempotent_ambiguous_transport_reconnects_without_replay(
        monkeypatch, failure):
    calls = {"count": 0}
    server = SimpleNamespace(_reconnect_event=SimpleNamespace(set=lambda: None))

    def _retry():
        calls["count"] += 1
        return '{"result":"duplicated"}'

    monkeypatch.setattr(handlers, "_lookup_reconnectable_server", lambda *_a, **_kw: server)
    monkeypatch.setattr(handlers, "_mcp_loop_running", lambda: True)
    reconnect = MagicMock(return_value=True)
    monkeypatch.setattr(handlers._loop, "_signal_reconnect_and_wait", reconnect)

    payload = json.loads(handlers._handle_non_idempotent_transport_without_replay(
        "ride", failure, _retry, "tools/call create_order", tool_name="create_order"))

    reconnect.assert_called_once()
    assert calls["count"] == 0
    assert payload["outcome_uncertain"] is True
    assert payload["automatic_replay"] is False
    assert payload["connection_recovered"] is True
