"""Regression tests for the owner's exact, one-shot high-impact approval lane."""

import time

import pytest

from tools import approval as A
from tools.approval_gateway_wait import _ApprovalEntry


SESSION = "agent:main:qqbot:dm:chat-1"
PRINCIPAL = ("qqbot", "chat-1", "dm", "", "owner-1", "", "", "")

@pytest.fixture(autouse=True)
def _clean_approval_state(monkeypatch):
    monkeypatch.setattr(A, "is_owner_high_impact_principal_allowed", lambda p: tuple(p or ()) == PRINCIPAL)
    with A._lock:
        A._gateway_queues.clear()
        A._gateway_notify_cbs.clear()
        A._session_approved.clear()
        A._session_yolo.clear()
        A._permanent_approved.clear()
    yield
    with A._lock:
        entries = [entry for queue in A._gateway_queues.values() for entry in queue]
        A._gateway_queues.clear()
        A._gateway_notify_cbs.clear()
    for entry in entries:
        entry.result = "deny"
        entry.event.set()


def _queue(*, request_id="req-1", principal=PRINCIPAL, timeout_s=300):
    entry = _ApprovalEntry(
        {
            "request_id": request_id,
            "approval_kind": A.OWNER_HIGH_IMPACT_APPROVAL_KIND,
            "command": "<didi_create_order>",
            "pattern_key": "plugin_rule:owner-high-impact:didi_create_order:x",
            "pattern_keys": ["plugin_rule:owner-high-impact:didi_create_order:x"],
            "description": "paid ride",
            "allow_session": False,
            "allow_permanent": False,
        },
        policy=A.OWNER_HIGH_IMPACT_APPROVAL_KIND,
        origin_principal=principal,
        timeout_s=timeout_s,
    )
    with A._lock:
        A._gateway_queues.setdefault(SESSION, []).append(entry)
    return entry


@pytest.mark.parametrize("choice", ["once", "session", "always", "approve", "owner_phrase_once"])
def test_generic_positive_resolvers_cannot_consume_high_impact(choice):
    entry = _queue()
    assert A.resolve_gateway_approval(SESSION, choice) == 0
    assert A.resolve_gateway_approval(SESSION, choice, resolve_all=True) == 0
    assert A.resolve_gateway_approval(SESSION, choice, request_id="req-1") == 0
    assert A.has_blocking_owner_high_impact_approval(SESSION)
    assert not entry.event.is_set()


def test_generic_deny_can_cancel_high_impact():
    entry = _queue()
    assert A.resolve_gateway_approval(SESSION, "deny", request_id="req-1") == 1
    assert entry.result == "deny"
    assert entry.event.is_set()


@pytest.mark.parametrize(
    "phrase",
    ["yes", "ok", "👍", "CONFIRM_DELETE ", " CONFIRM_DELETE", "CONFIRM_DELETE\n", "/approve", "confirm_delete"],
)
def test_exact_phrase_rejects_normalized_or_generic_variants(phrase):
    entry = _queue()
    assert A.resolve_owner_high_impact_approval(SESSION, phrase, "req-1", PRINCIPAL) == 0
    assert not entry.event.is_set()


def test_exact_phrase_requires_request_session_and_immutable_principal():
    entry = _queue()
    assert A.resolve_owner_high_impact_approval(SESSION, "CONFIRM_DELETE", "", PRINCIPAL) == 0
    assert A.resolve_owner_high_impact_approval("other", "CONFIRM_DELETE", "req-1", PRINCIPAL) == 0
    assert A.resolve_owner_high_impact_approval(SESSION, "CONFIRM_DELETE", "wrong", PRINCIPAL) == 0
    other_user = ("qqbot", "chat-1", "dm", "", "other", "", "", "")
    assert A.resolve_owner_high_impact_approval(SESSION, "CONFIRM_DELETE", "req-1", other_user) == 0
    assert not entry.event.is_set()


def test_exact_phrase_is_single_use_and_consumes_only_named_request():
    first = _queue(request_id="req-1")
    second = _queue(request_id="req-2")
    assert A.resolve_owner_high_impact_approval(SESSION, "CONFIRM_DELETE", "req-1", PRINCIPAL) == 1
    assert first.result == A.OWNER_HIGH_IMPACT_CHOICE
    assert first.event.is_set()
    assert not second.event.is_set()
    assert A.resolve_owner_high_impact_approval(SESSION, "CONFIRM_DELETE", "req-1", PRINCIPAL) == 0
    assert A.get_pending_owner_high_impact_approval(SESSION)["request_id"] == "req-2"


def test_expired_phrase_request_fails_closed_and_is_removed():
    entry = _queue(timeout_s=0)
    entry.expires_at = time.monotonic() - 1
    assert A.resolve_owner_high_impact_approval(SESSION, "CONFIRM_DELETE", "req-1", PRINCIPAL) == 0
    assert entry.event.is_set()
    assert not A.has_blocking_owner_high_impact_approval(SESSION)


def test_generic_fifo_does_not_jump_over_high_impact():
    protected = _queue()
    ordinary = _ApprovalEntry({"request_id": "ordinary", "command": "echo", "pattern_keys": ["x"]})
    with A._lock:
        A._gateway_queues[SESSION].append(ordinary)
    assert A.resolve_gateway_approval(SESSION, "once") == 0
    assert not ordinary.event.is_set()
    assert not protected.event.is_set()
    # Explicitly targeted ordinary approvals still work without touching the protected one.
    assert A.resolve_gateway_approval(SESSION, "once", request_id="ordinary") == 1
    assert ordinary.event.is_set()
    assert A.has_blocking_owner_high_impact_approval(SESSION)


def test_owner_principal_requires_platform_chat_and_user():
    assert not A._owner_high_impact_principal_from_values(platform="qqbot", chat_id="", user_id="u")
    assert not A._owner_high_impact_principal_from_values(platform="qqbot", chat_id="c", user_id="")
    assert A._owner_high_impact_principal_from_values(platform="qqbot", chat_id="c", user_id="u")


def test_core_fails_closed_for_legacy_netease_mutation_names():
    # Representative old direct endpoints from three mutation families.  These
    # used to be invisible to the core fallback when plugin dispatch failed.
    assert A.is_owner_high_impact_tool_name("netease_playlist_create")
    assert A.is_owner_high_impact_tool_name("mcp__netease_music__netease_comment_post")
    assert A.is_owner_high_impact_tool_name("netease_queue_clear")
    assert not A.is_owner_high_impact_tool_name("mcp__netease_music__netease_user_info")


def test_exact_phrase_rejects_a_stable_but_non_owner_principal(monkeypatch):
    entry = _queue()
    monkeypatch.setattr(A, "is_owner_high_impact_principal_allowed", lambda _p: False)
    assert A.resolve_owner_high_impact_approval(
        SESSION, A.OWNER_HIGH_IMPACT_PHRASE, "req-1", PRINCIPAL
    ) == 0
    assert not entry.event.is_set()


def test_request_gate_rejects_paired_non_owner(monkeypatch):
    monkeypatch.setattr(A, "get_current_session_key", lambda default="": SESSION)
    monkeypatch.setattr(A, "_is_gateway_approval_context", lambda: True)
    monkeypatch.setattr(A, "_is_cron_approval_context", lambda: False)
    monkeypatch.setattr(A, "_is_single_query_approval_context", lambda: False)
    monkeypatch.setattr(A, "_is_unattended_platform_approval_context", lambda: False)
    monkeypatch.setattr(A, "_get_session_platform", lambda: "qqbot")
    monkeypatch.setattr(A, "_current_owner_high_impact_principal", lambda: PRINCIPAL)
    monkeypatch.setattr(A, "is_owner_high_impact_principal_allowed", lambda _p: False)
    import gateway.session_context as session_context
    monkeypatch.setattr(session_context, "session_is_messaging_surface", lambda: True)

    result = A.request_owner_high_impact_approval(
        "didi_create_order", "paid ride",
        rule_key="owner-high-impact:didi_create_order:fresh",
    )
    assert result["approved"] is False
    assert "不在主人付费/账号写入授权名单" in result["message"]


def test_request_gate_ignores_yolo_and_requires_attended_messaging(monkeypatch):
    monkeypatch.setattr(A, "_YOLO_MODE_FROZEN", True)
    monkeypatch.setattr(A, "get_current_session_key", lambda default="": SESSION)
    monkeypatch.setattr(A, "_is_gateway_approval_context", lambda: False)
    result = A.request_owner_high_impact_approval(
        "didi_create_order", "paid ride",
        rule_key="owner-high-impact:didi_create_order:fresh",
    )
    assert result["approved"] is False
    assert "当前不是可交互" in result["message"]


@pytest.mark.parametrize(
    ("gateway_choice", "approved"),
    [("once", False), (A.OWNER_HIGH_IMPACT_CHOICE, True)],
)
def test_generic_request_entrypoint_ignores_yolo_and_accepts_only_dedicated_choice(
    monkeypatch, gateway_choice, approved,
):
    # Exercise the public generic resolver entrypoint: owner-prefixed rules must
    # be rerouted before its normal YOLO/mode-off/cache shortcuts can run.
    monkeypatch.setattr(A, "_YOLO_MODE_FROZEN", True)
    A.enable_session_yolo(SESSION)
    monkeypatch.setattr(A.approval_context, "_get_approval_mode", lambda: "off")
    monkeypatch.setattr(A, "get_current_session_key", lambda default="": SESSION)
    monkeypatch.setattr(A, "_is_gateway_approval_context", lambda: True)
    monkeypatch.setattr(A, "_is_cron_approval_context", lambda: False)
    monkeypatch.setattr(A, "_is_single_query_approval_context", lambda: False)
    monkeypatch.setattr(A, "_is_unattended_platform_approval_context", lambda: False)
    monkeypatch.setattr(A, "_get_session_platform", lambda: "qqbot")
    monkeypatch.setattr(A, "_current_owner_high_impact_principal", lambda: PRINCIPAL)
    monkeypatch.setattr(A, "_gateway_notify_cb", lambda _session: lambda _data: None)
    import gateway.session_context as session_context
    monkeypatch.setattr(session_context, "session_is_messaging_surface", lambda: True)

    seen = {}

    def fake_wait(*args, **kwargs):
        seen.update(kwargs)
        return {"resolved": True, "choice": gateway_choice, "reason": None}

    monkeypatch.setattr(A, "_await_gateway_decision", fake_wait)
    result = A.request_tool_approval(
        "didi_create_order", "paid ride",
        rule_key="owner-high-impact:didi_create_order:fresh",
    )
    assert result["approved"] is approved
    assert seen["policy"] == A.OWNER_HIGH_IMPACT_APPROVAL_KIND
    assert seen["origin_principal"] == PRINCIPAL
    assert seen["timeout_s"] == 300.0
