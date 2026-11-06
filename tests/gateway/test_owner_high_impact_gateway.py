"""End-to-end gateway contracts for owner exact-phrase approvals."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.run_turn_runner import TurnRunner
from gateway.session import SessionSource
from tools import approval as A
from tools.approval_gateway_wait import _ApprovalEntry


SESSION = "agent:main:qqbot:dm:owner-chat"


@pytest.fixture(autouse=True)
def _clean_approval_state(monkeypatch):
    monkeypatch.setattr(
        A, "is_owner_high_impact_principal_allowed",
        lambda p: len(tuple(p or ())) == 8 and tuple(p or ())[4] == "owner-1",
    )
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


def _source(user_id="owner-1"):
    return SessionSource(
        platform=Platform.QQBOT,
        chat_id="owner-chat",
        chat_type="dm",
        user_id=user_id,
        profile="default",
    )


def _event(text, *, user_id="owner-1", internal=False, allow_gateway_control=True):
    source = _source(user_id)
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=source,
        user_id=user_id,
        message_id="message-1",
        internal=internal,
        allow_gateway_control=allow_gateway_control,
    )


def _queue(principal):
    entry = _ApprovalEntry(
        {
            "request_id": "owner-request-1",
            "approval_kind": A.OWNER_HIGH_IMPACT_APPROVAL_KIND,
            "command": "<didi_create_order>",
            "pattern_key": "plugin_rule:owner-high-impact:didi_create_order:fresh",
            "pattern_keys": ["plugin_rule:owner-high-impact:didi_create_order:fresh"],
            "description": "paid ride",
            "allow_session": False,
            "allow_permanent": False,
        },
        policy=A.OWNER_HIGH_IMPACT_APPROVAL_KIND,
        origin_principal=principal,
        timeout_s=A.OWNER_HIGH_IMPACT_TIMEOUT_S,
    )
    with A._lock:
        A._gateway_queues.setdefault(SESSION, []).append(entry)
    return entry


def test_busy_router_accepts_only_exact_authorized_origin_once():
    owner = _source()
    principal = A.owner_high_impact_principal_from_source(owner)
    entry = _queue(principal)

    runner = object.__new__(GatewayRunner)
    adapter = SimpleNamespace(resume_typing_for_chat=lambda _chat_id: None)
    runner._adapter_for_source = lambda _source: adapter
    runner._is_user_authorized = lambda _source: True
    runner._send_busy_reply = AsyncMock()

    # No normalization: surrounding whitespace is ordinary conversational text.
    assert not asyncio.run(
        runner._route_plaintext_approval_while_busy(
            _event(A.OWNER_HIGH_IMPACT_PHRASE + " "), SESSION
        )
    )
    assert not entry.event.is_set()

    # The exact bytes are consumed but cannot cross an immutable sender identity.
    assert asyncio.run(
        runner._route_plaintext_approval_while_busy(
            _event(A.OWNER_HIGH_IMPACT_PHRASE, user_id="someone-else"), SESSION
        )
    )
    assert not entry.event.is_set()

    # Synthetic/internal text is likewise consumed without granting approval.
    assert asyncio.run(
        runner._route_plaintext_approval_while_busy(
            _event(A.OWNER_HIGH_IMPACT_PHRASE, internal=True), SESSION
        )
    )
    assert not entry.event.is_set()

    assert asyncio.run(
        runner._route_plaintext_approval_while_busy(
            _event(A.OWNER_HIGH_IMPACT_PHRASE), SESSION
        )
    )
    assert entry.event.is_set()
    assert entry.result == A.OWNER_HIGH_IMPACT_CHOICE

    # The consumed request cannot be replayed by repeating the phrase.
    assert not asyncio.run(
        runner._route_plaintext_approval_while_busy(
            _event(A.OWNER_HIGH_IMPACT_PHRASE), SESSION
        )
    )


def test_owner_notifier_uses_phrase_text_and_never_button_transport():
    class ButtonCapableAdapter:
        def __init__(self):
            self.sent = []
            self.button_calls = 0

        def pause_typing_for_chat(self, _chat_id):
            return None

        async def send(self, chat_id, content, *, metadata=None):
            self.sent.append((chat_id, content, metadata))
            return SimpleNamespace(success=True, message_id="prompt-1")

        async def send_exec_approval(self, **_kwargs):
            self.button_calls += 1
            return SimpleNamespace(success=True, message_id="button-1")

    class ImmediateFuture:
        def result(self, timeout=None):
            return SimpleNamespace(success=True, message_id="prompt-1")

    adapter = ButtonCapableAdapter()
    ctx = SimpleNamespace(
        _status_adapter=adapter,
        _status_chat_id="owner-chat",
        _status_thread_metadata={"thread_id": "thread-1"},
        stream_consumer_holder=[],
        session_key=SESSION,
    )
    turn = TurnRunner(None, ctx)

    def run_scheduled(coro, _log_message, loop=None):
        asyncio.run(coro)
        return ImmediateFuture()

    turn._schedule = run_scheduled
    turn._approval_notify_sync(
        {
            "request_id": "owner-request-123456789",
            "approval_kind": A.OWNER_HIGH_IMPACT_APPROVAL_KIND,
            "required_phrase": A.OWNER_HIGH_IMPACT_PHRASE,
            "command": "<didi_create_order>",
            "description": "paid ride",
            "allow_session": False,
            "allow_permanent": False,
        }
    )

    assert adapter.button_calls == 0
    assert len(adapter.sent) == 1
    chat_id, text, metadata = adapter.sent[0]
    assert chat_id == "owner-chat"
    assert A.OWNER_HIGH_IMPACT_PHRASE in text
    assert "/approve" in text and "无效" in text
    assert metadata["is_approval_prompt"] is True
    assert metadata["_interim_send"] is True
