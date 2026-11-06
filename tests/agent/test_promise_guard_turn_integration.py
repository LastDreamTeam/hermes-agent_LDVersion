"""Final-response seam tests for the high-precision promise guard."""

from types import SimpleNamespace

from agent.context_compressor import ContextCompressor
from agent.conversation_loop import (
    _CODEX_ACK_CONTINUATION_NUDGE,
    _PROMISE_GUARD_CONTINUATION_NUDGE,
)
from agent.turn_final_response import finish_text_response


class _Agent:
    def __init__(self, *, mode="enforce", legacy=False):
        self._promise_guard_mode = mode
        self._intent_ack_continuation = legacy
        self._stall_guards = True
        self.valid_tool_names = {"terminal"}
        self.model = "fixture"
        self.provider = "fixture"
        self.api_mode = "chat_completions"
        self.quiet_mode = True
        self.session_id = "offline-fixture"
        self.events = []

    def _strip_think_blocks(self, text):
        return text

    def _has_content_after_think_block(self, text):
        return bool(text)

    def _emit_pending_fallback_notice(self):
        pass

    def _clear_status_buffer(self):
        pass

    def _build_assistant_message(self, assistant, reason):
        return {"role": "assistant", "content": assistant.content, "finish_reason": reason}

    def _emit_interim_assistant_message(self, message):
        self.events.append(message["content"])

    def _looks_like_codex_intermediate_ack(self, **_kwargs):
        return bool(self._intent_ack_continuation)

    def _flush_messages_to_session_db(self, *_args):
        pass


def _finish(monkeypatch, agent, text, *, count=0, messages=None):
    monkeypatch.setattr(
        "agent.turn_final_response.apply_stop_gates",
        lambda *_args, **_kwargs: SimpleNamespace(
            pending_verification_response=None,
            pending_verification_response_previewed=False,
            continue_turn=False,
        ),
    )
    messages = [] if messages is None else messages
    verdict = finish_text_response(
        agent,
        assistant_message=SimpleNamespace(content=text, tool_calls=None),
        response=None,
        finish_reason="stop",
        messages=messages,
        api_messages=[],
        conversation_history=[],
        api_call_count=1,
        user_message="fixture request",
        active_system_prompt="",
        final_response=None,
        _turn_exit_reason=None,
        _preflight_compression_blocked=False,
        codex_ack_continuations=count,
        truncated_response_parts=[],
        length_continue_retries=0,
        _pending_verification_response=None,
        _pending_verification_response_previewed=False,
    )
    return verdict, messages


def test_ready_promise_continues_internally_without_explicit_interim_delivery(monkeypatch):
    agent = _Agent()
    verdict, messages = _finish(monkeypatch, agent, "我这就发。")
    assert verdict.action == "continue"
    assert verdict.codex_ack_continuations == 1
    assert messages[-1]["content"] == _PROMISE_GUARD_CONTINUATION_NUDGE
    assert agent.events == []


def test_waiting_for_approval_blocks_both_stall_and_legacy_paths(monkeypatch):
    agent = _Agent(legacy=True)
    verdict, messages = _finish(monkeypatch, agent, "你点头，我这就发。")
    assert verdict.action == "break"
    assert verdict.codex_ack_continuations == 0
    assert not any(m.get("content") in {
        _PROMISE_GUARD_CONTINUATION_NUDGE, _CODEX_ACK_CONTINUATION_NUDGE,
    } for m in messages)
    assert agent.events == []


def test_english_conditional_blocks_legacy_ack_bypass(monkeypatch):
    agent = _Agent(legacy=True)
    verdict, messages = _finish(monkeypatch, agent, "If you agree, I'll check the logs.")
    assert verdict.action == "break"
    assert not any(m.get("content") == _CODEX_ACK_CONTINUATION_NUDGE for m in messages)


def test_shadow_mode_records_no_behavioral_continuation(monkeypatch):
    agent = _Agent(mode="shadow")
    verdict, messages = _finish(monkeypatch, agent, "我这就发。")
    assert verdict.action == "break"
    assert verdict.codex_ack_continuations == 0
    assert len(messages) == 1
    assert messages[0]["content"] == "我这就发。"


def test_budget_is_cumulative_across_non_guard_stop_gate_continuations(monkeypatch):
    agent = _Agent()
    messages = []
    first, messages = _finish(monkeypatch, agent, "我这就发。", messages=messages)
    second, messages = _finish(
        monkeypatch, agent, "我马上检查日志。",
        count=first.codex_ack_continuations, messages=messages,
    )
    third, messages = _finish(
        monkeypatch, agent, "我立即发送通知。",
        count=second.codex_ack_continuations, messages=messages,
    )
    assert (first.action, second.action, third.action) == ("continue", "continue", "break")
    assert (
        first.codex_ack_continuations,
        second.codex_ack_continuations,
        third.codex_ack_continuations,
    ) == (1, 2, 2)
    assert sum(m.get("content") == _PROMISE_GUARD_CONTINUATION_NUDGE for m in messages) == 2


def test_compressor_recognizes_both_historical_and_new_internal_nudges():
    for content in (_CODEX_ACK_CONTINUATION_NUDGE, _PROMISE_GUARD_CONTINUATION_NUDGE):
        assert ContextCompressor._is_synthetic_compression_user_turn(
            {"role": "user", "content": content}
        )
