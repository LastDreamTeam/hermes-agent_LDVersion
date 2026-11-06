"""The finite chat -q/-Q path must not advertise async completion delivery."""

from unittest.mock import MagicMock

import pytest

import cli as cli_mod
from gateway import session_context


def test_chat_q_declares_stateless_before_claim(monkeypatch):
    session_context.reset_session_vars()
    monkeypatch.delenv("HERMES_SINGLE_QUERY_SESSION", raising=False)
    fake_cli = MagicMock()
    fake_cli._claim_active_session.return_value = False

    try:
        with pytest.raises(SystemExit) as exc:
            cli_mod._run_single_query_mode(
                fake_cli, query="finite turn", image=None, quiet=True, oneshot=False
            )
        assert exc.value.code == 1
        assert session_context.async_delivery_supported() is False
        fake_cli._claim_active_session.assert_called_once_with("cli", stderr=True)
    finally:
        session_context.reset_session_vars()


def test_chat_q_is_stateless_even_with_inherited_messaging_route(monkeypatch):
    session_context.reset_session_vars()
    monkeypatch.delenv("HERMES_SINGLE_QUERY_SESSION", raising=False)
    monkeypatch.setenv(
        "HERMES_SESSION_KEY",
        "agent:main:qqbot:dm:example-owner-route",
    )
    fake_cli = MagicMock()
    fake_cli._claim_active_session.return_value = False

    try:
        with pytest.raises(SystemExit):
            cli_mod._run_single_query_mode(
                fake_cli, query="finite turn", image=None, quiet=True, oneshot=False
            )
        assert session_context.async_delivery_supported() is False
    finally:
        session_context.reset_session_vars()
