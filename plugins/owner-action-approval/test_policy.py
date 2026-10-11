from contextlib import contextmanager
import importlib.util
import json
import pathlib
import sys
import time
import types
import unittest
from unittest.mock import patch


MODULE_PATH = pathlib.Path(__file__).with_name("__init__.py")


class OwnerActionApprovalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("owner_action_approval", MODULE_PATH)
        cls.module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(cls.module)
        cls.module._BROKER_SCHEMAS = (
            MODULE_PATH.parents[2]
            / "staging"
            / "credential-broker"
            / "hermes-owner-tool-schemas.json"
        )
        cls.tools_package = types.ModuleType("tools")
        cls.tools_package.__path__ = []
        cls.approval = types.ModuleType("tools.approval")
        cls.approval._current_owner_high_impact_principal = lambda: (
            "qqbot", "owner-chat", "dm", "", "owner-1", "", "", "default"
        )
        cls.approval.is_owner_high_impact_principal_allowed = lambda _principal: True
        cls.approval.owner_high_impact_principal_sha256 = lambda _principal: "a" * 64
        cls.tools_package.approval = cls.approval

    def setUp(self):
        self.broker_calls = []
        self._prepare_counter = 0
        with self.module._pending_lock:
            self.module._pending_by_call.clear()
            self.module._pending_by_rule.clear()

        def fake_broker(_socket_path, payload, *, timeout):
            self.broker_calls.append((payload, timeout))
            op = payload["op"]
            if op == "prepare":
                self._prepare_counter += 1
                return {
                    "ok": True,
                    "request_id": f"{self._prepare_counter:032x}",
                    "nonce": "N" * 43,
                    "expires_at": int(time.time()) + 300,
                }
            if op in {"confirm", "cancel"}:
                return {"ok": True, "state": "confirmed" if op == "confirm" else "cancelled"}
            if op == "execute":
                return {"ok": True, "state": "consumed", "result": {"fake": True}}
            if op == "read":
                return {"ok": True, "result": {"fake": True}}
            raise AssertionError(op)

        self.broker_patch = patch.object(
            self.module, "_broker_request", side_effect=fake_broker
        )
        self.broker_patch.start()

    def tearDown(self):
        self.broker_patch.stop()
        with self.module._pending_lock:
            self.module._pending_by_call.clear()
            self.module._pending_by_rule.clear()

    @contextmanager
    def _session_context(
        self,
        *,
        profile="default",
        session_key="owner-session-key",
        active_profile="default",
        context_error=None,
        active_profile_error=None,
    ):
        approval = self.approval
        gateway = types.ModuleType("gateway")
        session_context = types.ModuleType("gateway.session_context")

        def get_session_env(name, default=""):
            if context_error is not None:
                raise context_error
            return {
                "HERMES_SESSION_PROFILE": profile,
                "HERMES_SESSION_KEY": session_key,
                "HERMES_SESSION_PLATFORM": "qqbot",
                "HERMES_SESSION_CHAT_ID": "owner-chat",
                "HERMES_SESSION_CHAT_TYPE": "dm",
                "HERMES_SESSION_USER_ID": "owner-1",
            }.get(name, default)

        session_context.get_session_env = get_session_env
        gateway.session_context = session_context

        hermes_cli = types.ModuleType("hermes_cli")
        profiles = types.ModuleType("hermes_cli.profiles")

        def get_active_profile_name():
            if active_profile_error is not None:
                raise active_profile_error
            return active_profile

        profiles.get_active_profile_name = get_active_profile_name
        hermes_cli.profiles = profiles
        with patch.dict(
            sys.modules,
            {
                "tools": self.tools_package,
                "tools.approval": approval,
                "gateway": gateway,
                "gateway.session_context": session_context,
                "hermes_cli": hermes_cli,
                "hermes_cli.profiles": profiles,
            },
        ), patch.object(
            approval,
            "is_owner_high_impact_principal_allowed",
            return_value=True,
        ), patch.object(
            approval,
            "owner_high_impact_principal_sha256",
            return_value="a" * 64,
        ):
            yield

    def _call_as_owner(self, tool_name, args, **context_kwargs):
        with self._session_context(**context_kwargs):
            return self.module._on_pre_tool_call(
                tool_name, args, session_id="owner-agent-session",
                tool_call_id="tool-call-1",
            )

    def _registered_context(self):
        class FakeContext:
            def __init__(self):
                self.hooks = []
                self.tools = {}

            def register_hook(self, name, callback):
                self.hooks.append((name, callback))

            def register_tool(self, *, name, schema, handler, **kwargs):
                self.tools[name] = (schema, handler, kwargs)

        context = FakeContext()
        self.module.register(context)
        return context

    @contextmanager
    def _observability_context(
        self, *, session_id="owner-agent-session", tool_call_id="tool-call-1"
    ):
        approval_context = types.ModuleType("tools.approval_context")
        approval_context.get_current_observability_session_id = lambda: session_id
        approval_context.get_current_tool_call_id = lambda: tool_call_id
        with patch.dict(
            sys.modules, {"tools.approval_context": approval_context}
        ):
            yield

    def test_private_reads_are_preflight_gated_but_amap_remains_broad(self):
        with self._session_context(profile="daidai"):
            for name in (
                "mcp__didi__taxi_estimate",
                "mcp__netease_music__netease_user_library",
            ):
                with self.subTest(name=name):
                    result = self.module._on_pre_tool_call(
                        name, {}, session_id="public-session"
                    )
                    self.assertEqual("block", result["action"])
            self.assertIsNone(
                self.module._on_pre_tool_call(
                    "mcp__amap__maps_weather", {"city": "上海"},
                    session_id="public-session",
                )
            )

    def test_private_read_preflight_allows_only_approved_owner(self):
        with self._session_context(profile="default"):
            result = self.module._on_pre_tool_call(
                "mcp__didi__taxi_query_order", {"order_id": "synthetic"},
                session_id="owner-agent-session",
            )
        self.assertIsNone(result)

        with self._session_context(profile="default"), patch.object(
            self.approval, "owner_high_impact_principal_sha256", return_value=""
        ):
            result = self.module._on_pre_tool_call(
                "mcp__netease_music__netease_daily_recommendations", {},
                session_id="ordinary-group-session",
            )
        self.assertEqual("block", result["action"])

    def test_private_didi_and_netease_reads_require_owner_profile(self):
        context = self._registered_context()
        private_tools = (
            ("mcp__didi__taxi_query_order", {"order_id": "synthetic-order"}),
            ("mcp__netease_music__netease_user_library", {"action": "profile"}),
        )
        for name, arguments in private_tools:
            with self.subTest(name=name):
                self.broker_calls.clear()
                _, handler, _ = context.tools[name]
                with self._session_context(profile="daidai"), self._observability_context():
                    result = json.loads(
                        handler(arguments, session_id="owner-agent-session")
                    )
                self.assertFalse(result["ok"])
                self.assertIn("BLOCKED", result["error"])
                self.assertEqual([], self.broker_calls)

    def test_private_reads_reject_ordinary_user_in_default_profile(self):
        context = self._registered_context()
        approval = self.approval
        _, handler, _ = context.tools[
            "mcp__netease_music__netease_daily_recommendations"
        ]
        with self._session_context(profile="default"), patch.object(
            approval, "owner_high_impact_principal_sha256", return_value=""
        ), self._observability_context():
            result = json.loads(handler({}, session_id="ordinary-group-session"))
        self.assertFalse(result["ok"])
        self.assertIn("BLOCKED", result["error"])
        self.assertEqual([], self.broker_calls)

    def test_private_reads_reach_broker_for_approved_owner_only(self):
        context = self._registered_context()
        for name, arguments, provider in (
            (
                "mcp__didi__taxi_get_driver_location",
                {"order_id": "synthetic-order"},
                "didi",
            ),
            (
                "mcp__netease_music__netease_user_library",
                {"action": "profile"},
                "netease",
            ),
        ):
            with self.subTest(name=name):
                self.broker_calls.clear()
                _, handler, _ = context.tools[name]
                with self._session_context(profile="default"), self._observability_context():
                    result = json.loads(
                        handler(arguments, session_id="owner-agent-session")
                    )
                self.assertTrue(result["ok"])
                payload = self.broker_calls[-1][0]
                self.assertEqual("read", payload["op"])
                self.assertEqual(provider, payload["provider"])
                self.assertEqual("a" * 64, payload["principal_sha256"])

    def test_amap_safe_read_remains_available_to_daidai_and_ordinary_user(self):
        context = self._registered_context()
        approval = self.approval
        _, handler, _ = context.tools["mcp__amap__maps_weather"]
        with self._session_context(profile="daidai"), patch.object(
            approval, "owner_high_impact_principal_sha256", return_value=""
        ), self._observability_context(
            session_id="ordinary-group-session", tool_call_id="ordinary-tool-call"
        ):
            result = json.loads(
                handler({"city": "上海"}, session_id="ordinary-group-session")
            )
        self.assertTrue(result["ok"])
        payload = self.broker_calls[-1][0]
        self.assertEqual("read", payload["op"])
        self.assertEqual("amap", payload["provider"])
        self.assertRegex(payload["principal_sha256"], r"^[0-9a-f]{64}$")

    def test_didi_order_is_gated_and_phone_is_masked(self):
        result = self._call_as_owner(
            "didi_create_order",
            {
                "product_category": "express",
                "estimate_trace_id": "trace-123",
                "caller_car_phone": "13800138000",
            },
        )
        self.assertEqual("approve", result["action"])
        self.assertIn("CONFIRM_DELETE", result["message"])
        self.assertIn("尾号8000", result["message"])
        self.assertNotIn("13800138000", result["message"])

    def test_plugin_owned_didi_order_names_are_gated(self):
        first = self._call_as_owner(
            "didi_create_order", {"product_category": "express", "estimate_trace_id": "t"}
        )
        self.assertEqual(
            "approve",
            first["action"],
        )
        self.module._on_post_approval_response(
            pattern_key="plugin_rule:" + first["rule_key"], choice="deny",
            session_id="owner-agent-session", tool_call_id="tool-call-1",
        )
        self.assertEqual(
            "approve",
            self._call_as_owner("didi_cancel_order", {"order_id": "o"})["action"],
        )

    def test_netease_write_is_gated(self):
        legacy = self._call_as_owner(
            "mcp__netease_music__netease_playlist_create", {"name": "通勤"}
        )
        self.assertEqual("block", legacy["action"])
        self.assertIn("旧的直连", legacy["message"])

        grouped = self._call_as_owner(
            "netease_playlist_write", {"action": "create", "name": "通勤"}
        )
        self.assertEqual("approve", grouped["action"])
        self.assertIn("netease_playlist_write", grouped["message"])

    def test_non_main_profile_is_blocked(self):
        with self._session_context(profile="daidai"):
            result = self.module._on_pre_tool_call(
                "mcp__netease_music__netease_song_like",
                {"song_id": "song"},
                session_id="owner-agent-session",
            )
        self.assertEqual("block", result["action"])
        self.assertIn("daidai", result["message"])

    def test_paired_but_non_owner_identity_is_blocked(self):
        approval = self.approval
        with self._session_context(), patch.object(
            approval,
            "owner_high_impact_principal_sha256",
            return_value="",
        ):
            result = self.module._on_pre_tool_call(
                "didi_create_order",
                {"product_category": "express", "estimate_trace_id": "t"},
                session_id="owner-agent-session",
            )
        self.assertEqual("block", result["action"])
        self.assertIn("固定主人身份", result["message"])

    def test_each_call_has_fresh_non_reusable_rule_key(self):
        with self._session_context(), patch.object(
            self.module.secrets, "token_hex", side_effect=["a", "b"]
        ):
            first = self.module._on_pre_tool_call(
                "didi_cancel_order", {"order_id": "1"}, session_id="owner-agent-session",
                tool_call_id="tool-call-a",
            )
            self.module._on_post_approval_response(
                pattern_key="plugin_rule:" + first["rule_key"], choice="deny",
                session_id="owner-agent-session", tool_call_id="tool-call-a",
            )
            second = self.module._on_pre_tool_call(
                "didi_cancel_order", {"order_id": "1"}, session_id="owner-agent-session",
                tool_call_id="tool-call-b",
            )
        self.assertNotEqual(first["rule_key"], second["rule_key"])

    def test_main_profile_is_explicitly_allowed(self):
        result = self._call_as_owner(
            "didi_cancel_order", {"order_id": "o"}, profile="main"
        )
        self.assertEqual("approve", result["action"])

    def test_blank_gateway_profile_requires_explicit_active_profile_resolution(self):
        resolved = self._call_as_owner(
            "didi_cancel_order", {"order_id": "o"}, profile="", active_profile="default"
        )
        self.assertEqual("approve", resolved["action"])

        unresolved = self._call_as_owner(
            "didi_cancel_order", {"order_id": "o"}, profile="", active_profile=""
        )
        self.assertEqual("block", unresolved["action"])

    def test_session_context_errors_fail_closed(self):
        for context_kwargs in (
            {"context_error": RuntimeError("context unavailable")},
            {"profile": "", "active_profile_error": RuntimeError("profile unavailable")},
            {"session_key": ""},
        ):
            with self.subTest(context_kwargs=context_kwargs):
                result = self._call_as_owner(
                    "didi_create_order",
                    {"product_category": "express", "estimate_trace_id": "t"},
                    **context_kwargs,
                )
                self.assertEqual("block", result["action"])

        with self._session_context(profile="default"):
            missing_hook_session = self.module._on_pre_tool_call(
                "didi_cancel_order", {"order_id": "o"}, session_id=""
            )
        self.assertEqual("block", missing_hook_session["action"])

    def test_secret_like_fields_are_redacted(self):
        text = self.module._action_summary(
            "netease_note_publish", {"title": "公开内容", "access_token": "do-not-print"}
        )
        self.assertIn("title=公开内容", text)
        self.assertIn("access_token=[已隐藏]", text)
        self.assertNotIn("do-not-print", text)

    def test_register_exposes_fixed_didi_and_guarded_netease_mutations(self):
        ctx = self._registered_context()
        expected_reads = {
            f"mcp__didi__{name}" for name in self.module._DIDI_READS
        } | {
            f"mcp__amap__{name}" for name in self.module._AMAP_READS
        } | {
            f"mcp__netease_music__{name}" for name in self.module._NETEASE_READS
        }
        expected_writes = {
            "didi_create_order",
            "didi_cancel_order",
            *self.module._NETEASE_GROUP_WRITES,
        }
        self.assertEqual(expected_reads | expected_writes, set(ctx.tools))
        self.assertEqual(
            {"pre_tool_call", "post_approval_response"},
            {name for name, _callback in ctx.hooks},
        )

        approval_context = types.ModuleType("tools.approval_context")
        approval_context.get_current_observability_session_id = lambda: "owner-agent-session"
        approval_context.get_current_tool_call_id = lambda: "tool-call-1"
        _, handler, _ = ctx.tools["didi_create_order"]
        arguments = {"product_category": "express", "estimate_trace_id": "trace"}
        with self._session_context(), patch.dict(
            sys.modules, {"tools.approval_context": approval_context}
        ):
            proposed = self.module._on_pre_tool_call(
                "didi_create_order", arguments,
                session_id="owner-agent-session", tool_call_id="tool-call-1",
            )
            self.module._on_post_approval_response(
                pattern_key="plugin_rule:" + proposed["rule_key"],
                choice="owner_phrase_once",
                session_id="owner-agent-session", tool_call_id="tool-call-1",
            )
            result = handler(arguments)
        self.assertIn('"ok":true', result)
        self.assertEqual(["prepare", "confirm", "execute"], [call[0]["op"] for call in self.broker_calls])

        self.broker_calls.clear()
        _, read_handler, _ = ctx.tools["mcp__amap__maps_weather"]
        read_result = read_handler({"city": "上海"}, session_id="owner-agent-session")
        self.assertIn('"ok":true', read_result)
        self.assertEqual("read", self.broker_calls[-1][0]["op"])


if __name__ == "__main__":
    unittest.main()
