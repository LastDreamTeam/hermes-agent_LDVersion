"""Owner-only business tools for the designated owner host.

Didi and ordinary NetEase mutations execute directly for the authenticated
owner.  The broker still binds every call to the exact gateway process,
principal, session, tool call and canonical arguments, and keeps non-idempotent
calls single-shot.  Only true deletion operations use a small per-turn token:
the owner includes ``CONFIRM_DELETE`` once in the original turn.  This deliberately
does not use Hermes' native approval queue.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import re
import secrets
import socket
import stat
import tempfile
import threading
import time
from typing import Any


logger = logging.getLogger(__name__)


_OWNER_PROFILES = frozenset({"default", "main"})

_BROKER_READ_SOCKET = "/run/hermes-owner-broker/read.sock"
_BROKER_WRITE_SOCKET = "/run/hermes-owner-broker/write.sock"
_BROKER_SCHEMAS = Path("/etc/hermes-owner-broker/tool-schemas.json")
_AGENT_PLAYLIST_REGISTRY = Path(
    "/home/hermes/.hermes/private/netease-music/agent-managed-playlists.json"
)
_BROKER_PROTOCOL_VERSION = 1
_BROKER_FRAME_LIMIT = 1_048_576


_DIDI_WRITES = {
    "didi_create_order",
    "didi_cancel_order",
    "taxi_create_order",
    "taxi_cancel_order",
}

_NETEASE_GROUP_WRITES = {
    "netease_auth_maintenance",
    "netease_song_like_write",
    "netease_playlist_write",
    "netease_podcast_write",
    "netease_note_write",
    "netease_comment_write",
    "netease_cloud_write",
    "netease_ai_song_write",
    "netease_playback_device",
}

# Keep the earlier one-command names in the defensive hook as well.  They are
# not model-visible in the current wrapper, but a future configuration drift
# must not turn an old write name into an unapproved path.
_NETEASE_LEGACY_WRITES = {
    "netease_song_like",
    "netease_song_dislike",
    "netease_playlist_create",
    "netease_playlist_add",
    "netease_playlist_remove",
    "netease_playlist_reorder",
    "netease_playlist_update_name",
    "netease_playlist_update_desc",
    "netease_playlist_update_tags",
    "netease_playlist_update_cover",
    "netease_podcast_subscribe",
    "netease_podcast_unsubscribe",
    "netease_podcast_voicelist_create",
    "netease_podcast_voicelist_edit",
    "netease_podcast_voice_upload",
    "netease_podcast_voice_update",
    "netease_note_publish",
    "netease_note_delete",
    "netease_comment_post",
    "netease_comment_reply",
    "netease_cloud_download_file",
    "netease_cloud_update_sort",
    "netease_cloudupload_upload",
    "netease_cloudupload_upload_file",
    "netease_cloudupload_stop",
    "netease_aisong_daily_login",
    "netease_aisong_generate",
    "netease_aisong_generate_lyric",
    "netease_play",
    "netease_pause",
    "netease_resume",
    "netease_stop",
    "netease_next",
    "netease_prev",
    "netease_seek",
    "netease_volume",
    "netease_queue_add",
    "netease_queue_clear",
}

_NETEASE_WRITES = _NETEASE_GROUP_WRITES | _NETEASE_LEGACY_WRITES

_DIDI_READS = frozenset({
    "maps_textsearch", "maps_regeocode", "maps_place_around",
    "maps_direction_driving", "maps_direction_transit",
    "maps_direction_walking", "maps_direction_bicycling", "taxi_estimate",
    "taxi_generate_ride_app_link", "taxi_query_order", "taxi_get_driver_location",
})
_AMAP_READS = frozenset({
    "maps_around_search", "maps_direction_bicycling", "maps_direction_driving",
    "maps_direction_transit_integrated", "maps_direction_walking", "maps_distance",
    "maps_geo", "maps_ip_location", "maps_regeocode", "maps_search_detail",
    "maps_text_search", "maps_weather", "maps_schema_personal_map",
    "maps_schema_navi", "maps_schema_take_taxi",
})
_NETEASE_READS = frozenset({
    "netease_search_song", "netease_search_playlist", "netease_search_album",
    "netease_search_all", "netease_search_media", "netease_user_library",
    "netease_playlist_detail", "netease_playlist_read",
    "netease_daily_recommendations", "netease_recommend", "netease_catalog_read",
    "netease_cloud_read", "netease_podcast_read", "netease_community_read",
    "netease_ai_song_read",
})
_READ_TOOLS = {
    "didi": _DIDI_READS,
    "amap": _AMAP_READS,
    "netease": _NETEASE_READS,
}
_WRITE_TOOLS = {
    "didi": frozenset({"taxi_create_order", "taxi_cancel_order"}),
    "netease": frozenset(_NETEASE_GROUP_WRITES),
}

_grant_lock = threading.Lock()
_DELETE_CONFIRM_TOKEN = "CONFIRM_DELETE"
_TASK_GRANT_TTL_SECONDS = 6 * 60 * 60
_delete_grants: dict[tuple[str, str, str], float] = {}
_used_confirmation_origins: dict[str, float] = {}
_agent_playlist_registry_lock = threading.Lock()

_WRITE_ALIAS: dict[str, tuple[str, str]] = {
    "didi_create_order": ("didi", "taxi_create_order"),
    "didi_cancel_order": ("didi", "taxi_cancel_order"),
    **{name: ("netease", name) for name in _NETEASE_GROUP_WRITES},
}

_SENSITIVE_KEY = re.compile(
    r"(?:secret|token|cookie|password|authorization|private.?key|api.?key|credential)",
    re.IGNORECASE,
)
_PHONE_KEY = re.compile(r"phone|mobile", re.IGNORECASE)


def _base_tool_name(tool_name: str) -> str:
    """Return the provider tool name from either a bare or mcp__server__tool name."""
    return str(tool_name or "").rsplit("__", 1)[-1]


def _compact(value: Any, *, limit: int = 120) -> str:
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    else:
        text = str(value)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _masked_value(key: str, value: Any) -> str:
    if _SENSITIVE_KEY.search(key):
        return "[已隐藏]"
    text = _compact(value)
    if _PHONE_KEY.search(key):
        digits = "".join(ch for ch in text if ch.isdigit())
        return f"尾号{digits[-4:]}" if len(digits) >= 4 else "[已隐藏]"
    return text


def _action_summary(base_name: str, args: Any) -> str:
    safe_args = args if isinstance(args, dict) else {}
    preferred = {
        "taxi_create_order": ("product_category", "estimate_trace_id", "caller_car_phone"),
        "taxi_cancel_order": ("order_id", "reason"),
    }.get(base_name)
    keys = list(preferred or safe_args.keys())
    parts = []
    for key in keys:
        if key not in safe_args:
            continue
        parts.append(f"{key}={_masked_value(key, safe_args[key])}")
        if len(parts) >= 8:
            break
    detail = "；".join(parts) if parts else "无可显示参数"
    return f"动作={base_name}；{detail}"


def _is_guarded(tool_name: str) -> bool:
    base_name = _base_tool_name(tool_name)
    return base_name in _DIDI_WRITES or base_name in _NETEASE_WRITES


def _is_forbidden_qualified_write(tool_name: str) -> bool:
    """Reject every MCP-qualified account mutation path.

    Model-visible writes owned by this plugin are intentionally bare names.  A
    qualified name is therefore a direct MCP route (including a renamed server)
    and must never inherit the friendly alias that leads to the broker handler.
    """

    value = str(tool_name or "")
    base_name = _base_tool_name(value)
    if value.startswith("mcp__netease_music__"):
        # The fixed read set is the only direct NetEase surface.  Fail closed
        # for newly added/drifted account tools rather than guessing whether a
        # previously unseen operation mutates the private account.
        return base_name not in _NETEASE_READS
    if value.startswith("mcp__didi__"):
        return base_name not in _DIDI_READS
    return value.startswith("mcp__") and _is_guarded(value)


def _is_private_account_read(tool_name: str) -> bool:
    """Identify account-backed reads without confusing Didi/Amap name overlap."""

    value = str(tool_name or "")
    if value.startswith("mcp__didi__"):
        return _base_tool_name(value) in _DIDI_READS
    if value.startswith("mcp__netease_music__"):
        return _base_tool_name(value) in _NETEASE_READS
    return False


def _private_owner_context_allowed(session_id: Any) -> bool:
    """Check the root-managed owner identity before any private read handler."""

    owner_context, _profile = _owner_session_profile(session_id)
    if not owner_context:
        return False
    principal_sha256, _, _ = _current_binding(session_id=session_id)
    return bool(principal_sha256)


def _safe_broker_error(code: Any, *, write: bool = False) -> str:
    clean = re.sub(r"[^a-z0-9_-]", "", str(code or "internal_error").lower())[:80]
    if write:
        return (
            "外部高影响操作没有得到可确认的执行结果，系统不会自动重试。"
            f"错误代码={clean or 'internal_error'}；请先查询订单/账号状态后再决定。"
        )
    return f"外部只读能力暂时不可用（错误代码={clean or 'internal_error'}）。"


def _broker_request(socket_path: str, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
    """Send exactly one bounded frame to the local credential broker.

    The fixed AF_UNIX path is deliberately not configurable from Hermes YAML or
    model-authored arguments.  The broker independently checks SO_PEERCRED, the
    exact gateway MainPID and every fixed provider/tool schema.
    """
    encoded = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    if not encoded or len(encoded) > _BROKER_FRAME_LIMIT:
        raise RuntimeError("request_too_large")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.set_inheritable(False)
        connection.settimeout(timeout)
        connection.connect(socket_path)
        connection.sendall(encoded + b"\n")
        response = bytearray()
        while True:
            chunk = connection.recv(min(65_536, _BROKER_FRAME_LIMIT + 1 - len(response)))
            if not chunk:
                raise RuntimeError("incomplete_response")
            response.extend(chunk)
            if len(response) > _BROKER_FRAME_LIMIT:
                raise RuntimeError("response_too_large")
            newline = response.find(b"\n")
            if newline < 0:
                continue
            if newline != len(response) - 1:
                raise RuntimeError("invalid_response_frame")
            value = json.loads(response[:newline].decode("utf-8"))
            if not isinstance(value, dict):
                raise RuntimeError("invalid_response")
            return value


def _current_binding(*, session_id: Any = "", tool_call_id: Any = "") -> tuple[str, str, str]:
    """Return broker principal/session/tool-call ids from trusted Hermes context."""
    try:
        from tools.approval import (
            _current_owner_high_impact_principal,
            owner_high_impact_principal_sha256,
        )

        principal = _current_owner_high_impact_principal()
        principal_sha256 = owner_high_impact_principal_sha256(principal)
    except Exception:
        principal_sha256 = ""
    clean_session = str(session_id or "").strip()
    clean_tool_call = str(tool_call_id or "").strip()
    return principal_sha256, clean_session, clean_tool_call


def _broker_payload(
    op: str, *, provider: str, tool: str, arguments: dict[str, Any],
    principal_sha256: str, session_id: str, tool_call_id: str,
    request_id: str = "", nonce: str = "",
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "v": _BROKER_PROTOCOL_VERSION,
        "op": op,
        "provider": provider,
        "tool": tool,
        "arguments": arguments,
        "principal_sha256": principal_sha256,
        "session_id": session_id,
        "tool_call_id": tool_call_id,
    }
    if op == "execute":
        payload.update({"request_id": request_id, "nonce": nonce})
    return payload


def _load_fixed_tool_specs() -> dict[tuple[str, str], dict[str, Any]]:
    """Load the root-owned, deployment-captured provider schemas fail-closed."""
    try:
        info = _BROKER_SCHEMAS.stat()
        if not _BROKER_SCHEMAS.is_absolute() or not _BROKER_SCHEMAS.is_file():
            raise RuntimeError("schema_path_invalid")
        if info.st_size <= 0 or info.st_size > 1_048_576:
            raise RuntimeError("schema_size_invalid")
        if os.name == "posix" and (info.st_uid != 0 or info.st_mode & 0o022):
            raise RuntimeError("schema_permissions_invalid")
        document = json.loads(_BROKER_SCHEMAS.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError("fixed tool schema unavailable") from exc
    if not isinstance(document, dict) or document.get("version") != 1:
        raise RuntimeError("fixed tool schema version invalid")
    servers = document.get("servers")
    if not isinstance(servers, dict):
        raise RuntimeError("fixed tool schema structure invalid")
    expected = {
        "didi": _DIDI_READS | _WRITE_TOOLS["didi"],
        "amap": _AMAP_READS,
        "netease_music": _NETEASE_READS | _WRITE_TOOLS["netease"],
    }
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for server, expected_names in expected.items():
        entries = servers.get(server)
        if not isinstance(entries, list):
            raise RuntimeError(f"fixed schema missing {server}")
        for entry in entries:
            if not isinstance(entry, dict):
                raise RuntimeError("fixed schema entry invalid")
            full_name = str(entry.get("name") or "")
            prefix = f"mcp__{server}__"
            if not full_name.startswith(prefix):
                raise RuntimeError("fixed schema tool namespace invalid")
            raw_name = full_name[len(prefix):]
            parameters = entry.get("parameters")
            if raw_name not in expected_names or not isinstance(parameters, dict):
                raise RuntimeError("fixed schema tool drift detected")
            result[(server, raw_name)] = {
                "name": full_name,
                "description": str(entry.get("description") or ""),
                "parameters": parameters,
            }
        present = {name for srv, name in result if srv == server}
        if present != set(expected_names):
            raise RuntimeError(f"fixed schema set mismatch for {server}")
    return result


def _ncm_write_tool_specs() -> list[tuple[str, dict[str, Any]]]:
    """Return friendly model-visible NCM write schemas from root-owned metadata."""
    fixed = _load_fixed_tool_specs()
    result: list[tuple[str, dict[str, Any]]] = []
    for name in sorted(_NETEASE_GROUP_WRITES):
        source = fixed[("netease_music", name)]
        result.append(
            (
                name,
                {
                    "name": name,
                    "description": source["description"],
                    "parameters": source["parameters"],
                },
            )
        )
    return result


def _owner_session_profile(session_id: Any) -> tuple[bool, str]:
    """Resolve the authenticated owner context without treating blanks as trust.

    A normal primary gateway turn currently binds ``HERMES_SESSION_PROFILE`` to
    an empty string when no profile route was needed.  In that one case we ask
    Hermes to resolve the process' active profile explicitly.  Empty values are
    never themselves accepted: a missing/failed resolver, missing routing key,
    missing hook session id, or any non-owner profile fails closed.
    """
    try:
        from gateway.session_context import get_session_env

        profile = str(
            get_session_env("HERMES_SESSION_PROFILE", "") or ""
        ).strip().lower()
        session_key = str(
            get_session_env("HERMES_SESSION_KEY", "") or ""
        ).strip()
        hook_session_id = str(session_id or "").strip()
        if not profile:
            # The gateway intentionally leaves the primary route's source
            # profile blank.  Resolve it from HERMES_HOME rather than granting
            # authority to the blank value itself.
            from hermes_cli.profiles import get_active_profile_name

            profile = str(get_active_profile_name() or "").strip().lower()
    except Exception:
        return False, ""

    context_complete = bool(session_key and hook_session_id)
    return context_complete and profile in _OWNER_PROFILES, profile


def _prune_runtime_grants(now: float | None = None) -> None:
    current = time.time() if now is None else now
    with _grant_lock:
        for key, expiry in list(_delete_grants.items()):
            if expiry <= current:
                _delete_grants.pop(key, None)
        for key, expiry in list(_used_confirmation_origins.items()):
            if expiry <= current:
                _used_confirmation_origins.pop(key, None)


def _message_contains_delete_token(value: Any) -> bool:
    """Find the one task token anywhere in the selected, trusted author input."""
    if isinstance(value, str):
        return _DELETE_CONFIRM_TOKEN in value
    if isinstance(value, list):
        pieces: list[str] = []
        for item in value:
            if isinstance(item, dict) and item.get("type") in {"text", "input_text"}:
                pieces.append(str(item.get("text") or ""))
        return _DELETE_CONFIRM_TOKEN in "\n".join(pieces)
    return False


def _origin_fingerprint(
    *, session_id: str, turn_id: str, messaging: bool, platform: str = "",
    session_key: str = "", sender_id: str = "", message_id: str = "",
) -> str:
    """Hash a gateway-authenticated message origin without retaining routing ids."""
    if messaging:
        # session_key is stable across durable session rotation; session_id is not. Include the
        # platform and authenticated sender because some transports scope message ids per chat.
        material = f"message\0{platform}\0{session_key}\0{sender_id}\0{message_id}"
    else:
        # CLI/Desktop input has no platform event id. Hermes generates a unique turn id before
        # this hook, which is the strongest available local task boundary.
        material = f"turn\0{session_id}\0{turn_id}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _on_pre_llm_call(
    session_id: Any = "", turn_id: Any = "", user_message: Any = None,
    sender_id: Any = "", trusted_user_message: Any = None,
    trusted_user_message_id: Any = None,
    trusted_user_message_internal: Any = None, **_: Any,
):
    """Capture one owner-authenticated deletion grant for this exact turn."""
    try:
        from gateway.session_context import get_session_env, session_is_messaging_surface

        platform = str(get_session_env("HERMES_SESSION_PLATFORM", "") or "").strip()
        session_key = str(get_session_env("HERMES_SESSION_KEY", "") or "").strip()
        bound_sender = str(get_session_env("HERMES_SESSION_USER_ID", "") or "").strip()
        messaging = bool(session_is_messaging_surface())
    except Exception:
        # Unknown surfaces are messaging by default, hence fail closed below.
        platform = session_key = bound_sender = ""
        messaging = True

    clean_sender = str(sender_id or "").strip()
    clean_message_id = str(trusted_user_message_id or "").strip()
    if messaging:
        # These values are independent hook arguments populated only from MessageEvent by the
        # gateway. Never recover them from enriched user_message or HERMES_SESSION_MESSAGE_ID
        # (the latter is a reply/source anchor, not necessarily the inbound QQ/Weixin id).
        if (
            trusted_user_message_internal is not False
            or not isinstance(trusted_user_message, str)
            or not clean_message_id
            or not session_key
            or not bound_sender
            or clean_sender != bound_sender
        ):
            return None
        token_source = trusted_user_message
    else:
        # Direct CLI/Desktop input reaches this hook without messaging enrichment. Synthetic
        # callers are still excluded when an explicit internal marker is present.
        if trusted_user_message_internal is True:
            return None
        token_source = user_message
    if not _message_contains_delete_token(token_source):
        return None

    owner_context, _profile = _owner_session_profile(session_id)
    principal_sha256, bound_session, _ = _current_binding(session_id=session_id)
    clean_turn = str(turn_id or "").strip()
    if not owner_context or not principal_sha256 or not bound_session or not clean_turn:
        return None
    _prune_runtime_grants()
    origin = _origin_fingerprint(
        session_id=bound_session, turn_id=clean_turn, messaging=messaging,
        platform=platform, session_key=session_key, sender_id=clean_sender,
        message_id=clean_message_id,
    )
    key = (bound_session, clean_turn, principal_sha256)
    now = time.time()
    with _grant_lock:
        if _delete_grants.get(key, 0) <= now:
            if _used_confirmation_origins.get(origin, 0) > now:
                return {
                    "context": (
                        "[OWNER TASK CONSENT REPLAY BLOCKED] 此入口消息已使用过删除确认，"
                        "不会为新的任务轮次重复授权。"
                    )
                }
            expiry = now + _TASK_GRANT_TTL_SECONDS
            _delete_grants[key] = expiry
            _used_confirmation_origins[origin] = expiry
    return {
        "context": (
            "[OWNER TASK CONSENT] 本轮主人已给出一次删除确认。仅允许用于本轮明确请求的"
            "网易云真实删除或非 Agent 歌单移除歌曲；不得扩展到下一轮。"
        )
    }


def _current_turn_id() -> str:
    try:
        from tools.approval_context import get_current_turn_id

        return str(get_current_turn_id() or "").strip()
    except Exception:
        return ""


def _has_delete_grant(session_id: str, principal_sha256: str, turn_id: str) -> bool:
    _prune_runtime_grants()
    with _grant_lock:
        return _delete_grants.get((session_id, turn_id, principal_sha256), 0) > time.time()


def _owner_created_playlist_name(value: Any, playlist_id: str) -> str:
    """Parse only the fixed ``created_playlists`` result shape."""

    if not isinstance(value, dict) or value.get("success") is not True:
        return ""
    data = value.get("data")
    records = data.get("records") if isinstance(data, dict) else None
    if not isinstance(records, list):
        return ""
    names = [
        item.get("name")
        for item in records
        if isinstance(item, dict)
        and str(item.get("id") or "").casefold() == playlist_id.casefold()
        and isinstance(item.get("name"), str)
    ]
    return names[0] if len(names) == 1 else ""


def _read_agent_playlist_registry_unlocked() -> set[str]:
    """Read the private Agent-playlist registry, failing closed on any drift."""

    path = Path(_AGENT_PLAYLIST_REGISTRY)
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        return set()
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise RuntimeError("agent_playlist_registry_not_regular")
        if info.st_size <= 0 or info.st_size > 65_536:
            raise RuntimeError("agent_playlist_registry_size_invalid")
        if os.name == "posix" and info.st_mode & 0o077:
            raise RuntimeError("agent_playlist_registry_permissions_invalid")
        with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
            descriptor = -1
            document = json.load(handle)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if not isinstance(document, dict) or document.get("version") != 1:
        raise RuntimeError("agent_playlist_registry_version_invalid")
    values = document.get("playlist_ids")
    if not isinstance(values, list):
        raise RuntimeError("agent_playlist_registry_structure_invalid")
    clean: set[str] = set()
    for value in values:
        item = str(value or "").strip().casefold()
        if not re.fullmatch(r"[0-9a-f]{32}", item) or item in clean:
            raise RuntimeError("agent_playlist_registry_id_invalid")
        clean.add(item)
    return clean


def _read_agent_playlist_registry() -> set[str]:
    with _agent_playlist_registry_lock:
        return _read_agent_playlist_registry_unlocked()


def _write_agent_playlist_registry_unlocked(values: set[str]) -> None:
    """Atomically replace the private registry with a mode-0600 regular file."""

    path = Path(_AGENT_PLAYLIST_REGISTRY)
    parent = path.parent
    parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if parent.is_symlink() or not parent.is_dir():
        raise RuntimeError("agent_playlist_registry_parent_invalid")
    if path.is_symlink():
        raise RuntimeError("agent_playlist_registry_symlink")
    document = {
        "version": 1,
        "playlist_ids": sorted(values),
    }
    encoded = (
        json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(parent)
    )
    temporary = Path(temporary_name)
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(descriptor, 0o600)
        else:
            os.chmod(temporary, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
        if os.name == "posix":
            directory = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _record_agent_playlist_id(playlist_id: Any) -> None:
    clean_id = str(playlist_id or "").strip().casefold()
    if not re.fullmatch(r"[0-9a-f]{32}", clean_id):
        raise ValueError("agent_playlist_registry_id_invalid")
    with _agent_playlist_registry_lock:
        values = _read_agent_playlist_registry_unlocked()
        if clean_id in values:
            return
        values.add(clean_id)
        _write_agent_playlist_registry_unlocked(values)


def _is_registered_agent_playlist(playlist_id: Any) -> bool:
    clean_id = str(playlist_id or "").strip().casefold()
    if not re.fullmatch(r"[0-9a-f]{32}", clean_id):
        return False
    try:
        return clean_id in _read_agent_playlist_registry()
    except Exception:
        logger.exception("Agent playlist registry failed closed")
        return False


def _is_current_owner_agent_playlist(
    playlist_id: Any, *, principal_sha256: str, session_id: str,
    tool_call_id: str,
) -> bool:
    clean_id = str(playlist_id or "").strip()
    if (
        not re.fullmatch(r"[0-9A-Fa-f]{32}", clean_id)
        or not _is_registered_agent_playlist(clean_id)
    ):
        return False
    response = _broker_request(
        _BROKER_READ_SOCKET,
        _broker_payload(
            "read",
            provider="netease",
            tool="netease_user_library",
            arguments={"action": "created_playlists", "limit": 500, "offset": 0},
            principal_sha256=principal_sha256,
            session_id=session_id,
            tool_call_id=(tool_call_id[:220] + ":agent-playlist-check"),
        ),
        timeout=50.0,
    )
    name = (
        _owner_created_playlist_name(response.get("result"), clean_id)
        if response.get("ok") is True
        else ""
    )
    return "agent" in name.casefold()


def _protected_delete_kind(
    provider: str, upstream_tool: str, args: dict[str, Any]
) -> str:
    """Validate protected-delete parameters without I/O or classification."""

    if provider != "netease":
        return ""
    action = str(args.get("action") or "")
    if upstream_tool == "netease_note_write" and action == "delete":
        if not str(args.get("event_id") or "").isdigit():
            raise ValueError("note_delete_requires_event_id")
        return "always"
    if upstream_tool == "netease_playlist_write" and action == "remove_tracks":
        song_ids = args.get("song_ids")
        if (
            not re.fullmatch(r"[0-9A-Fa-f]{32}", str(args.get("playlist_id") or ""))
            or not isinstance(song_ids, list)
            or not song_ids
            or any(not re.fullmatch(r"[0-9A-Fa-f]{32}", str(item or "")) for item in song_ids)
        ):
            raise ValueError("playlist_remove_requires_encrypted_ids")
        return "playlist_remove"
    return ""


def _requires_delete_grant(
    provider: str, upstream_tool: str, args: dict[str, Any], *,
    principal_sha256: str, session_id: str, tool_call_id: str, turn_id: str,
) -> bool:
    kind = _protected_delete_kind(provider, upstream_tool, args)
    if kind == "always":
        return True
    if kind == "playlist_remove":
        return not _is_current_owner_agent_playlist(
            args["playlist_id"], principal_sha256=principal_sha256,
            session_id=session_id, tool_call_id=tool_call_id,
        )
    return False


def _on_pre_tool_call_impl(
    tool_name: str = "", args: Any = None, session_id: Any = "",
    tool_call_id: Any = "", turn_id: Any = "", **_: Any
):
    if _is_forbidden_qualified_write(tool_name):
        return {
            "action": "block",
            "message": "BLOCKED: MCP 直连账号写入路径已永久停用，请使用指定主控助手的工具。",
        }
    if not _is_guarded(tool_name):
        return None
    owner_context, profile = _owner_session_profile(session_id)
    if not owner_context:
        return {
            "action": "block",
            "message": (
                "BLOCKED: 该外部账号操作只授权给指定 Host 默认主控助手，"
                f"当前 profile/session 未授权 (profile={(profile or 'unavailable')!r})。"
            ),
        }
    principal_sha256, bound_session, bound_tool_call = _current_binding(
        session_id=session_id, tool_call_id=tool_call_id
    )
    base_name = _base_tool_name(tool_name)
    target = _WRITE_ALIAS.get(base_name)
    if target is None:
        return {
            "action": "block",
            "message": "BLOCKED: 旧的直连账号写入路径已永久停用，请使用指定主控助手的工具。",
        }
    if not principal_sha256 or not isinstance(args, dict) or not bound_session or not bound_tool_call:
        return {
            "action": "block",
            "message": "BLOCKED: 外部账号操作缺少可信主人、会话或 tool-call 绑定。",
        }
    provider, upstream_tool = target
    clean_turn = str(turn_id or _current_turn_id()).strip()
    if not clean_turn:
        return {"action": "block", "message": "BLOCKED: 缺少本轮任务标识。"}
    # This hook must remain local and fast.  Agent-playlist membership and the
    # current owner-created name require a private broker read, so that concrete
    # classification is deliberately deferred to the plugin handler below.
    delete_kind = _protected_delete_kind(provider, upstream_tool, args)
    if delete_kind == "always" and not _has_delete_grant(
        bound_session, principal_sha256, clean_turn
    ):
        return {
            "action": "block",
            "message": (
                "这是一项受保护的真实删除：请在包含本次删除指令的同一条主人消息中"
                "写一次“CONFIRM_DELETE”后重试。本轮其他受保护删除不会再次询问。"
            ),
        }
    return None


def _resolution_payload(op: str, grant: dict[str, Any]) -> dict[str, Any]:
    return {
        "v": _BROKER_PROTOCOL_VERSION,
        "op": op,
        "request_id": grant["request_id"],
        "nonce": grant["nonce"],
        "principal_sha256": grant["principal_sha256"],
        "session_id": grant["session_id"],
        "tool_call_id": grant["tool_call_id"],
    }


def _on_pre_tool_call(
    tool_name: str = "", args: Any = None, session_id: Any = "", **kwargs: Any
):
    """Fail closed if any part of this security policy itself malfunctions.

    Hermes historically isolates ordinary Python hook exceptions and continues
    the tool call.  That behaviour is useful for observer plugins but unsafe for
    the one hook protecting paid orders and account mutations, so this wrapper
    converts every policy exception into an explicit block directive.
    """
    private_read = _is_private_account_read(tool_name)
    forbidden_qualified = _is_forbidden_qualified_write(tool_name)
    if not _is_guarded(tool_name) and not private_read and not forbidden_qualified:
        return None
    if private_read:
        try:
            if _private_owner_context_allowed(session_id):
                return None
        except Exception:
            logger.exception(
                "Private account read policy failed closed for %s",
                _base_tool_name(tool_name),
            )
        return {
            "action": "block",
            "message": (
                "BLOCKED: 滴滴订单/账户与网易云私人账户读取只授权给"
                "指定 Host 默认主控助手及固定主人身份。"
            ),
        }
    try:
        return _on_pre_tool_call_impl(
            tool_name=tool_name, args=args, session_id=session_id, **kwargs
        )
    except Exception:
        logger.exception("Owner high-impact policy failed closed for %s", _base_tool_name(tool_name))
        return {
            "action": "block",
            "message": (
                "BLOCKED: 主人订单/账号写操作的安全策略发生异常，已拒绝执行。"
                "请联系维护者检查，禁止绕过或自动重试。"
            ),
        }


_on_pre_tool_call._hermes_fail_closed_on_exception = True


def _runtime_read_binding(
    session_id: Any = "", *, require_private_owner: bool = False
) -> tuple[str, str, str] | None:
    """Build correlation ids, optionally requiring the private owner boundary.

    Didi and NetEase reads reveal order/account or private-library data and
    therefore require both the Shanghai owner profile and a root-allowlisted
    authenticated principal.  Amap reads do not carry private account state and
    may use the non-authorizing system correlation principal.
    """
    try:
        from tools.approval_context import (
            get_current_observability_session_id,
            get_current_tool_call_id,
        )

        current_session = get_current_observability_session_id()
        current_call = get_current_tool_call_id()
    except Exception:
        current_session = current_call = ""
    bound_session = str(current_session or session_id or "").strip()[:256]
    bound_call = str(current_call or "").strip()[:256]
    principal_sha256, _, _ = _current_binding(
        session_id=bound_session, tool_call_id=bound_call
    )
    if require_private_owner:
        owner_context, _profile = _owner_session_profile(bound_session)
        if not owner_context or not principal_sha256 or not bound_session or not bound_call:
            return None
    if not principal_sha256:
        principal_sha256 = hashlib.sha256(b"hermes-owner-system-read-v1").hexdigest()
    return (
        principal_sha256,
        bound_session or "gateway-read-session",
        bound_call or f"read-{secrets.token_hex(16)}",
    )


def _call_broker_read(
    provider: str, upstream_tool: str, params: Any, *, session_id: Any = "",
) -> str:
    if not isinstance(params, dict):
        return json.dumps({"ok": False, "error": "参数必须为对象"}, ensure_ascii=False)
    private_account = provider in {"didi", "netease"}
    binding = _runtime_read_binding(
        session_id, require_private_owner=private_account
    )
    if binding is None:
        return json.dumps(
            {
                "ok": False,
                "error": (
                    "BLOCKED: 滴滴订单/账户与网易云私人账户读取只授权给"
                    "指定 Host 默认主控助手及固定主人身份。"
                ),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    principal, bound_session, bound_call = binding
    try:
        response = _broker_request(
            _BROKER_READ_SOCKET,
            _broker_payload(
                "read", provider=provider, tool=upstream_tool, arguments=params,
                principal_sha256=principal, session_id=bound_session,
                tool_call_id=bound_call,
            ),
            timeout=65.0 if provider == "didi" else 50.0,
        )
    except Exception:
        response = {"ok": False, "error": "broker_unavailable"}
    if response.get("ok") is not True:
        response = {
            "ok": False,
            "error": _safe_broker_error(response.get("error"), write=False),
        }
    return json.dumps(response, ensure_ascii=False, separators=(",", ":"))


def _record_verified_agent_playlist_create(
    provider: str, upstream_tool: str, params: dict[str, Any], response: dict[str, Any]
) -> bool:
    """Record only a plugin-created Agent playlist whose effect was verified."""

    if (
        provider != "netease"
        or upstream_tool != "netease_playlist_write"
        or params.get("action") != "create"
    ):
        return False
    requested_name = str(params.get("name") or "").strip()
    if "agent" not in requested_name.casefold() or response.get("ok") is not True:
        return False
    result = response.get("result")
    if (
        not isinstance(result, dict)
        or result.get("success") is not True
        or result.get("changed") is not True
        or result.get("effectVerified") is not True
    ):
        return False
    playlist = result.get("playlist")
    if not isinstance(playlist, dict):
        return False
    returned_name = str(playlist.get("name") or "").strip()
    playlist_id = str(playlist.get("id") or "").strip()
    if returned_name != requested_name or "agent" not in returned_name.casefold():
        return False
    if not re.fullmatch(r"[0-9A-Fa-f]{32}", playlist_id):
        return False
    _record_agent_playlist_id(playlist_id)
    return True


def _call_broker_write(
    provider: str, upstream_tool: str, params: Any, *, session_id: Any = "",
) -> str:
    """Execute one owner write through a fresh bound broker transaction."""
    if not isinstance(params, dict):
        return json.dumps({"ok": False, "error": "参数必须为对象"}, ensure_ascii=False)
    try:
        from tools.approval_context import (
            get_current_observability_session_id,
            get_current_tool_call_id,
            get_current_turn_id,
        )

        bound_session = str(get_current_observability_session_id() or session_id or "")
        bound_call = str(get_current_tool_call_id() or "")
        bound_turn = str(get_current_turn_id() or "")
    except Exception:
        bound_session = bound_call = bound_turn = ""
    owner_context, _profile = _owner_session_profile(bound_session)
    principal_sha256, _, _ = _current_binding(
        session_id=bound_session, tool_call_id=bound_call
    )
    if not owner_context or not principal_sha256 or not bound_session or not bound_call or not bound_turn:
        return json.dumps(
            {"ok": False, "error": "BLOCKED: 缺少可信主人、会话、任务或 tool-call 绑定。"},
            ensure_ascii=False, separators=(",", ":"),
        )
    try:
        needs_delete_grant = _requires_delete_grant(
            provider, upstream_tool, params,
            principal_sha256=principal_sha256, session_id=bound_session,
            tool_call_id=bound_call, turn_id=bound_turn,
        )
    except Exception:
        return json.dumps(
            {"ok": False, "error": "BLOCKED: 删除目标或参数无法安全确认。"},
            ensure_ascii=False, separators=(",", ":"),
        )
    if needs_delete_grant and not _has_delete_grant(
        bound_session, principal_sha256, bound_turn
    ):
        return json.dumps(
            {"ok": False, "error": "BLOCKED: 本轮缺少一次删除确认“CONFIRM_DELETE”。"},
            ensure_ascii=False, separators=(",", ":"),
        )

    try:
        prepared = _broker_request(
            _BROKER_WRITE_SOCKET,
            _broker_payload(
                "prepare", provider=provider, tool=upstream_tool, arguments=params,
                principal_sha256=principal_sha256, session_id=bound_session,
                tool_call_id=bound_call,
            ),
            timeout=3.0,
        )
        request_id = str(prepared.get("request_id") or "")
        nonce = str(prepared.get("nonce") or "")
        expires_at = prepared.get("expires_at")
        now = int(time.time())
        if (
            prepared.get("ok") is not True
            or not re.fullmatch(r"[0-9a-f]{32}", request_id)
            or not re.fullmatch(r"[A-Za-z0-9_-]{40,64}", nonce)
            or not isinstance(expires_at, int)
            or expires_at <= now
            or expires_at > now + 300
        ):
            raise RuntimeError("broker_prepare_failed")
        grant = {
            "request_id": request_id,
            "nonce": nonce,
            "principal_sha256": principal_sha256,
            "session_id": bound_session,
            "tool_call_id": bound_call,
        }
        confirmed = _broker_request(
            _BROKER_WRITE_SOCKET, _resolution_payload("confirm", grant), timeout=3.0
        )
        if confirmed.get("ok") is not True:
            raise RuntimeError("broker_confirm_failed")
    except Exception:
        response = {"ok": False, "error": "broker_authorization_failed"}
    else:
        try:
            response = _broker_request(
                _BROKER_WRITE_SOCKET,
                _broker_payload(
                    "execute", provider=provider, tool=upstream_tool, arguments=params,
                    principal_sha256=principal_sha256, session_id=bound_session,
                    tool_call_id=bound_call, request_id=request_id, nonce=nonce,
                ),
                timeout=(75.0 if provider == "didi" else 305.0),
            )
        except Exception:
            # The upstream may already have accepted a non-idempotent operation.
            # Never replay it automatically.
            response = {"ok": False, "error": "outcome_uncertain"}
    if response.get("ok") is True:
        try:
            if _record_verified_agent_playlist_create(
                provider, upstream_tool, params, response
            ):
                response["agentPlaylistRegistry"] = {"recorded": True}
        except Exception:
            # The remote create is already confirmed and must never be replayed.
            # Keep the successful result, but make the local policy degradation
            # explicit; this playlist will require the deletion token until a
            # maintainer repairs/records it safely.
            logger.exception("Created Agent playlist could not be registered")
            response["agentPlaylistRegistry"] = {
                "recorded": False,
                "warning": "歌单已创建，但 Agent 管理注册表写入失败；移除歌曲仍需删除确认。",
            }
    if response.get("ok") is not True:
        response = {
            "ok": False,
            "error": _safe_broker_error(response.get("error"), write=True),
        }
    return json.dumps(response, ensure_ascii=False, separators=(",", ":"))


def register(ctx) -> None:
    ctx.register_hook("pre_llm_call", _on_pre_llm_call)
    ctx.register_hook("pre_tool_call", _on_pre_tool_call)
    fixed_specs = _load_fixed_tool_specs()

    # Do not expose the upstream order mutations directly from the MCP server.
    # These two plugin-owned tools are the only model-visible path; disabling
    # this plugin therefore removes the write capability instead of bypassing
    # its approval hook.
    create_schema = {
        **fixed_specs[("didi", "taxi_create_order")],
        "name": "didi_create_order",
        "description": (
            "Create a real paid Didi ride order from the latest estimate. "
            "Owner-only; executes directly through the single-shot credential broker."
        ),
    }
    cancel_schema = {
        **fixed_specs[("didi", "taxi_cancel_order")],
        "name": "didi_cancel_order",
        "description": (
            "Cancel an existing Didi ride order. Owner-only; query and explain the "
            "current order first. The broker never retries an uncertain cancellation."
        ),
    }

    ctx.register_tool(
        name="didi_create_order",
        toolset="didi_owner_order_control",
        schema=create_schema,
        handler=lambda params, **kwargs: _call_broker_write(
            "didi", "taxi_create_order", params, session_id=kwargs.get("session_id", "")
        ),
    )
    ctx.register_tool(
        name="didi_cancel_order",
        toolset="didi_owner_order_control",
        schema=cancel_schema,
        handler=lambda params, **kwargs: _call_broker_write(
            "didi", "taxi_cancel_order", params, session_id=kwargs.get("session_id", "")
        ),
    )

    try:
        ncm_specs = _ncm_write_tool_specs()
    except Exception as exc:
        # Fail closed for NetEase mutations while keeping the independent Didi
        # guarded tools available.  The deployment health check requires all
        # nine registrations, so this cannot pass unnoticed in production.
        logger.error("NetEase owner write tools were not registered: %s", exc)
        ncm_specs = []
    for name, schema in ncm_specs:
        def _handler(params, _upstream=name, **kwargs):
            return _call_broker_write(
                "netease", _upstream, params, session_id=kwargs.get("session_id", "")
            )

        ctx.register_tool(
            name=name,
            toolset="netease_owner_account_control",
            schema=schema,
            handler=_handler,
        )

    # Read-only capabilities also stay behind the credential broker so the
    # gateway never receives provider keys or NetEase login-state files.  They
    # do not need the phrase gate, but the broker still requires the exact
    # trusted gateway MainPID; the public daidai gateway and terminal children
    # therefore cannot reach the private account.
    for server, provider, names in (
        ("didi", "didi", _DIDI_READS),
        ("amap", "amap", _AMAP_READS),
        ("netease_music", "netease", _NETEASE_READS),
    ):
        for upstream_name in sorted(names):
            schema = fixed_specs[(server, upstream_name)]

            def _read_handler(params, _provider=provider, _tool=upstream_name, **kwargs):
                return _call_broker_read(
                    _provider, _tool, params, session_id=kwargs.get("session_id", "")
                )

            ctx.register_tool(
                name=schema["name"],
                toolset=server,
                schema=schema,
                handler=_read_handler,
            )
