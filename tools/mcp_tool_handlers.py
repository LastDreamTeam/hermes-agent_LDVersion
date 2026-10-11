"""Registry-facing sync handlers for MCP tools and utility tools (resources/prompts), plus the per-call recovery
ladder: trust gating, circuit breaker, auth (401) refresh, session-expired reconnect and dead-stdio respawn. Recovery
replay is suppressed when a native tool explicitly declares ``annotations.idempotentHint=false``."""

import logging
import asyncio
import contextvars
import inspect
import json
import time
from contextlib import asynccontextmanager
from functools import partial
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple
from tools.registry import tool_error
from tools.ansi_strip import strip_unicode_tags
from tools.mcp_tool_common import _exc_str, _sanitize_error, mcp_field, _core
from tools import mcp_tool_loop as _loop
from tools.mcp_tool_content import (
    _MCP_HARD_RESULT_CAP_CHARS, _cache_mcp_audio_block, _cache_mcp_image_block,
    _render_mcp_dropped_block_notice, _render_mcp_resource_block, _strip_reserved_meta_keys,
    _truncate_mcp_text_result)
from tools.mcp_tool_errors import _is_auth_error, _is_session_expired_error

logger = logging.getLogger("tools.mcp_tool")
_MISSING = object()

_NEEDS_REAUTH_MSG = (
    "MCP server '{s}' requires re-authentication. Run `hermes mcp login {s}` (or delete the tokens file under "
    "~/.hermes/mcp-tokens/ and restart). Do NOT retry this tool — ask the user to re-authenticate.")
_STDIO_NO_RESPAWN_MSG = (
    "MCP server '{s}' stdio subprocess had exited (this is not a timeout — the call never reached the server). A "
    "respawn was requested but no fresh session came back within {t:.0f}s. Wait a few seconds before retrying; if it "
    "keeps failing the server is not starting and needs the user.")
_STDIO_DIED_AGAIN_MSG = (
    "MCP server '{s}' respawned its stdio subprocess and it exited again immediately. The server is not starting "
    "cleanly — do NOT retry this tool; ask the user to check the server's command and its stderr log.")


def _non_idempotent_uncertain_result(server_name: str, tool_name: str, recovery: str, **extra) -> str:
    """Fail closed after recovering a transport used by a non-idempotent call.

    The first RPC may have reached the remote service even though its response
    was lost.  Retrying could duplicate an order, payment, message, or other
    side effect, so the caller must query authoritative state first.
    """
    return tool_error(
        f"MCP tool '{tool_name}' on server '{server_name}' declares annotations.idempotentHint=false. "
        f"Its call encountered {recovery} while dispatching or awaiting a response, so the external outcome "
        "may be uncertain. Hermes "
        "recovered or "
        "requested recovery of the connection but deliberately did NOT automatically replay the tool. Query "
        "authoritative status with a read-only/query tool before deciding whether a new attempt is safe.",
        outcome_uncertain=True, automatic_replay=False, server=server_name, tool=tool_name, **extra)


def _trust_gate_check(server_name: str, tool_name: str) -> Optional[str]:
    """Approval gate for write-capable tools on ``trust: untrusted`` servers. None to proceed,
    else a ``tool_error``. Fail-closed: approval-system errors block."""
    if (_core._server_trust_levels.get(server_name, _core._TRUST_FULL) != _core._TRUST_UNTRUSTED
            or _core._tool_read_only_hints.get(server_name, {}).get(tool_name) is True):
        return None
    try:  # lazy: tools.approval routes the prompt to whichever surface owns the session
        from tools.approval_prompt import request_elicitation_consent
        answer = request_elicitation_consent(
            f"MCP tool '{tool_name}' on UNTRUSTED server '{server_name}' wants to run. This tool is write-capable "
            f"(no readOnlyHint=true annotation) and may modify external state.",
            f"Server '{server_name}' is configured 'trust: untrusted'. "
            f"Approve to run '{tool_name}' once, or deny to block it.",
            surface=f"mcp-trust/{server_name}")
    except Exception as exc:
        logger.error("MCP trust gate: approval check failed for %s.%s: %s", server_name, tool_name, exc, exc_info=True)
        return tool_error(f"MCP tool '{tool_name}' on untrusted server '{server_name}' was blocked: the approval "
                          f"system was unavailable (fail-closed).")
    if answer == "accept":
        return None
    logger.info("MCP trust gate: user %s '%s' on untrusted server '%s'",
                "cancelled" if answer == "cancel" else "denied", tool_name, server_name)
    return tool_error(f"The user did not approve running write-capable MCP tool '{tool_name}' on untrusted server "
                      f"'{server_name}'. The command was NOT run. Do not retry without explicit user direction.")


def _check_circuit_breaker(server_name: str) -> Optional[str]:
    """Open-breaker error, or None when calls may proceed. After the cooldown the breaker is
    half-open: the next call probes; success resets, failure re-bumps and re-arms the cooldown."""
    failures = _core._server_error_counts.get(server_name, 0)
    age = time.monotonic() - _core._server_breaker_opened_at.get(server_name, 0.0)
    if failures < _core._CIRCUIT_BREAKER_THRESHOLD or age >= _core._CIRCUIT_BREAKER_COOLDOWN_SEC:
        return None
    return tool_error(f"MCP server '{server_name}' is unreachable after {failures} consecutive failures. "
                      f"Auto-retry available in ~{max(1, int(_core._CIRCUIT_BREAKER_COOLDOWN_SEC - age))}s. Do NOT retry "
                      f"this tool yet — use alternative approaches or ask the user to check the MCP server.")


def _acquire_call_server(server_name: str, tool_timeout: float):
    """``(server, None)`` when a call may be dispatched, else ``(None, error)``. No session: a
    reconnect may be completing, so wait briefly before a breaker strike; still down -> ask the
    server task to rebuild (probing a dead transport would re-arm the breaker forever)."""
    from tools import mcp_tool_discovery as _discovery  # lazy: discovery -> registration -> handlers cycle
    not_connected = tool_error(f"MCP server '{server_name}' is not connected")
    server = _discovery._get_connected_server_for_call(server_name)
    wait = min(5.0, float(tool_timeout or 5.0))
    if server and (server.session or _loop._wait_for_server_session_ready(server, timeout=wait)):
        return server, None
    _core._bump_server_error(server_name)
    if server and _loop._signal_reconnect(server):
        return None, tool_error(f"MCP server '{server_name}' transport is down; reconnect requested. Do NOT retry this "
                                f"tool immediately — give it a few seconds to come back.")
    return None, not_connected


def _result_is_error(result) -> bool:
    """True only for a JSON payload carrying an ``error`` key (non-JSON = success)."""
    try:
        return "error" in json.loads(result)
    except (json.JSONDecodeError, TypeError):
        return False


def _record_call_outcome(server_name: str, result) -> Any:
    """Breaker bookkeeping: an error payload from the tool itself still counts as a strike."""
    (_core._bump_server_error if _result_is_error(result) else _core._reset_server_error)(server_name)
    return result


def _strike(server_name: str, message: str, **extra) -> str:
    """Breaker strike + the ``tool_error`` payload for *message*."""
    _core._bump_server_error(server_name)
    return tool_error(message, **extra)


def _mcp_loop_running() -> bool:
    return _core._mcp_loop is not None and _core._mcp_loop.is_running()


def _lookup_reconnectable_server(server_name: str, require_loop: bool = False):
    """The registered server object when it can be signalled to reconnect, else None.
    With *require_loop*, also None unless the MCP loop is running (nothing to wait on)."""
    with _core._lock:
        srv = _core._servers.get(server_name)
    ok = srv is not None and hasattr(srv, "_reconnect_event") and (_mcp_loop_running() or not require_loop)
    return srv if ok else None


def _retry_once(server_name: str, retry_call, op_description: str, what: str):
    """Re-run ``retry_call`` after a recovery step. Returns the result (closing the breaker)
    when it is not an error payload; None when the retry raised or errored (caller falls through)."""
    try:
        result = retry_call()
    except Exception as retry_exc:
        logger.warning("MCP %s/%s retry after %s failed: %s", server_name, op_description, what, retry_exc)
        return None
    if _result_is_error(result):
        return None
    _core._reset_server_error(server_name)
    return result


def _handle_auth_error_and_retry(server_name: str, exc: BaseException, retry_call, op_description: str, *,
                                 allow_replay: bool = True, tool_name: Optional[str] = None):
    """OAuth recovery + at most one retry; None when *exc* is not an auth error.

    ``handle_401`` decides viability.  An explicit non-idempotent caller still
    refreshes/reconnects but returns an uncertain result without replaying.
    """
    if not _is_auth_error(exc):
        return None
    from tools.mcp_oauth_manager import get_manager
    try:
        recovered = _loop._run_on_mcp_loop(lambda: get_manager().handle_401(server_name, None), timeout=10)
    except Exception as rec_exc:
        logger.warning("MCP OAuth '%s': recovery attempt failed: %s", server_name, rec_exc)
        recovered = False
    if recovered:
        srv = _lookup_reconnectable_server(server_name)
        reconnected = False
        # Recovery + reconnect is independent evidence of viability: close the breaker here, not only on
        # retry success (else a failing retry pins it open forever).
        if srv is not None and _loop._signal_reconnect_and_wait(
                server_name, srv, op_description=f"{op_description} after OAuth recovery", timeout=15):
            reconnected = True
            _core._reset_server_error(server_name)
        if not allow_replay:
            if not reconnected:
                _core._bump_server_error(server_name)
            return _non_idempotent_uncertain_result(
                server_name, tool_name or op_description, "authentication/transport recovery",
                connection_recovered=reconnected)
        result = _retry_once(server_name, retry_call, op_description, "auth recovery")
        if result is not None:
            return result
    if not allow_replay:
        _core._bump_server_error(server_name)
        return _non_idempotent_uncertain_result(
            server_name, tool_name or op_description, "authentication recovery",
            needs_reauth=True)
    return _strike(server_name, _NEEDS_REAUTH_MSG.format(s=server_name), needs_reauth=True, server=server_name)


def _handle_session_expired_and_retry(server_name: str, exc: BaseException, retry_call, op_description: str, *,
                                      allow_replay: bool = True, tool_name: Optional[str] = None):
    """Transport reconnect + at most one retry on session expiry; None to fall through.

    Non-idempotent callers reconnect but never replay.  This skips ``handle_401``:
    the token is valid, only the server-side session is stale.

    Unlike :func:`_handle_auth_error_and_retry`, this does **not** call the OAuth manager's ``handle_401`` —
    the access token is still valid, only the server-side session state is stale. Setting
    ``_reconnect_event`` causes the server task's lifecycle loop to tear down the current
    ``streamablehttp_client`` + ``ClientSession`` and rebuild them, reusing the existing OAuth provider
    instance. See #13383.
    """
    if not _is_session_expired_error(exc):
        return None
    srv = _lookup_reconnectable_server(server_name, require_loop=True)
    if srv is None:
        if not allow_replay:
            _core._bump_server_error(server_name)
            return _non_idempotent_uncertain_result(
                server_name, tool_name or op_description, "session/transport failure",
                connection_recovered=False)
        return None
    logger.info("MCP server '%s': %s failed with session-expired error (%s); signalling transport reconnect "
                "and retrying once.", server_name, op_description, exc)
    if not _loop._signal_reconnect_and_wait(server_name, srv, op_description=op_description, timeout=15):
        logger.warning("MCP server '%s': reconnect did not ready within 15s after session-expired error; "
                       "falling through to error response.", server_name)
        if not allow_replay:
            _core._bump_server_error(server_name)
            return _non_idempotent_uncertain_result(
                server_name, tool_name or op_description, "session/transport recovery",
                connection_recovered=False)
        return None
    if not allow_replay:
        _core._reset_server_error(server_name)
        return _non_idempotent_uncertain_result(
            server_name, tool_name or op_description, "session/transport recovery",
            connection_recovered=True)
    return _retry_once(server_name, retry_call, op_description, "session reconnect")


class _StdioChildExited(RuntimeError):
    """Stdio subprocess gone before or during a call.

    ``dispatched=False`` is reserved for the pre-call liveness check: no RPC
    coroutine was created, so retrying after a respawn cannot duplicate a
    side effect. Every other construction defaults to the conservative
    ``True`` state.
    """

    def __init__(self, *args, dispatched: bool = True):
        super().__init__(*args)
        self.dispatched = bool(dispatched)


def _handle_stdio_child_exited_and_retry(server_name: str, exc: Exception, retry_call, op_description: str, *,
                                         allow_replay: bool = True, tool_name: Optional[str] = None):
    """Respawn a dead stdio child and retry at most once; None if not our error. Never spawns itself: it
    sets ``_reconnect_event`` and waits, so spawn frequency stays governed by ``run()``'s
    rapid-drop budget. Non-idempotent callers stop after respawn without replay.

    Why retrying here cannot hot-cycle respawns: this function never spawns anything. It sets
    ``_reconnect_event`` (one signal, same as before) and waits for the server task to publish a fresh
    session. Spawn frequency stays governed entirely by ``run()``'s rapid-drop budget, which parks a
    transport that keeps dropping without proving healthy (#62212).
    """
    if not isinstance(exc, _StdioChildExited):
        return None
    # The pre-call PID check proves that no RPC was dispatched. A single
    # attempt on the respawned child is therefore safe even for a tool whose
    # remote operation is non-idempotent. Mid-call death remains ambiguous.
    replay_is_safe = allow_replay or not exc.dispatched
    reconnected = False
    srv = _lookup_reconnectable_server(server_name)
    if srv is not None:
        logger.info("MCP server '%s': %s found the stdio subprocess dead (%s); respawning and retrying once.",
                    server_name, op_description, exc)
        if _mcp_loop_running():
            reconnected = _loop._signal_reconnect_and_wait(
                server_name, srv, op_description=op_description, timeout=_core._STDIO_RESPAWN_WAIT_SEC)
        else:  # No MCP loop to wait on (non-async adapters, tests): still request the respawn.
            _loop._signal_reconnect(srv)
    if not reconnected:
        if not replay_is_safe:
            _core._bump_server_error(server_name)
            return _non_idempotent_uncertain_result(
                server_name, tool_name or op_description, "stdio transport recovery",
                connection_recovered=False)
        return _strike(server_name, _STDIO_NO_RESPAWN_MSG.format(s=server_name, t=_core._STDIO_RESPAWN_WAIT_SEC))
    if not replay_is_safe:
        _core._reset_server_error(server_name)
        return _non_idempotent_uncertain_result(
            server_name, tool_name or op_description, "stdio transport recovery",
            connection_recovered=True)
    try:
        return _record_call_outcome(server_name, retry_call())
    except _StdioChildExited as retry_exc:
        if not allow_replay and retry_exc.dispatched:
            _core._bump_server_error(server_name)
            return _non_idempotent_uncertain_result(
                server_name, tool_name or op_description,
                "stdio transport failure after a safe pre-dispatch respawn",
                connection_recovered=False)
        # Died again right after respawn: broken server; run()'s budget takes it to the park.
        logger.warning("MCP server '%s': %s stdio subprocess exited again right after respawn (%s); not retrying "
                       "further.", server_name, op_description, retry_exc)
        return _strike(server_name, _STDIO_DIED_AGAIN_MSG.format(s=server_name))
    except Exception as retry_exc:
        if not allow_replay:
            _core._bump_server_error(server_name)
            return _non_idempotent_uncertain_result(
                server_name, tool_name or op_description,
                "a failure after a safe pre-dispatch stdio respawn",
                connection_recovered=False)
        logger.warning("MCP %s/%s retry after stdio respawn failed: %s", server_name, op_description, retry_exc)
        return _strike(server_name, _sanitize_error(
            f"MCP call failed after respawning the stdio subprocess for '{server_name}': "
            f"{type(retry_exc).__name__}: {_exc_str(retry_exc)}"))


def _iter_exception_tree(exc: BaseException):
    """Yield a bounded, cycle-safe exception graph."""
    stack = [exc]
    seen: set[int] = set()
    for _ in range(256):
        if not stack:
            break
        current = stack.pop()
        if not isinstance(current, BaseException) or id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        nested = getattr(current, "exceptions", ())
        if isinstance(nested, (list, tuple)):
            stack.extend(nested)
        stack.extend(c for c in (
            getattr(current, "__cause__", None), getattr(current, "__context__", None))
                     if isinstance(c, BaseException))


def _http_transport_error_types() -> tuple:
    """Transport exception bases from both the SDK HTTP stack and Hermes' httpx.

    MCP 2.x may use ``httpx2`` while the rest of Hermes uses ``httpx``. Neither
    library's timeout/network exceptions inherit the built-in TimeoutError or
    ConnectionError, and their messages are frequently empty.
    """
    modules = []
    try:
        modules.append(_core.sdk_httpx())
    except Exception:
        pass
    try:
        import httpx
        modules.append(httpx)
    except ImportError:
        pass
    bases = []
    for module in modules:
        base = getattr(module, "TransportError", None) if module is not None else None
        if isinstance(base, type) and base not in bases:
            bases.append(base)
    return tuple(bases)


def _is_reconnect_teardown_failure(exc: BaseException) -> bool:
    """True when the lifecycle already cancelled this RPC to rebuild the transport."""
    return any("aborted by a reconnect teardown" in str(current).lower()
               for current in _iter_exception_tree(exc))


def _is_ambiguous_transport_failure(exc: BaseException) -> bool:
    """Narrow fallback for transport failures not covered by auth/session handlers.

    In particular, some Streamable HTTP implementations answer a dispatched
    JSON-RPC request with ``text/plain``; MCP SDK versions surface that as a
    generic runtime error rather than a typed transport error.  A timeout or
    connection error is equally ambiguous for a side-effecting operation.
    """
    http_transport_errors = _http_transport_error_types()
    for current in _iter_exception_tree(exc):
        if isinstance(current, (ConnectionError, TimeoutError) + http_transport_errors):
            return True
        status_code = getattr(getattr(current, "response", None), "status_code", None)
        if isinstance(status_code, int) and 500 <= status_code <= 599:
            return True
        message = str(current).lower()
        unexpected_plaintext = (
            "text/plain" in message
            and any(marker in message for marker in ("content-type", "content type", "unexpected", "mcp")))
        if unexpected_plaintext or "aborted by a reconnect teardown" in message:
            return True
    return False


def _handle_non_idempotent_transport_without_replay(
        server_name: str, exc: BaseException, retry_call, op_description: str, *, tool_name: str):
    """Reconnect an ambiguous transport for an explicit non-idempotent tool.

    This recoverer is installed only for ``idempotentHint=false`` and never
    calls ``retry_call``.  Earlier recoverers own auth, expired-session, and
    dead-stdio failures; this catches timeout/connection/plaintext-response
    variants that otherwise fall through as generic errors.
    """
    del retry_call  # document and enforce: this recovery path cannot replay
    if not _is_ambiguous_transport_failure(exc):
        return None
    # This error is emitted only after the lifecycle has already requested and
    # begun a transport rebuild. Setting the event again can survive into the
    # fresh session and tear it down a second time.
    if _is_reconnect_teardown_failure(exc):
        _core._bump_server_error(server_name)
        return _non_idempotent_uncertain_result(
            server_name, tool_name, "an in-progress reconnect teardown",
            connection_recovered=False, recovery_in_progress=True)
    srv = _lookup_reconnectable_server(server_name)
    reconnected = False
    if srv is not None:
        if _mcp_loop_running():
            reconnected = _loop._signal_reconnect_and_wait(
                server_name, srv, op_description=op_description, timeout=15)
        else:
            _loop._signal_reconnect(srv)
    (_core._reset_server_error if reconnected else _core._bump_server_error)(server_name)
    return _non_idempotent_uncertain_result(
        server_name, tool_name, "ambiguous transport response",
        connection_recovered=reconnected)


def _dispatch(server_name: str, server: Any, op: str, call, tool_timeout: float, recoverers,
              on_final_failure: Callable[[BaseException], None], record_outcome: bool = False) -> str:
    """Mark the call started on *server* (doubles may lack ``mark_tool_call``), run coroutine function *call*
    on the MCP loop and, on failure, walk ``recoverers`` (``(server_name, exc, retry_call, op) -> Optional[str]``,
    None = not its kind; order matters). Unrecovered exceptions go through ``on_final_failure`` and become the
    generic call-failed error. ``record_outcome`` applies breaker bookkeeping to the FIRST attempt only."""
    if callable(getattr(server, "mark_tool_call", None)):
        server.mark_tool_call()

    def call_once():
        return _loop._run_on_mcp_loop(call, timeout=tool_timeout)

    try:
        result = call_once()
        return _record_call_outcome(server_name, result) if record_outcome else result
    except InterruptedError:
        return tool_error("MCP call interrupted: user sent a new message")
    except Exception as exc:
        for recover in recoverers:
            recovered = recover(server_name, exc, call_once, op)
            if recovered is not None:
                return recovered
        on_final_failure(exc)
        return tool_error(_sanitize_error(f"MCP call failed: {type(exc).__name__}: {_exc_str(exc)}"))


@asynccontextmanager
async def _track_inflight_rpc(server: Any, server_name: str, op: str):
    """Register the running RPC so teardown can fail it fast. A deliberate teardown
    (``_reconnecting`` set first) turns the cancel into a retryable RuntimeError; external
    cancels propagate unchanged. Doubles without ``_inflight_tasks`` skip tracking.

    Every user-visible request family wraps its RPC in this context (#48069 salvage). If a deliberate
    reconnect/shutdown teardown cancels the task (``_fail_inflight_calls`` sets ``_reconnecting`` first),
    the cancel is converted into a clean retryable RuntimeError instead of a raw CancelledError; external
    cancels (caller timeout, user interrupt) propagate unchanged.
    """
    inflight, task = getattr(server, "_inflight_tasks", None), asyncio.current_task()
    tracked = task is not None and inflight is not None
    if tracked:
        inflight.add(task)
    try:
        yield
    except asyncio.CancelledError:
        if getattr(server, "_reconnecting", False):
            raise RuntimeError(f"MCP {op} on '{server_name}' was aborted by a reconnect teardown; retry the "
                               f"request on the rebuilt session") from None
        raise
    finally:
        if tracked:
            inflight.discard(task)


async def _call_tool_racing_stdio_death(server, server_name: str, tool_name: str, args: dict):
    """``session.call_tool`` that fails fast when the stdio child is/gets dead: pre-call (a dead
    child must not hold the slot for the full timeout) and mid-call (race against
    ``_watch_stdio_children``). Both raise :class:`_StdioChildExited` for the respawn path, which
    owns the reconnect signal. callable()/``is True`` because MagicMock attributes are truthy."""
    # Fast-fail (#81995): a stdio subprocess that is already dead must not own this call slot — fail
    # immediately instead of waiting out the full tool timeout on a transport nobody will ever answer.
    _stdio_dead = getattr(server, "_stdio_children_dead", None)
    if callable(_stdio_dead) and _stdio_dead() is True:
        raise _StdioChildExited(
            f"MCP stdio subprocess for '{server_name}' had already exited before the call was dispatched",
            dispatched=False)
    _call_coro = server.session.call_tool(tool_name, arguments=args)
    _watch_children = getattr(server, "_watch_stdio_children", None)
    if not (inspect.iscoroutinefunction(_watch_children) and asyncio.iscoroutine(_call_coro)):
        # Stubbed sessions return a non-awaitable, or there is no child-watcher to race: plain await.
        return await _call_coro if asyncio.iscoroutine(_call_coro) else _call_coro
    # Fast-fail machinery (#81995): the RPC races a stdio-children watcher so a dead subprocess fails the
    # call immediately instead of riding out the full tool timeout.
    rpc_task = asyncio.ensure_future(_call_coro)
    watch_task = asyncio.ensure_future(_watch_children())
    try:
        done, _pending = await asyncio.wait({rpc_task, watch_task}, return_when=asyncio.FIRST_COMPLETED)
        if watch_task in done and not rpc_task.done():
            rpc_task.cancel()
            raise _StdioChildExited(f"MCP stdio subprocess for '{server_name}' exited mid-call", dispatched=True)
        return await rpc_task
    finally:
        watch_task.cancel()
        if not rpc_task.done():
            rpc_task.cancel()
        await asyncio.gather(rpc_task, watch_task, return_exceptions=True)


# ---------------------------------------------------------- result rendering

def _error_result_text(result) -> str:
    """Concatenated text of an ``isError`` result's blocks (EmbeddedResource error payloads
    carry text under ``.resource.text``)."""
    texts = (getattr(b, "text", None) or getattr(getattr(b, "resource", None), "text", None) for b in (result.content or []))
    return "".join(str(t) for t in texts if t)


def _render_content_blocks(result, server_name: str) -> Tuple[str, int]:
    """Text passes through; image/audio blocks are cached (MEDIA: tags); resource blocks are
    materialized rather than silently dropped; unsupported blocks become an inline drop notice
    (kimi-code#3227). Returns ``(text, usable_parts)`` — the count of REAL rendered blocks
    (whitespace-only text and drop notices excluded) that the structuredContent arbitration uses."""
    parts: List[str] = []
    usable_parts = 0
    # MCP tool results can also include ImageContent blocks (screenshot / Blockbench / Playwright etc.);
    # cache those via the gateway's image-cache helper so they flow through Hermes' MEDIA: tag convention
    # and out to messaging adapters that render images natively. Without this, image blocks were silently
    # dropped and the agent got an empty response. Distilled from #17915 (c3115644151) and #10848
    # (gnanirahulnutakki), both too stale to cherry-pick. #10848's approach (integrate with Hermes' MEDIA
    # tag + cache_image_from_bytes) was the cleaner of the two — plugs into existing infrastructure.
    for block in (result.content or []):
        if getattr(block, "text", None):
            parts.append(strip_unicode_tags(block.text))
            if block.text.strip():
                usable_parts += 1
            continue
        rendered = _cache_mcp_image_block(block) or _cache_mcp_audio_block(block) or _render_mcp_resource_block(block, server_name)
        if rendered:
            parts.append(rendered)
            usable_parts += 1
            continue
        block_type = getattr(block, "type", None) or type(block).__name__
        if block_type in {"text", "resource", "audio", "image"}:  # benign empty render
            logger.debug("MCP %s: content block type %r rendered empty", server_name, block_type)
        else:
            logger.warning("MCP %s: dropping unsupported content block type %r", server_name, block_type)
            # Surface the drop to the MODEL, not just the log: a silent drop leaves the agent
            # believing the tool returned less than it did, with no way to recover.
            parts.append(_render_mcp_dropped_block_notice(block, block_type))
    # Hard-cap pathological payloads; ordinary large results pass to spillover.
    return _truncate_mcp_text_result("\n".join(parts)), usable_parts


def _capped_structured_content(result):
    """``structuredContent`` (or None); over the hard cap it degrades to the head+tail
    truncated JSON string (multi-MB JSON flood guard)."""
    # Hard-cap pathological payloads before they propagate (#56059); ordinary large results pass untouched
    # to the spillover layer.
    # content and structuredContent are ALTERNATIVES — never both forwarded (ported from
    # MoonshotAI/kimi-code#3234). Spec-following servers already render their data into content (the
    # verbatim dual-emit SHOULD, or a faithful human reorganisation), so forwarding both sent the same
    # information to the model twice. content wins whenever it rendered anything usable; there is no
    # reliable signal that the structured payload is richer than what the server put in content (semantic
    # equality misses faithful reorganisations, size ratios misjudge both directions), so no heuristic is
    # attempted. structuredContent fills in only when the content blocks rendered effectively empty, which
    # keeps structuredContent-only servers working. Server-level `_meta` is also surfaced (ported from
    # MoonshotAI/kimi-code#2596): servers return namespaced metadata there (validated contracts,
    # browser-handoff payloads, ...) that was previously invisible to the agent. Protocol-reserved keys are
    # dropped first (kimi-code#2600) — per the MCP spec's key-name rules a prefix is reserved when a
    # `modelcontextprotocol` or `mcp` label is followed by at least one more label (e.g.
    # `modelcontextprotocol.io/...`, `tools.mcp.com/...`); those carry host/protocol plumbing, not
    # model-facing data. Unprefixed and vendor-namespaced keys (`com.example.mcp/...`) pass through — their
    # semantics belong to the server.
    structured = mcp_field(result, "structured_content", "structuredContent")
    try:
        as_json = json.dumps(structured, ensure_ascii=False, default=str) if structured is not None else ""
    except (TypeError, ValueError):
        return structured
    return _truncate_mcp_text_result(as_json) if len(as_json) > _MCP_HARD_RESULT_CAP_CHARS else structured


def _render_call_tool_result(result, server_name: str) -> str:
    """Pure: ``CallToolResult`` -> handler JSON. ``content`` and ``structuredContent`` are
    ALTERNATIVES, never both forwarded (kimi-code#3234): spec-following servers already render
    their data into content, so forwarding both sent it twice. content wins whenever it rendered
    anything usable (no richness heuristic is attempted — none is reliable); structuredContent
    fills in only when the blocks rendered effectively empty, keeping structuredContent-only
    servers working. ``_meta`` minus reserved keys is always surfaced."""
    if mcp_field(result, "is_error", "isError", False):
        return tool_error(_sanitize_error(_truncate_mcp_text_result(_error_result_text(result) or "MCP tool returned an error")))
    text_result, usable_parts = _render_content_blocks(result, server_name)
    structured = _capped_structured_content(result)
    meta = _strip_reserved_meta_keys(mcp_field(result, "meta", "meta"))
    if structured is not None and usable_parts > 0:
        structured = None  # drop notices do not count as usable content
    if structured is None and meta is None:
        return json.dumps({"result": text_result}, ensure_ascii=False)
    # Key order is part of the output: "result" leads when there is text, otherwise "_meta" precedes it.
    payload: Dict[str, Any] = {"result": text_result} if text_result else {}
    # Cap structuredContent too — a malicious server could flood context via a multi-MB JSON payload
    # (#56059). When the serialized form exceeds the hard cap, replace it with the truncated string (head +
    # tail preserved) so it degrades gracefully instead of flooding downstream.
    if structured is not None:
        payload["structuredContent" if text_result else "result"] = structured
    if meta is not None:
        payload["_meta"] = meta
    payload.setdefault("result", text_result)
    try:
        return json.dumps(payload, ensure_ascii=False)
    except (TypeError, ValueError):  # Non-serializable metadata: drop the extras, keep the call.
        return json.dumps({"result": text_result}, ensure_ascii=False)


def _make_tool_handler(
        server_name: str, tool_name: str, tool_timeout: float, *, allow_replay: bool = True):
    """Build a synchronous MCP tool handler.

    ``allow_replay=False`` is an operator-side hard floor used by trusted
    plugin wrappers for financial/non-idempotent calls.  Unlike discovery
    metadata it cannot be overwritten by a lazy reconnect's upstream manifest.
    """
    op = f"tools/call {tool_name}"

    def _handler(args: dict, **kwargs) -> str:
        # Security boundary: untrusted-server write tools need approval before ANY transport work (incl. lazy spawn).
        error = _trust_gate_check(server_name, tool_name) or _check_circuit_breaker(server_name)
        if error is not None:
            return error
        server, error = _acquire_call_server(server_name, tool_timeout)
        if server is None:
            return error

        async def _call():
            async with server._rpc_lock, _track_inflight_rpc(server, server_name, op):
                server._pending_call_context = contextvars.copy_context()  # for the elicitation callback
                try:
                    result = await _call_tool_racing_stdio_death(server, server_name, tool_name, args)
                finally:
                    server._pending_call_context = None
            if getattr(server, "_mark_session_proven", None) is not None:  # round-trip done: transport healthy
                server._mark_session_proven()
            return _render_call_tool_result(result, server_name)

        def _on_failure(exc):
            _core._bump_server_error(server_name)
            logger.error("MCP tool %s/%s call failed: %s", server_name, tool_name, exc)
        recoverers = (
            _handle_stdio_child_exited_and_retry, _handle_auth_error_and_retry,
            _handle_session_expired_and_retry)
        # Only an explicit false disables recovery replay.  ``True`` and
        # missing/invalid annotations intentionally retain legacy behaviour.
        if (allow_replay is False
                or _core._tool_idempotent_hints.get(server_name, {}).get(tool_name) is False):
            recoverers = tuple(partial(
                recover, allow_replay=False, tool_name=tool_name) for recover in recoverers)
            recoverers += (partial(
                _handle_non_idempotent_transport_without_replay, tool_name=tool_name),)
        return _dispatch(
            server_name, server, op, _call, tool_timeout,
            recoverers,
            _on_failure, record_outcome=True)
    return _handler


def _make_utility_handler(op: str, log_label: str, rpc, render, required: Optional[str] = None):
    """``(server_name, tool_timeout) -> sync handler`` for one utility tool: ``rpc(session, args,
    server_name)`` awaited under ``_rpc_lock``, ``render(result, server_name)`` -> JSON-able
    payload, ``required`` validated before any transport work."""
    def _factory(server_name: str, tool_timeout: float):
        def _handler(args: dict, **kwargs) -> str:
            from tools import mcp_tool_discovery as _discovery  # lazy: import cycle
            server = _discovery._get_connected_server_for_call(server_name)
            if not server or not server.session:
                return tool_error(f"MCP server '{server_name}' is not connected")
            if required and not args.get(required):
                return tool_error(f"Missing required parameter '{required}'")

            async def _call():
                async with server._rpc_lock:
                    result = await rpc(server.session, args, server_name)
                return json.dumps(render(result, server_name), ensure_ascii=False)
            return _dispatch(
                server_name, server, op, _call, tool_timeout,
                (_handle_auth_error_and_retry, _handle_session_expired_and_retry),
                lambda exc: logger.error("MCP %s/%s failed: %s", server_name, log_label, exc))
        return _handler
    return _factory


def _pick(obj, *specs) -> dict:
    """``{out_key: value}`` for each ``(out_key, attr[, truthy])`` present on *obj* (presence check so SDK models
    and stubs behave alike; ``truthy`` also skips falsy). Key order = spec order."""
    entry = {}
    for out_key, attr, *truthy in specs:
        value = getattr(obj, attr, _MISSING)
        if value is not _MISSING and (value or not (truthy and truthy[0])):
            entry[out_key] = value
    return entry


def _render_resource_list(all_resources, server_name: str) -> dict:
    resources = []
    for r in all_resources:
        entry = _pick(r, ("uri", "uri"), ("name", "name"), ("description", "description", True))
        if "uri" in entry:
            entry["uri"] = str(entry["uri"])
        mime = mcp_field(r, "mime_type", "mimeType")
        if mime:
            entry["mimeType"] = mime  # camelCase: this is the tool's own JSON output shape
        resources.append(entry)
    return {"resources": resources}


def _render_read_resource(result, server_name: str) -> dict:
    parts: List[str] = []
    for block in getattr(result, "contents", []):
        if getattr(block, "text", None) is not None:
            parts.append(strip_unicode_tags(block.text))
        elif getattr(block, "blob", None) is not None:  # binary -> document cache, like EmbeddedResource blocks
            rendered = _render_mcp_resource_block(SimpleNamespace(type="resource", resource=block), server_name)
            parts.append(rendered or f"[binary data, {len(block.blob)} bytes]")
    return {"result": "\n".join(parts)}


def _render_prompt_list(all_prompts, server_name: str) -> dict:
    prompts = []
    for p in all_prompts:
        entry = _pick(p, ("name", "name"), ("description", "description", True))
        if getattr(p, "arguments", None):
            entry["arguments"] = [{"name": a.name, **_pick(a, ("description", "description", True), ("required", "required"))}
                                  for a in p.arguments]
        prompts.append(entry)
    return {"prompts": prompts}


def _render_get_prompt(result, server_name: str) -> dict:
    messages = []
    for msg in getattr(result, "messages", []):
        entry = _pick(msg, ("role", "role"))
        if hasattr(msg, "content"):
            entry["content"] = strip_unicode_tags(msg.content.text if hasattr(msg.content, "text") else str(msg.content))
        messages.append(entry)
    return {"messages": messages, **_pick(result, ("description", "description", True))}


_make_list_resources_handler = _make_utility_handler(
    "resources/list", "list_resources",
    lambda session, args, sn: _core._paginate_full_list(session.list_resources, "resources", sn), _render_resource_list)
_make_read_resource_handler = _make_utility_handler(
    "resources/read", "read_resource",
    lambda session, args, sn: session.read_resource(args["uri"]), _render_read_resource, required="uri")
_make_list_prompts_handler = _make_utility_handler(
    "prompts/list", "list_prompts",
    lambda session, args, sn: _core._paginate_full_list(session.list_prompts, "prompts", sn), _render_prompt_list)
_make_get_prompt_handler = _make_utility_handler(
    "prompts/get", "get_prompt",
    lambda session, args, sn: session.get_prompt(args["name"], arguments=args.get("arguments", {})),
    _render_get_prompt, required="name")


def _make_check_fn(server_name: str):
    """Connection-alive check; lazy (schema-cache registered) servers count as available."""
    def _check() -> bool:
        with _core._lock:
            server = _core._servers.get(server_name)
            return ((server is not None and (server.session is not None or server._is_recycled_stdio()))
                    or server_name in _core._lazy_server_configs)
    return _check
