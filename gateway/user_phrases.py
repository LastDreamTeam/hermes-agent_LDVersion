"""Profile-aware, fail-closed localization for user-visible Gateway status text.

The canonical English strings stay in this module because several Hermes paths use English
messages as event/matching keys.  Localization happens only at the final display boundary.
Profiles may override known templates in ``<HERMES_HOME>/display_phrases.yaml`` or under
``display.phrases`` / ``display.platforms.<platform>.phrases``.  Files are read on demand so a
profile can tune its own voice without restarting the Gateway.

Invalid, empty, oversized, or placeholder-changing overrides fall back to the canonical English
template.  This is deliberate: a cosmetic mistake must never suppress a warning or leak a raw
``{placeholder}`` to the user.
"""

from __future__ import annotations

import os
import string
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from hermes_constants import get_hermes_home

_MAX_FILE_BYTES = 64 * 1024
_MAX_TEMPLATE_CHARS = 1600

# Canonical English display strings.  Keep these intact even when a translated surface is used.
ENGLISH_TEMPLATES: dict[str, str] = {
    "busy.steer": "⏩ Steered into current run{details}. Your message arrives after the next tool call.",
    "busy.redirect": "↪ Redirected current run{details}. I'll adjust using your correction.",
    "busy.subagent": (
        "⏳ Subagent working{details} — your message is queued for when it finishes "
        "(use /stop to cancel everything)."
    ),
    "busy.compress": (
        "⏳ Compressing context{details} — your message is queued for when it finishes "
        "(use /stop to cancel everything)."
    ),
    "busy.queue": "⏳ Queued for the next turn{details}. I'll respond once the current task finishes.",
    "busy.interrupt": "⚡ Interrupting current task{details}. I'll respond to your message shortly.",
    "heartbeat.working": "⏳ Working — {minutes} min{details}",
    "shutdown.stop": "⚠️ Gateway shutting down — Your current task will be interrupted.",
    "shutdown.restart": (
        "⚠️ Gateway restarting — Your current task will be interrupted. "
        "Send any message after restart and I'll try to resume where you left off."
    ),
    "shutdown.cron_interrupted_stop": (
        "⚠️ Cron job '{job_name}' was interrupted — the gateway is shutting down and killed the run "
        "before it finished. No result was produced for this run."
    ),
    "shutdown.cron_interrupted_restart": (
        "⚠️ Cron job '{job_name}' was interrupted — the gateway is restarting and killed the run "
        "before it finished. No result was produced for this run."
    ),
    "startup.online": "♻️ Gateway online — Hermes is back and ready.",
    "memory.updated": "💾 Memory updated",
    "memory.review": "💾 Self-improvement review: {summary}",
    "provider.authentication_failed": (
        "⚠️ Provider authentication failed. Check the configured account authorization or API key; "
        "raw provider details are in the Gateway logs."
    ),
    "skill.updated": "✅ Skill updated",
    "skill.failed": "✗ Skill update failed",
    "session.started": "✨ New session started!",
    "session.started_titled": "✨ New session started: {title}",
    "onboarding.busy.queue": (
        "💡 First-time tip — I queued your message instead of interrupting. Send `/busy interrupt` "
        "to make new messages stop the current task immediately, or `/busy status` to check. "
        "This notice won't appear again."
    ),
    "onboarding.busy.steer": (
        "💡 First-time tip — I steered your message into the current run; it will arrive after the "
        "next tool call instead of interrupting. Send `/busy interrupt` or `/busy queue` to change "
        "this, or `/busy status` to check. This notice won't appear again."
    ),
    "onboarding.busy.redirect": (
        "💡 First-time tip — I redirected the current run using your message. Completed work stays "
        "in context, and `/stop` still cancels the task. Send `/busy queue` to wait for a separate "
        "turn, or `/busy status` to check. This notice won't appear again."
    ),
    "onboarding.busy.interrupt": (
        "💡 First-time tip — I just interrupted my current task to answer you. Send `/busy queue` "
        "to queue follow-ups for after the current task instead, `/busy steer` to inject them "
        "mid-run without interrupting, or `/busy status` to check. This notice won't appear again."
    ),
}

ZH_TEMPLATES: dict[str, str] = {
    "busy.steer": "⏩ 收到，已经补进手头这件事{details}。下一次工具调用后会带上。",
    "busy.redirect": "↪ 收到，方向已经按你的纠正调整{details}。",
    "busy.subagent": "⏳ 队友还在忙{details}。你的消息已记下，等它完成后接着处理；发 /stop 可全部停止。",
    "busy.compress": "⏳ 我在整理上下文{details}。你的消息已记下，整理完就接着处理；发 /stop 可全部停止。",
    "busy.queue": "⏳ 我记下了{details}。忙完手上这件就回复你。",
    "busy.interrupt": "⚡ 我先停一下手上的任务{details}，马上看你的新消息。",
    "heartbeat.working": "⌛ 还在处理 —— 已 {minutes} 分钟{details}",
    "shutdown.stop": "⚠️ Gateway 即将停止——当前任务会被打断。",
    "shutdown.restart": "⚠️ Gateway 即将重启——当前任务会被打断。重启后发任意消息，我会尝试接着处理。",
    "shutdown.cron_interrupted_stop": "⚠️ 定时任务“{job_name}”已中断——Gateway 正在停止，本轮没有生成结果。",
    "shutdown.cron_interrupted_restart": "⚠️ 定时任务“{job_name}”已中断——Gateway 正在重启，本轮没有生成结果。",
    "startup.online": "♻️ Gateway 已恢复，Hermes 可以继续工作了。",
    "memory.updated": "💾 记忆已更新",
    "memory.review": "💾 自我回顾：{summary}",
    "provider.authentication_failed": (
        "⚠️ 模型服务认证失败，请检查已配置的账号授权或 API 密钥；原始错误已写入 Gateway 日志。"
    ),
    "skill.updated": "✅ 技能已更新",
    "skill.failed": "✗ 技能更新失败",
    "session.started": "✨ 新会话已开始！",
    "session.started_titled": "✨ 新会话已开始：{title}",
    "onboarding.busy.queue": (
        "💡 小提示（只显示一次）——你的消息已排到下一轮，没有打断当前任务。"
        "发 `/busy interrupt` 可改为直接打断，发 `/busy status` 可查看当前模式。"
    ),
    "onboarding.busy.steer": (
        "💡 小提示（只显示一次）——你的消息已并入当前任务，会在下一次工具调用后生效。"
        "发 `/busy interrupt` 或 `/busy queue` 可切换模式，发 `/busy status` 可查看。"
    ),
    "onboarding.busy.redirect": (
        "💡 小提示（只显示一次）——当前任务已按你的消息调整，已完成的内容仍会保留。"
        "发 `/stop` 可取消，发 `/busy queue` 可改为排到下一轮。"
    ),
    "onboarding.busy.interrupt": (
        "💡 小提示（只显示一次）——刚才为了回复你，当前任务被打断了。"
        "发 `/busy queue` 可改为排队，发 `/busy steer` 可改为不中断地并入当前任务。"
    ),
}

_TOOL_NAMES_ZH: dict[str, str] = {
    "terminal": "执行命令",
    "execute_code": "执行代码",
    "a2a_call": "找队友问话",
    "message_agent": "给队友发消息",
    "delegate_task": "派任务给队友",
    "web_search": "搜索网页",
    "web_fetch": "读取网页",
    "browser": "操作浏览器",
    "computer": "操作电脑",
    "read_file": "读取文件",
    "write_file": "写入文件",
    "apply_patch": "修改文件",
    "image_generate": "生成图片",
    "image_gen": "生成图片",
    "text_to_speech": "合成语音",
    "session_search": "检索历史会话",
    "kanban": "更新任务看板",
}


def _normalise_language(value: Any) -> str:
    raw = str(value or "").strip().lower().replace("_", "-")
    if raw in {"zh", "zh-cn", "zh-hans", "chinese", "mandarin"}:
        return "zh"
    return "en"


def _owner_language_from_profile_home(home: Path | None) -> str | None:
    """Read the owner language for a conventional ``<owner>/profiles/<name>`` profile."""
    try:
        profile_home = Path(home) if home is not None else None
        if profile_home is None or profile_home.parent.name != "profiles":
            return None
        owner_config = profile_home.parent.parent / "config.yaml"
        if not owner_config.is_file() or owner_config.stat().st_size > _MAX_FILE_BYTES:
            return None
        loaded = yaml.safe_load(owner_config.read_text(encoding="utf-8")) or {}
        display = loaded.get("display") if isinstance(loaded, Mapping) else None
        return _normalise_language(display.get("language")) if isinstance(display, Mapping) and display.get("language") else None
    except Exception:
        return None


def resolve_language(user_config: Mapping[str, Any] | None, *, home: Path | None = None) -> str:
    env = os.environ.get("HERMES_LANGUAGE")
    if env:
        return _normalise_language(env)
    display = user_config.get("display") if isinstance(user_config, Mapping) else None
    if isinstance(display, Mapping) and display.get("language"):
        return _normalise_language(display.get("language"))
    inherited = _owner_language_from_profile_home(home)
    if inherited:
        return inherited
    return "en"


def humanize_tool_name(tool_name: Any, language: str) -> str:
    raw = str(tool_name or "").strip()
    if language != "zh" or not raw:
        return raw
    return _TOOL_NAMES_ZH.get(raw.lower(), raw)


def _flatten(node: Any, prefix: str = "", out: dict[str, str] | None = None) -> dict[str, str]:
    result = {} if out is None else out
    if isinstance(node, Mapping):
        for key, value in node.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            _flatten(value, name, result)
    elif isinstance(node, str):
        result[prefix] = node
    return result


def _known_overrides(section: Any) -> dict[str, str]:
    if not isinstance(section, Mapping):
        return {}
    raw = section.get("phrases") if isinstance(section.get("phrases"), Mapping) else section
    flat = _flatten(raw)
    return {key: value for key, value in flat.items() if key in ENGLISH_TEMPLATES}


def _file_overrides(home: Path) -> dict[str, str]:
    path = home / "display_phrases.yaml"
    try:
        if not path.is_file() or path.stat().st_size > _MAX_FILE_BYTES:
            return {}
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return _known_overrides(loaded)


def _config_overrides(user_config: Mapping[str, Any] | None, platform_key: str | None) -> dict[str, str]:
    display = user_config.get("display") if isinstance(user_config, Mapping) else None
    if not isinstance(display, Mapping):
        return {}
    merged = _known_overrides(display.get("phrases"))
    platforms = display.get("platforms")
    if platform_key and isinstance(platforms, Mapping) and isinstance(platforms.get(platform_key), Mapping):
        merged.update(_known_overrides(platforms[platform_key].get("phrases")))
    return merged


def _template_fields(template: str) -> set[str] | None:
    if not template or len(template) > _MAX_TEMPLATE_CHARS:
        return None
    fields: set[str] = set()
    try:
        for _literal, field, format_spec, conversion in string.Formatter().parse(template):
            if field is None:
                continue
            # Attribute/index traversal and format mini-language are unnecessary here and make
            # validation harder.  Profile templates are intentionally plain named placeholders.
            if not field or any(ch in field for ch in ".[]") or format_spec or conversion:
                return None
            fields.add(field)
    except ValueError:
        return None
    return fields


def _valid_override(key: str, template: str) -> bool:
    fields = _template_fields(template)
    canonical = _template_fields(ENGLISH_TEMPLATES[key])
    return fields is not None and canonical is not None and fields == canonical


def resolve_user_phrase(
    key: str,
    *,
    user_config: Mapping[str, Any] | None = None,
    platform_key: str | None = None,
    home: Path | None = None,
    **values: Any,
) -> str:
    """Render one known display phrase with profile/platform overrides and strict fallback."""
    english = ENGLISH_TEMPLATES.get(key, key)
    profile_home = Path(home or get_hermes_home())
    language = resolve_language(user_config, home=profile_home)
    template = ZH_TEMPLATES.get(key, english) if language == "zh" else english
    overrides = _file_overrides(profile_home)
    overrides.update(_config_overrides(user_config, platform_key))
    if key in overrides:
        candidate = overrides[key]
        # Requirement: a malformed profile override falls all the way back to the preserved English
        # original, rather than risking an incomplete warning in a translated template.
        template = candidate if _valid_override(key, candidate) else english
    try:
        return template.format(**values)
    except (KeyError, IndexError, ValueError):
        try:
            return english.format(**values)
        except Exception:
            return english


def format_busy_details(
    *, user_config: Mapping[str, Any] | None, elapsed_minutes: int | None,
    iteration: int | None, max_iterations: int | None, tool_name: Any, home: Path | None = None,
) -> str:
    """Parenthesized activity detail for busy acks; all available runtime parameters survive."""
    language = resolve_language(user_config, home=home)
    parts: list[str] = []
    if elapsed_minutes is not None and elapsed_minutes > 0:
        parts.append(f"已运行 {elapsed_minutes} 分钟" if language == "zh" else f"{elapsed_minutes} min elapsed")
    if max_iterations:
        parts.append(
            f"第 {int(iteration or 0)}/{int(max_iterations)} 步"
            if language == "zh" else f"iteration {int(iteration or 0)}/{int(max_iterations)}"
        )
    if tool_name:
        tool = humanize_tool_name(tool_name, language)
        parts.append(f"正在：{tool}" if language == "zh" else f"running: {tool}")
    if not parts:
        return ""
    return f"（{'，'.join(parts)}）" if language == "zh" else f" ({', '.join(parts)})"


def format_heartbeat_details(
    *, user_config: Mapping[str, Any] | None, iteration: int | None,
    max_iterations: int | None, tool_name: Any, home: Path | None = None,
) -> str:
    language = resolve_language(user_config, home=home)
    parts: list[str] = []
    if max_iterations:
        parts.append(
            f"第 {int(iteration or 0)}/{int(max_iterations)} 步"
            if language == "zh" else f"iteration {int(iteration or 0)}/{int(max_iterations)}"
        )
    if tool_name:
        tool = humanize_tool_name(tool_name, language)
        parts.append(f"正在：{tool}" if language == "zh" else str(tool))
    if not parts:
        return ""
    return " —— " + ("，".join(parts) if language == "zh" else ", ".join(parts))


__all__ = [
    "ENGLISH_TEMPLATES", "ZH_TEMPLATES", "format_busy_details", "format_heartbeat_details",
    "humanize_tool_name", "resolve_language", "resolve_user_phrase",
]
