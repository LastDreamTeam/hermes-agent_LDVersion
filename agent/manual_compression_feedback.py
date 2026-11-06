"""User-facing summaries for manual compression commands."""

from __future__ import annotations

from typing import Any, Sequence

from agent.redact import redact_sensitive_text
from agent.i18n import zh_text


def describe_compression_lock_skip(lock_signal: Any) -> str:
    """User-facing text for a manual /compress skipped by the compression lock.

    ``lock_signal`` is a holder string when another compressor CONFIRMED holds
    the lock, else ``True``/``None``. A failed acquire is NOT proof another
    compression is running (``try_acquire_compression_lock`` swallows
    ``sqlite3.Error``), so the two cases are worded differently.
    """
    if isinstance(lock_signal, str) and lock_signal.strip():
        return (
            zh_text(
                "⏳ Compression already in progress for this session "
                "(holder: {holder}). Please wait for it to finish.",
                "⏳ 此会话正在压缩（占用者：{holder}），请等待完成。",
                holder=lock_signal,
            )
        )
    return zh_text(
        "⏳ Compression skipped: could not acquire this session's compression lock. Another compression may "
        "still be running, or the lock check failed — try again shortly.",
        "⏳ 已跳过压缩：无法取得此会话的压缩锁。可能仍有压缩任务在运行，或锁检查失败；请稍后重试。",
    )


def summarize_manual_compression(
    before_messages: Sequence[dict[str, Any]],
    after_messages: Sequence[dict[str, Any]],
    before_tokens: int,
    after_tokens: int,
    *,
    compression_state: Any = None,
) -> dict[str, Any]:
    """Consistent user-facing feedback (headline, token line, optional note) for manual compression."""
    before_count = len(before_messages)
    after_count = len(after_messages)
    noop = list(after_messages) == list(before_messages)

    def flag(name: str) -> bool:
        return getattr(compression_state, name, False) is True

    aborted = flag("_last_compress_aborted")
    refused_would_grow = flag("_last_compress_refused_would_grow")
    fallback_used = flag("_last_summary_fallback_used")
    failure_reason = getattr(compression_state, "_last_summary_error", None)
    if not isinstance(failure_reason, str) or not failure_reason.strip():
        failure_reason = None

    note = None
    if refused_would_grow:
        headline = zh_text(
            "Compression refused (summary would grow the conversation): {count} messages preserved",
            "已拒绝压缩（摘要反而会增大上下文）：保留 {count} 条消息",
            count=before_count,
        )
        note = zh_text(
            "The generated summary was larger than what it would replace; no messages were removed.",
            "生成的摘要比原内容更大，因此没有移除任何消息。",
        )
    elif aborted:
        headline = zh_text(
            "Compression aborted: {count} messages preserved", "压缩已中止：保留 {count} 条消息",
            count=before_count,
        )
        note = zh_text("Summary generation failed; no messages were removed.", "摘要生成失败，没有移除任何消息。")
    elif fallback_used:
        headline = zh_text(
            "Compressed with fallback: {before} → {after} messages", "已使用回退方案压缩：{before} → {after} 条消息",
            before=before_count, after=after_count,
        )
        dropped_count = getattr(compression_state, "_last_summary_dropped_count", None)
        if not isinstance(dropped_count, int) or isinstance(dropped_count, bool):
            dropped_count = max(before_count - after_count, 0)
        note = zh_text(
            "Summary generation failed; Hermes used limited fallback context and removed {count} message(s).",
            "摘要生成失败；Hermes 使用了受限的回退上下文，并移除 {count} 条消息。",
            count=dropped_count,
        )
    elif noop:
        headline = zh_text(
            "No changes from compression: {count} messages", "压缩后无变化：{count} 条消息", count=before_count,
        )
    else:
        headline = zh_text(
            "Compressed: {before} → {after} messages", "已压缩：{before} → {after} 条消息",
            before=before_count, after=after_count,
        )
        if after_count < before_count and after_tokens > before_tokens:
            note = zh_text(
                "Note: fewer messages can still raise this estimate when compression rewrites the transcript into denser summaries.",
                "注意：压缩把对话改写为更密集的摘要时，即使消息更少，估算令牌数仍可能上升。",
            )

    if (noop and after_tokens == before_tokens) or refused_would_grow:
        token_line = zh_text(
            "Approx request size: ~{before:,} tokens (unchanged)", "估算请求大小：约 {before:,} 个令牌（未变化）",
            before=before_tokens,
        )
    else:
        token_line = zh_text(
            "Approx request size: ~{before:,} → ~{after:,} tokens",
            "估算请求大小：约 {before:,} → 约 {after:,} 个令牌",
            before=before_tokens, after=after_tokens,
        )

    if failure_reason and (aborted or fallback_used):
        # Crosses a user-facing UI boundary: never let a disabled global redaction
        # preference expose credentials embedded in provider exception text.
        reason_label = zh_text("Reason:", "原因：")
        note = f"{note} {reason_label} {redact_sensitive_text(failure_reason.strip(), force=True)}"

    return {
        "noop": noop,
        "aborted": aborted,
        "refused_would_grow": refused_would_grow,
        "fallback_used": fallback_used,
        "headline": headline,
        "token_line": token_line,
        "note": note,
    }
