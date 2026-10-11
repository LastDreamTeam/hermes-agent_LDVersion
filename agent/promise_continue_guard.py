"""High-precision, text-only candidate inspector for Hermes' final-response guard.

This module is intentionally standalone so the frozen corpus can exercise it without
importing the Hermes runtime.  It does not grant authorization or execute anything.
"""

from __future__ import annotations

import re
from typing import Dict


READY = "READY"
WAIT_AUTH = "WAIT_AUTH"
PAST_TENSE = "PAST_TENSE"
NEGATED = "NEGATED"
QUOTED = "QUOTED"
AMBIGUOUS = "AMBIGUOUS"
NO_CANDIDATE = "NO_CANDIDATE"

_EN_IMMEDIATE_RE = re.compile(
    r"(?:\blet me now\b|\bi(?:['\u2019])?ll now\b|\bi will now\b"
    r"|\bnow i(?:['\u2019]ll| will)\b|\bnext[,:] i\b)"
    r"[^.!?\n]{0,100}[.:\u2026]?\s*$",
    re.IGNORECASE,
)

_EN_AUTH_RE = re.compile(
    r"\b(?:if you (?:agree|approve|want|wish|authorize|confirm)|"
    r"once you (?:agree|approve|authorize|confirm)|"
    r"after you (?:agree|approve|authorize|confirm)|"
    r"waiting for your (?:approval|authorization|confirmation|decision)|"
    r"would you like me to|shall i|may i|can i)\b",
    re.IGNORECASE,
)

_ZH_ACTION = (
    r"发送|转发|通知|转告|联系|询问|回复|提醒|跟进|调用|运行|执行|"
    r"启动|重启|停止|创建|新建|建立|修改|更新|部署|提交|上传|下载|"
    r"安装|配置|修复|接入|删除|清理|同步|写入|保存|移动|复制|打开|关闭|"
    r"检查|查看|查询|查证|检索|搜索|读取|核对|确认|验证|测试|排查|调查|"
    r"看一眼|看一下|看下|"
    r"写(?=入|进|到|出|好|一|文|代|报|脚|档|记|给|下|$)|"
    r"查(?=一下|下|看|询|验|证|日志|配置|记录|原因|问题|清|明|QQ|群|$)|"
    r"发(?=送|给|出|布|起|一|通|消息|通知|邮件|语音|图片|报告|[。.!！…]|$)"
)

# The object bridge is intentionally short.  It supports natural phrases such as
# “我马上把通知发给队友” without allowing a verb from a later proposition to be
# captured as the promised action.
_ZH_IMMEDIATE_RE = re.compile(
    r"(?:^|[。，！？；,!?;\n]|[-—]{2,})"
    r"(?P<lead>[^。！？；!?：:;\n]{0,12}?)"
    r"(?:"
    r"我(?P<urgency>这就|现在就|马上(?:就)?|立即(?:就)?|先)"
    r"|(?:接下来|下一步)(?:[，,:：]\s*)?我(?:会|要|来|就|将|准备)?"
    r")"
    r"(?!(?:不|无法|不能|没法|暂不))"
    r"(?:去|来|把|给你|帮你|替你)?"
    r"(?P<object>[^。，！？；,!?：:;\n]{0,16}?)"
    rf"(?P<action>{_ZH_ACTION})"
    r"(?![了过完好着])"
    r"(?P<rest>[^。！？!?\n]{0,100})"
    r"(?:[。.!！…？?]+|$)(?=[\s\S]{0,160}\Z)"
)

_ZH_PRE_AUTH_RE = re.compile(
    r"如果(?:你|主人|需要)|若(?:你|主人|需要)|要是(?:你|主人|需要)|"
    r"要不要|要我|需要我|需不需要|"
    r"(?:等|待)(?:你|主人)|你(?:点头|同意|确认|授权|批准|拍板)(?:.{0,8}(?:后|再))?|"
    r"(?:确认|授权|批准|同意)后|"
    r"(?:跑完|完成|执行完|处理完|重启完).{0,24}(?:后|告诉我)|"
    r"告诉我.{0,16}(?:我(?:这就|马上|现在就|立即))"
)

_ZH_POST_AUTH_RE = re.compile(
    r"要不要我|要我.{0,16}(?:吗|[?？])|"
    r"(?:等|待)(?:你|主人).{0,16}(?:点头|同意|确认|授权|批准|回复|决定|选择)|"
    r"(?:你|主人).{0,16}(?:点头|同意|确认|授权|批准|拍板|说一声|决定|选择|倾向哪个)|"
    r"(?:可以|行|好|同意|允许|批准|确认|授权)(?:吗|[?？])|"
    r"(?:请|需要|需)(?:你|主人).{0,8}(?:确认|授权|批准|决定|选择)|"
    r"(?:点头|同意|确认|授权|批准)后(?:再|才)|"
    r"我才(?:碰|做|处理|执行|开始|发送|修改|配置|部署|删除|重启)"
)

_ZH_GLOBAL_AUTH_RE = re.compile(
    r"要不要我|要我现在|你说一声|你点头|等你(?:确认|授权|批准|回复|决定)|"
    r"待你(?:确认|授权|批准|回复|决定)|你倾向哪个|你选哪个|你来决定|"
    r"如果你(?:同意|允许|确认|授权|批准)|(?:可以|行|好)(?:吗|[?？])|"
    r"要(?:现在)?(?:开始|执行|处理|做|安装|配置|修改|发送|检查).{0,8}(?:吗|[?？])|"
    r"我等你|你直接说(?:派|做|开始|执行)"
)

_ZH_NEGATED_RE = re.compile(r"(?:我(?:这就|现在就|马上|立即|先))(?:不|无法|不能|没法|暂不)")
_ZH_INLINE_DELIVERY_RE = re.compile(
    r"(?:发|写|列|说明).{0,16}(?:在这里|如下|给你).{0,12}[：:]|"
    r"(?:完整说明|检查结果|结论)(?:如下|是)[：:]"
)

_ZH_QUERY_OBJECT_RE = re.compile(r"(?:到底)?(?:支不支持|能不能|有没有|是否|为何|为什么|怎么|如何)")
_ZH_LOCAL_COMPLETED_RE = re.compile(
    r"^[^，,；;]{0,24}(?:了|过|完(?:成)?|好了|完毕)(?:[，,；;]|$)"
)

# A later, separately authorized high-risk action must not cancel an already announced
# read-only inspection: “我先检查日志，如果需要重启，再请你授权。”
_ZH_LATER_AUTH_FOR_OTHER_ACTION_RE = re.compile(
    r"[，,；;]\s*(?:如果|若|要是).{0,24}"
    r"(?:重启|停止|删除|修改|写入|发送|部署).{0,24}"
    r"(?:再)?请(?:你|主人).{0,8}(?:确认|授权|批准)"
)


def _plain_intent_text(text: str) -> str:
    """Remove regions that quote an utterance instead of making it."""
    value = text or ""
    value = re.sub(r"```[\s\S]*?(?:```|\Z)", "", value)
    value = re.sub(r"`[^`\n]*(?:`|$)", "", value)
    value = re.sub(r"(?m)^\s*>.*$", "", value)
    value = re.sub(r"“[^”]*(?:”|$)|「[^」]*(?:」|$)|\"[^\"\n]*(?:\"|$)", "", value)
    return value.replace("**", "").replace("__", "")


def continuation_safety_veto_reason(text: str) -> str:
    """Return a conservative common veto usable by both stall and legacy paths."""
    raw = (text or "").strip()
    if not raw:
        return NO_CANDIDATE
    plain = _plain_intent_text(raw)
    if not plain.strip() and raw.strip():
        return QUOTED
    tail = plain[-240:]
    if _EN_AUTH_RE.search(tail):
        return WAIT_AUTH
    if _ZH_NEGATED_RE.search(tail):
        return NEGATED
    if _ZH_GLOBAL_AUTH_RE.search(tail):
        return WAIT_AUTH
    return ""


def inspect_continue_candidate(text: str) -> Dict[str, str]:
    """Classify only a high-confidence, immediate action near the reply tail."""
    raw = (text or "").strip()
    if not raw:
        return {"status": NO_CANDIDATE, "action": "", "candidate": ""}
    plain = _plain_intent_text(raw)
    if not plain.strip():
        return {"status": QUOTED, "action": "", "candidate": ""}

    tail_start = max(0, len(plain) - 1000)
    tail = ("" if tail_start == 0 else "\u200b") + plain[tail_start:]
    blocked = continuation_safety_veto_reason(plain)
    matches = list(_ZH_IMMEDIATE_RE.finditer(tail))
    for match in reversed(matches):
        candidate = match.group(0).strip()
        action = match.group("action")
        lead = match.group("lead") or ""
        rest = match.group("rest") or ""
        prefix_offset = match.group(0).find("我")
        absolute_prefix = match.start() + max(prefix_offset, 0)
        sentence_breaks = [tail.rfind(mark, 0, absolute_prefix) for mark in "。！？!?\n"]
        sentence_start = max(sentence_breaks) + 1
        scoped_before = tail[sentence_start:absolute_prefix]
        scoped_after = tail[absolute_prefix:]
        if _ZH_PRE_AUTH_RE.search(scoped_before) or _ZH_PRE_AUTH_RE.search(lead):
            return {"status": WAIT_AUTH, "action": action, "candidate": candidate}
        if _ZH_INLINE_DELIVERY_RE.search(candidate):
            return {"status": PAST_TENSE, "action": action, "candidate": candidate}
        if _ZH_LOCAL_COMPLETED_RE.search(rest):
            return {"status": PAST_TENSE, "action": action, "candidate": candidate}
        if candidate.rstrip().endswith(("?", "？")) and not (
            action.startswith(("查", "检索", "搜索", "询问"))
            and _ZH_QUERY_OBJECT_RE.search(candidate)
        ):
            return {"status": WAIT_AUTH, "action": action, "candidate": candidate}
        post_auth = _ZH_POST_AUTH_RE.search(scoped_after)
        if post_auth:
            other_action_auth = _ZH_LATER_AUTH_FOR_OTHER_ACTION_RE.search(scoped_after)
            if not other_action_auth:
                return {"status": WAIT_AUTH, "action": action, "candidate": candidate}
        if _ZH_GLOBAL_AUTH_RE.search(scoped_after):
            if not _ZH_LATER_AUTH_FOR_OTHER_ACTION_RE.search(scoped_after):
                return {"status": WAIT_AUTH, "action": action, "candidate": candidate}
        if blocked == NEGATED:
            return {"status": NEGATED, "action": action, "candidate": candidate}
        return {"status": READY, "action": action, "candidate": candidate}

    english_tail = plain[-160:]
    if len(plain) <= 400 and _EN_IMMEDIATE_RE.search(english_tail):
        if blocked in {WAIT_AUTH, NEGATED}:
            return {"status": blocked, "action": "english", "candidate": english_tail}
        return {"status": READY, "action": "english", "candidate": english_tail}
    if blocked:
        return {"status": blocked, "action": "", "candidate": ""}
    return {"status": NO_CANDIDATE, "action": "", "candidate": ""}


def trailing_continue_intent(text: str) -> bool:
    """Compatibility wrapper retained for the existing final-response seam."""
    return inspect_continue_candidate(text)["status"] == READY
