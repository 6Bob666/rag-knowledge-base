"""对 LLM 输出做最小安全校验。"""

import re


class UnsafeOutputError(ValueError):
    """表示模型输出为空或疑似包含不应泄露的敏感信息。"""


# 这里只拦截明显的密钥/密码表达式，避免把普通的“API”讨论误判为泄露。
_SENSITIVE_PATTERNS = (
    re.compile(r"(?:api[_ -]?key|secret[_ -]?key|password|passwd)\s*[:=]\s*\S+", re.I),
    re.compile(r"\bsk-[A-Za-z0-9_-]{10,}\b"),
)


def validate_answer(answer: str) -> str:
    """校验并返回可交付的答案。"""
    if not isinstance(answer, str) or not answer.strip():
        raise UnsafeOutputError("LLM 返回了空答案")

    if any(pattern.search(answer) for pattern in _SENSITIVE_PATTERNS):
        raise UnsafeOutputError("LLM 输出疑似包含敏感信息")

    return answer.strip()
