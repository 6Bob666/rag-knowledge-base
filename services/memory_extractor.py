"""从用户话语中抽取值得长期记住的信息。"""

import re
from dataclasses import dataclass


@dataclass
class MemoryCandidate:
    content: str
    kind: str
    confidence: float


class RuleBasedMemoryExtractor:
    """规则版提取器：不调用 LLM，稳定、零成本、便于测试。"""

    _PATTERNS = (
        (re.compile(r"我叫\s*([^\s，。,.！!？?；;]{1,20})"), "identity", 0.95, "用户自称：{0}"),
        (re.compile(r"我是(?!不是)\s*([^\s，。,.！!？?；;]{1,20})"), "identity", 0.8, "用户身份：{0}"),
        (
            re.compile(
                r"我的([^\s，。,.！!？?；;]{1,10})是([^\s，。,.！!？?；;]{1,30})"
            ),
            "profile",
            0.85,
            "用户资料：{0}是{1}",
        ),
        (re.compile(r"我(?:在|正在)\s*([^\s，。,.！!？?；;]{1,30})"), "profile", 0.7, "用户当前：{0}"),
        (
            re.compile(
                r"我(?:关注|研究|从事|学习|做)\s*([^\s，。,.！!？?；;]{1,30})"
            ),
            "interest",
            0.85,
            "用户关注：{0}",
        ),
        (re.compile(r"(?:请)?记住[：:\s]*([^。！!？?\n]{1,60})"), "explicit", 0.98, "用户要求记住：{0}"),
        (re.compile(r"以后(?:请)?([^。！!？?\n]{1,60})"), "preference", 0.9, "用户偏好：{0}"),
        (
            re.compile(
                r"我(?:喜欢|偏好|想要|希望|需要)\s*([^\s，。,.！!？?；;]{1,30})"
            ),
            "preference",
            0.85,
            "用户偏好：{0}",
        ),
    )

    def extract(self, question: str) -> list[MemoryCandidate]:
        text = (question or "").strip()
        if not text:
            return []
        candidates: list[MemoryCandidate] = []
        seen: set[str] = set()
        for pattern, kind, confidence, template in self._PATTERNS:
            for match in pattern.finditer(text):
                content = template.format(*match.groups()).strip()
                if len(content) > 120 or content in seen:
                    continue
                seen.add(content)
                candidates.append(
                    MemoryCandidate(
                        content=content,
                        kind=kind,
                        confidence=confidence,
                    )
                )
        return candidates
