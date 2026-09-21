"""Decide whether a chat-mode question must search the knowledge base.

Rule based on purpose: no intent classifier, no model call, fully testable.
"""

from __future__ import annotations

from collections.abc import Sequence

RELATED_KEYWORDS = (
    "这篇", "那篇", "论文", "文章", "资料", "文档", "文献", "第", "节", "章",
    "作者", "公式", "表", "图", "页", "摘要", "引言", "实验", "结果", "方法",
    "指标", "数据集", "定义", "定理", "证明", "模型", "架构",
)

QUESTION_MARKERS = (
    "?", "？", "什么", "如何", "怎么", "为什么", "为何", "多少", "几", "是否", "区别", "比较", "解释", "讲一下", "说说",
)

GREETINGS = ("你好", "您好", "hi", "hello", "谢谢", "感谢", "在吗", "早上好", "晚上好", "再见")


def mentions_document(text: str, filenames: Sequence[str]) -> bool:
    lowered = text.lower()
    return any(name and name.lower() in lowered for name in filenames)


def looks_like_question(text: str) -> bool:
    value = text.strip()
    if len(value) < 4:
        return False
    return any(marker in value.lower() for marker in QUESTION_MARKERS)


def is_greeting(text: str) -> bool:
    value = text.strip().lower()
    return len(value) <= 12 and any(value.startswith(item) for item in GREETINGS)


def is_related(text: str, *, has_collection: bool, filenames: Sequence[str] = ()) -> bool:
    """Chat mode must search when any of the three signals fires."""

    if not has_collection or is_greeting(text):
        return False
    if mentions_document(text, filenames):
        return True
    if any(keyword in text for keyword in RELATED_KEYWORDS):
        return True
    return looks_like_question(text)
