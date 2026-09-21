"""Deterministic Chinese-to-English term table for the indexed corpus.

A lexical (BM25) channel can only match English text if the Chinese question is
expressed with English tokens, and an LLM rewrite occasionally paraphrases a
*name* into something the document never says ("多头注意力" -> "multi attention
heads" instead of "Multi-Head Attention").  This table is the deterministic
floor: it never hallucinates, it never costs a request, and it is small enough
to read.  Extend it as new documents introduce new vocabulary.
"""

from __future__ import annotations


# Chinese term -> English term as printed in the source documents.
TERM_TABLE: dict[str, str] = {
    "注意力机制": "Attention",
    "注意力": "Attention",
    "自注意力机制": "Self-Attention",
    "自注意力": "Self-Attention",
    "多头注意力": "Multi-Head Attention",
    "缩放点积注意力": "Scaled Dot-Product Attention",
    "点积注意力": "Scaled Dot-Product Attention",
    "编码器": "Encoder",
    "解码器": "Decoder",
    "编解码器": "Encoder-Decoder",
    "位置编码": "Positional Encoding",
    "位置前馈网络": "Position-wise Feed-Forward Network",
    "前馈网络": "Feed-Forward Network",
    "前馈": "Feed-Forward",
    "残差连接": "Residual Connection",
    "层归一化": "Layer Normalization",
    "归一化": "Normalization",
    "子层": "sub-layer",
    "层数": "number of layers",
    "堆叠": "stack",
    "训练数据": "Training Data",
    "训练集": "training set",
    "测试集": "test set",
    "验证集": "development set",
    "训练时间": "Training Time",
    "训练步数": "training steps",
    "批量": "Batching",
    "批次": "batch",
    "硬件": "Hardware GPU",
    "显卡": "GPU",
    "学习率": "learning rate",
    "优化器": "optimizer",
    "超参数": "hyperparameters",
    "词表": "vocabulary",
    "词汇表": "vocabulary",
    "检查点": "checkpoint",
    "标签平滑": "label smoothing",
    "丢弃": "dropout",
    "束搜索": "beam search",
    "解码": "decoding",
    "推理": "inference",
    "微调": "fine-tuning",
    "参数量": "parameters",
    "模型规模": "model size",
    "基座模型": "base model",
    "基础模型": "base model",
    "大模型": "big model",
    "BLEU": "BLEU",
    "准确率": "accuracy",
    "精确匹配": "Exact Match",
    "分数": "score",
    "平均值": "average",
    "基准": "benchmark",
    "数据集": "dataset",
    "任务": "task",
    "检索增强生成": "Retrieval-Augmented Generation",
    "检索器": "retriever",
    "检索": "retrieval",
    "段落": "passage",
    "生成": "generation",
    "无条件": "unconditional",
    "条件": "conditional",
    "交叉验证": "cross-validation",
    "十折交叉验证": "10-fold cross-validation",
    "三元组": "triplet",
    "语义相似度": "semantic textual similarity",
    "语义文本相似度": "semantic textual similarity",
    "句子嵌入": "sentence embedding",
    "自然语言推理": "natural language inference",
    "迁移学习": "transfer learning",
    "零样本": "zero-shot",
    "表格": "table",
    "图": "figure",
    "图表": "chart",
    "公式": "equation",
    "引用": "citation",
    "章节": "section",
    "页": "page",
    "编码": "encoding",
}

# Terms whose Chinese form is a substring of a longer term must not win the
# match; longest-first replacement keeps "多头注意力" from becoming "多HeadAttention".
_ORDERED_TERMS = sorted(TERM_TABLE.items(), key=lambda item: len(item[0]), reverse=True)


STOPWORDS = frozenset({
    "the", "a", "an", "of", "and", "or", "to", "in", "on", "for", "with", "is",
    "are", "was", "were", "be", "been", "it", "its", "this", "that", "these",
    "those", "as", "at", "by", "from", "into", "than", "then", "what", "which",
    "how", "why", "when", "where", "who", "does", "do", "did", "using", "used",
    "use", "number", "many", "much", "分别", "什么", "哪些", "多少", "如何",
    "为什么", "以及", "论文", "中", "的", "和", "与",
})


def matched_terms(text: str, *, min_term_chars: int = 1) -> list[str]:
    """English terms whose Chinese form appears in ``text``, longest match first."""

    found: list[str] = []
    for chinese, english in _ORDERED_TERMS:
        if len(chinese) < min_term_chars or chinese in STOPWORDS:
            continue
        if chinese in text and english not in found:
            found.append(english)
    return found


def term_pairs_for(text: str) -> list[tuple[str, str]]:
    """Chinese/English pairs whose Chinese form appears in ``text``."""

    return [(chinese, english) for chinese, english in _ORDERED_TERMS if chinese in text]
