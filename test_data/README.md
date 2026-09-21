# test_data

评测与复现所需的固定输入。**仓库内只保留问题集与题库，不含论文 PDF**（第三方版权作品，不随仓库分发）。

| 文件 | 用途 |
|---|---|
| `evaluation_questions.jsonl` | 端到端评测（L2–L4）问题集：问题、期望文档/页、参考答案、可否回答 |
| `evaluation_questions_refusal.jsonl` | 拒答 / 否证类问题集（考「不该答的不答」） |
| `attention_is_all_you_need_test_questions.md` | 单篇论文的固定问题，用于核对解析与切分质量 |

## 需要自备的样例 PDF

评测基线用的三篇论文**未随仓库分发**。请自行从 arXiv 下载后放入 `test_data/pdfs/`，
**文件名需与下表一致**（代码与基线按此命名）：

| 文件名 | 论文 | arXiv |
|---|---|---|
| `attention-is-all-you-need.pdf` | Attention Is All You Need | [1706.03762](https://arxiv.org/abs/1706.03762) |
| `sentence-bert.pdf` | Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks | [1908.10084](https://arxiv.org/abs/1908.10084) |
| `retrieval-augmented-generation.pdf` | Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks | [2005.11401](https://arxiv.org/abs/2005.11401) |

下载示例（把 `<id>` 换成上表编号）：

```bash
curl -L -o test_data/pdfs/attention-is-all-you-need.pdf https://arxiv.org/pdf/1706.03762
curl -L -o test_data/pdfs/sentence-bert.pdf               https://arxiv.org/pdf/1908.10084
curl -L -o test_data/pdfs/retrieval-augmented-generation.pdf https://arxiv.org/pdf/2005.11401
```

放好后按主 README 的「快速开始」把三篇上传入库（建一个资料集），即可复现 L1 检索基线与 L2–L4 端到端基线。

## 相关

- 评测口径与运行方式：`eval/README.md`
- 检索探针：`python -m src.evaluation.retrieval_probe`
