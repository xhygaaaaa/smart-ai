# 评测集说明（100 题）

## 字段

| 字段 | 说明 |
|------|------|
| `question` | 用户问题 |
| `expected_answer` | 标准答案，保留设备手册原词，供 bigram 召回判分 |
| `category` | `connectivity` / `functionality` / `specification` / `troubleshooting` / `out_of_scope` |
| `difficulty` | `easy` / `medium` / `hard`（仅用于切片分析，不参与判分） |

## 分布

| 类别 | 题数 |
|------|------|
| connectivity | 12 |
| functionality | 30 |
| specification | 18 |
| troubleshooting | 30 |
| out_of_scope | 10 |
| **总计** | **100** |

难度：easy 18 / medium 52 / hard 30。

## 为什么不用公开数据集

查过现有中文 RAG/问答评测集，主要三类的实际情况：

| 数据集 | 规模/形态 | 为什么不能用在这里 |
|--------|-----------|-------------------|
| [CRUD-RAG](https://github.com/IAAR-Shanghai/CRUD_RAG) | 8 万篇新闻文档 + 四类 CRUD 生成任务 | 新闻续写/摘要/问答，没有「设备手册 + 拒答」的形态；建索引要 3 小时 |
| [bank-rag-benchmark](https://github.com/sjtu-suzhou-ai-zhiwei-tech/bank-rag-benchmark) | 100 份银行 PDF + 450 条测试案例 | 金融领域，答案口径与本项目 5 份手册无交集 |
| CMRC / DuReader 一类阅读理解集 | 篇章级抽取式 QA | 标准答案是原文片段，与本项目「多句操作步骤」的答案形态不一致 |

共同问题：这些集的「标准答案」都锚定在它们自己的语料上。直接混进来，标准答案就脱离
本地知识库，评测变成测模型世界知识，而不是测这套 RAG 的检索+生成+拒答链路。

所以这里的做法是：**从 `data/*_manual.txt` 反向出题**——每道题的答案都能在
手册里找到原文依据。超纲题的答案统一写成明确拒答（含 `config.REFUSAL_SIGNALS`
里的信号词），这样「拒答」也能被判对。

> 方法论上借鉴了 CRUD-RAG 的思路：先定文档，再从文档派生问题，并单独统计
> 「应该拒答」的样本，避免把所有题都当成有答案。

## 扩题原则

1. 一题只问一个点，避免多跳问题混淆归因；
2. 标准答案用手册原词，不写同义改写（否则 bigram 召回会虚低）；
3. 覆盖 5 类设备 × 5 类问题，规格题优先问可验证的数字；
4. 超纲题只问手册明确没有的信息（价格、保修、购买、兼容型号、会员费）；
5. 新增题后跑 `python -m evaluation.bg_runner --judge keyword --retriever bm25 --sleep 0.3`
   （或页面「一键评测」），确认 `errors=0` 且超纲拒答率不下降。

## 判分口径

- `groundedness`：回答关键词被检索文档覆盖的比例（防幻觉）；
- `correctness_recall`：回答对标准答案的字符 bigram 召回，`>= 0.6` 记为答对；
- `task_success_rate`：（有效题答对 + 超纲题正确拒答）/ 总题数；
- 拒答判定（`Evaluator.is_refusal`）：短答案见到拒答话术即算；长答案要看**开场第一句**——
  一上来就声明“不在指南范围内/无法作答”的算拒答，正文写完之后才补一句
  “如需具体步骤此处无法作答”的不算（否则会把正常作答误判成超纲）；
- 需要更严格的语义判分：`python -m evaluation.bg_runner --judge llm`。

## 最近一次结果

`evaluation/results/run_100_v3/summary.json`（deepseek-chat, temperature=0.3,
BM25, TOP_K=5, 100 题）：

```json
{
  "total": 100,
  "valid": 90,
  "out_of_scope": 10,
  "errors": 0,
  "groundedness_avg": 0.6074,
  "correctness_recall_avg": 0.7113,
  "correct_rate": 0.7556,
  "refusal_rate": 1.0,
  "task_success_rate": 0.78
}
```

> temperature=0.3 下模型输出有波动，逐题答对情况会在 ±2 题内摆动，
> 已跑过的对照：`run_100`（73.3% / 76.0%）、`run_100_v2`（73.3% / 76.0%）。
> 引用时按区间描述，不要当成精确值。
