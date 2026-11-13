# SmartGuide AI · 后端开发深度解读(逐行讲解)

> 这份文档面向后端/算法开发视角。我会**逐个文件、尽量逐行**地讲清楚每段代码"做了什么、为什么这么写、有没有更好的写法、藏着哪些坑"。目标是让你不仅能复述代码,还能在面试里被追问到任何一行都答得上来。
>
> 阅读方式:跟着文件顺序读,每个文件先看「整体职责」,再看「逐行解读」,最后看「这个文件的面试考点」。全部读完后用文末的「综合面试题库」自测。

---

## 目录

- [第零部分:开始之前——技术栈与数据流](#第零部分开始之前)
- [第一部分:`config.py` 配置中心](#第一部分configpy-配置中心)
- [第二部分:`src/simple_rag.py` 检索引擎](#第二部分srcsimple_ragpy-检索引擎)
- [第三部分:`src/llm_client.py` 大模型调用层](#第三部分srcllm_clientpy-大模型调用层)
- [第四部分:`src/evaluator.py` 评测引擎](#第四部分srcevaluatorpy-评测引擎)
- [第五部分:`问答系统.py` 主页](#第五部分问答系统py-主页)
- [第六部分:`pages/实时评测.py`](#第六部分pages实时评测py)
- [附录 A:`src/telemetry.py` 埋点与自助解决率](#附录-atelemetrypy-埋点与自助解决率)
- [附录 B:`src/evaluation_runner.py` 可复现评测引擎](#附录-bevaluation_runnerpy-可复现评测引擎)
- [附录 C:`src/auth.py` 与 `src/admin_panel.py` 服务端](#附录-cauthpy-与-srcadmin_panelpy-服务端)
- [附录 D:文档同步记录](#附录-d文档同步记录)
- [第七部分:`pages/一键评测.py`](#第七部分pages一键评测py)
- [第八部分:综合面试题库](#第八部分综合面试题库)
- [第九部分:知识点总结速查表](#第九部分知识点总结速查表)

---

## 第零部分:开始之前

### 0.1 技术栈一览

| 层 | 技术 | 在项目里干什么 |
|----|------|---------------|
| 前端/UI | Streamlit | 用 Python 写网页,不用碰 HTML/JS |
| 检索 | BM25（默认）/ bigram / 混合检索 | 从文档里找相关段落 |
| 大模型 | OpenAI SDK（连 DeepSeek） | 生成回答 |
| 配置 | python-dotenv | 从 `.env` 读密钥等 |
| 数据 | pandas + SQLite | 读测试集 CSV、存埋点事件 |
| 可视化 | plotly | 画柱状图/折线图/直方图 |

### 0.2 一次完整请求的数据流(必须背下来)

```
用户在网页输入问题 "智能手环防水等级?"
        │
        ▼
[问答系统.py] 接收输入
        │
        ▼
[evaluator.is_valid_question()] 第一级:有效性检查
        │ (有效)
        ▼
[retriever.retrieve()] 设备定向 + 同义词扩展后，从 159 段里检索 top_k=5 段
        │  返回 [(设备名, 段落, 得分), ...]
        ▼
[llm_client.answer_question()] 把文档+问题拼成Prompt → 调大模型
        │  内部走 _call_api_with_retry() 处理429重试
        │  返回 AI 回答文本
        ▼
[evaluator.classify_question()] 第二级:是否拒答→分类
        │ (有效)
        ▼
[evaluator.calculate_accuracy_a()] 算有据可查率（groundedness）
        │
        ▼
[telemetry.log_query()] 写 SQLite 埋点（耗时/分类/有据可查率/是否追问）
        │
        ▼
存入 st.session_state.current_result + history
        │
        ▼
页面渲染:答案 + 分类 + 有据可查率 + 引用文档 + 👍/👎 反馈
```

**记忆口诀:校验 → 检索 → 生成 → 分类 → 评分 → 埋点 → 存储 → 渲染。**

### 0.3 模块依赖关系

```
config.py  ◄────┬────────┬────────┬────────┬────────┐
   ▲             │        │        │        │        │
   │         retrieval  llm_client evaluator telemetry auth
   │             ▲        ▲        ▲        ▲        ▲
   │             └────────┴────┬───┴────────┴────────┘
   └───────────────────────────┤
                          问答系统.py / pages/*.py / evaluation_runner.py
```

- `config.py` 是最底层,谁都依赖它,它不依赖任何业务模块。
- `src/` 下的模块互相独立(低耦合),只依赖 config。`retrieval.py` 会引用
  `simple_rag.py` 的同义词表,这是唯一的模块间依赖。
- 页面和 `evaluation_runner.py` 是"编排层",把 src 模块组合起来用。

> **架构考点:这是典型的分层架构(Layered Architecture)。** 配置层 → 能力层(src) → 编排层(页面)。好处是 src 里的每个模块都能单独测试、单独替换。比如想换检索算法,只动 `retrieval.py`,上层完全不用改——因为大家都依赖"接口"(`retrieve(query, top_k)` 的签名),而不是实现。这也是为什么 bigram / bm25 / hybrid 三种实现可以随时切换。

---

## 第一部分:`config.py` 配置中心

### 1.1 整体职责

集中读取环境变量、定义所有常量。它是"单一事实来源"(Single Source of Truth)——任何配置项在全项目只在这里定义一次。

### 1.2 逐行解读

```python
"""项目配置文件"""
import os
from dotenv import load_dotenv
```
- `import os`:用来读环境变量(`os.getenv`)和拼路径(`os.path.join`)。
- `from dotenv import load_dotenv`:`python-dotenv` 库,作用是把 `.env` 文件里的 `KEY=VALUE` 加载进系统环境变量。

```python
# 加载 .env 文件
load_dotenv()
```
- 执行后,`.env` 文件里写的 `LLM_API_KEY=xxx` 就能用 `os.getenv("LLM_API_KEY")` 读到了。
- **坑点**:`load_dotenv()` 默认找当前工作目录的 `.env`。如果从别的目录启动可能找不到。生产环境更稳妥的写法是 `load_dotenv(dotenv_path=...)` 指定绝对路径。
- **为什么用 `.env` 而不是直接写代码里**:密钥写进代码会被提交到 git 泄露。`.env` 被 `.gitignore` 排除,只存在本地。这是**密钥管理**的基本规范。

```python
# LLM配置
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
```
- `os.getenv(key, default)`:读不到环境变量时返回第二个参数(默认值)。这是**防御性编程**——即使没配 `.env`,程序也有合理默认值,不会直接崩。
- `float(...)`:环境变量读出来**永远是字符串**,温度要参与数值计算,必须转成 float。这是个高频面试坑点:"环境变量是什么类型?"答:字符串,用数字要手动转。
- **`temperature=0.3` 的含义**:控制大模型输出的随机性。0 最确定(每次几乎一样),1+ 更发散。客服场景要稳定、可复现,所以用低温 0.3。**这是产品语义,不是随便填的。**

```python
# RAG配置
TOP_K = int(os.getenv("TOP_K", "5"))
```
- 同理 `int(...)` 把字符串转整数。`TOP_K=5` 表示检索时返回最相关的 5 段。

```python
# 知识库路径
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
```
- `__file__`:当前文件(config.py)的路径。
- `os.path.dirname(__file__)`:取它所在的目录(项目根目录)。
- `os.path.join(..., "data")`:拼出 `data` 目录的路径。
- **为什么不直接写 `"data/"`**:相对路径依赖"从哪启动程序"。用 `__file__` 算出的是**相对于代码文件的绝对路径**,无论从哪启动都对。这是**跨平台、跨启动目录健壮性**的关键技巧,Windows/Linux 路径分隔符也由 `os.path.join` 自动处理(不会手写 `/` 或 `\`)。

```python
MANUALS = {
    "智能手环": os.path.join(DATA_DIR, "smart_band_manual.txt"),
    "智能手表": os.path.join(DATA_DIR, "smart_watch_manual.txt"),
    "智能音箱": os.path.join(DATA_DIR, "smart_speaker_manual.txt"),
    "智能台灯": os.path.join(DATA_DIR, "smart_lamp_manual.txt"),
    "智能门锁": os.path.join(DATA_DIR, "smart_lock_manual.txt"),
}
```
- 用字典管理"设备名→文件路径"的映射。
- **为什么用字典**:① 加设备只加一行,可扩展;② 检索结果里能带上设备名(字典的 key),方便展示"这段来自哪个设备的说明书"。

```python
# 评测配置
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "evaluation", "test_set.csv")
```
- 同样的绝对路径技巧,指向 100 题测试集。

```python
# 拒答信号词(用于识别超纲问题)
REFUSAL_SIGNALS = [
    "不在", "无法", "不能", "没有收录", "不包含", "超出",
    "不提供", "无法作答", "不在手册", "不在指南", "建议咨询"
]
```
- 这是"超纲检测"的关键词表。`evaluator` 会检查 AI 回答里有没有这些词,有就判定为"拒答→超纲"。
- **为什么放配置里**:运营发现 AI 又用了新的拒答说法(比如"暂未提供"),加一个词就行,不用改逻辑代码。

### 1.3 这个文件的面试考点

- **Q:为什么用 `.env` 管理密钥?** → 防止密钥提交到 git 泄露;配置与代码分离;不同环境(开发/生产)用不同 `.env`。
- **Q:`os.getenv` 读出来是什么类型?** → 永远是字符串,数值要手动 `int()`/`float()`。
- **Q:为什么用 `os.path.join(os.path.dirname(__file__), ...)` 而不是相对路径?** → 不依赖启动目录,跨平台自动处理分隔符。
- **Q:这个配置文件有什么可改进的?** → ① 可以用 `pydantic-settings` 做类型校验和必填检查(比如 API_KEY 为空时启动就报错,而不是调用时才失败);② 路径用 `pathlib.Path` 比 `os.path` 更现代;③ 可以区分 dev/prod 多套配置。

---

## 第二部分:`src/retrieval.py` 检索引擎

> ⚠️ 这一部分已按当前代码重写。旧版本用的是 `simple_rag.py` 的 bigram + Jaccard，
> 现在 `simple_rag.py` 只作为降级兜底保留，默认走 `retrieval.py` 的 BM25。

### 2.0 现在的检索层结构

```
build_retriever(documents)          # 工厂函数，按 config.RETRIEVER 选实现
   ├── "bigram"  → SimpleRAG        # 最早的 bigram 版，零依赖，兜底
   ├── "bm25"    → BM25RAG          # ← 默认
   └── "hybrid"  → HybridRAG        # BM25 + 向量(RRF) [+ 可选 rerank]
                     ├── 向量依赖装不上 → 自动降级为 BM25
                     └── RERANK=true → 再加 bge-reranker 重排
```

三个检索器对外接口完全一致：

```python
retriever.retrieve(query, top_k) -> [(设备名, 段落, 分数), ...]
```

**接口统一是这里最值得讲的设计**：上层（主页、评测、一键评测页）完全不关心底层是哪一
个，只要换配置就能对比三种检索方案的效果。评测里 `--retriever` 参数就是靠这个做的。

### 2.1 BM25RAG（默认）

#### 算法原理

BM25 是关键词检索的工业标准（Elasticsearch 默认打分），解决的是"怎么给一段文字和一句
查询算相关性"。

```
idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
score  = Σ idf(t) · [tf·(k1+1)] / [tf + k1·(1 - b + b·dl/avgdl)]
参数：k1 = 1.5，b = 0.75
```

三个部分各自解决一个问题：

| 部分 | 作用 | 不这么做会怎样 |
|---|---|---|
| `idf(t)` | 越稀有的词权重越高 | "的""是"这类词会主导打分 |
| `k1 = 1.5` | 词频**饱和**：3 次比 1 次明显加分，40 次比 20 次几乎不涨 | 堆砌关键词就能刷高分 |
| `b = 0.75` | 长段落打折 | 长段落靠词多霸榜 |

#### 相比教科书 BM25，这里加了两处领域适配

**① 标题权重 2 倍**（`retrieval.py` 的 `_chunk_tokens`）：

```python
return tokenize(body) + tokenize(heading) * 2
```

标题的词在词表里**出现两遍**（列表拼接，`* 2` 就是两份，不是三份）。效果：问"续航多久"，
优先落在标题是"电池续航"的段落，而不是正文里顺带提了一句续航的段落。
**这是标准 BM25 没有的，属于为手册类文档做的改造。**
<!-- 已按代码核对 2026-10-27 -->

**② 分数归一化**（`retrieve` 末尾）：

```python
best = top[0][2] or 1.0
top = [(d, c, s / best) for d, c, s in top]
```

第一名恒为 1.0，其余按比例缩放。好处有二：前端能显示"相关度 100%"；不同查询之间可比。

#### 分词（`tokenize`）

不引入 jieba，保持零依赖：

- 中文连续段：切成**所有相邻二字组**（"防水等级" → 防水/水等/等级）
- 英文和数字：**整段保留**（"5atm"、"100" 不被拆开）

**为什么英文数字整段保留**：型号、规格必须精确匹配，"5ATM" 拆成 "5a"/"at"/"tm"
就废了。这是中文 bigram + 英文整词的混合策略。

#### 设备定向与同义词扩展（`detect_device` / `expand_query`）

1. 问题里命中设备词（手环/手表/音箱/台灯/门锁）→ 只在该设备的段落里打分
2. 命中之后**把设备词从查询里删掉**再算分——否则"手环"会让"产品概述"这种泛泛的段落
   拿高分
3. 口语词扩展成手册术语（"充不进电" → 无法充电/充电故障/充电触点/充电器）

> **为什么第 2 步必须做**：这是实测踩出来的坑。设备词在几乎所有段落里都出现，idf 会
> 把它压得很低，但它仍会贡献一点分数，足够把泛泛的段落顶上去。

### 2.2 DenseRAG（向量）

```python
model = SentenceTransformer(config.EMBEDDING_MODEL)   # 默认 BAAI/bge-small-zh-v1.5
embeddings = model.encode(chunks, normalize_embeddings=True)
similarity = q · c                                    # 归一化后点积 = 余弦相似度
```

- **懒加载**：`SentenceTransformer` 在 `__init__` 里才导入，没装依赖时抛异常
- **自动降级**：`_try_dense()` 捕获所有异常返回 None，工厂退回 BM25，主流程不受影响

**面试点**：向量强在"换个说法也能匹配"（充不进电 ≈ 无法充电），弱在数字型号容易糊
（"支持 5ATM" 和 "支持 10ATM" 在语义空间里很近）。所以它和 BM25 是**互补**关系。

### 2.3 HybridRAG（BM25 + 向量，RRF 融合）

#### 为什么不能直接把两路分数相加

BM25 的分数是归一化后的相对值，向量是余弦相似度，**量纲和分布都不同**。直接相加等于
拿米加公斤。

#### RRF：只看名次，不看分数

```
score(chunk) = Σ  weight / (60 + rank)

BM25 名次第 1 → (1-α)/61
向量 名次第 1 → α/61
```

- `α = HYBRID_ALPHA = 0.5`：BM25 权重 `1-α`，向量权重 `α`。α=0 纯 BM25，α=1 纯向量
- `60`：平滑常数，来自原论文。作用是压平名次之间的差距——第 1 名 1/61、第 2 名 1/62，
  差别很小，结果稳定，不会因为某一路的第一名特别高就把另一路全压死

#### 候选池先放大再收缩

```python
pool = max(top_k * 4, 20)          # top_k=5 时各召回 20 条
fused = sorted(rrf.items())[:pool] # 融合后再取 5
```

融合的价值就在于"一路漏掉的能被另一路补回来"，所以必须先把候选池放大。

#### 可选重排

```python
if self.reranker is not None:
    pairs = [(query, chunk) for chunk in fused]
    scores = self.reranker.predict(pairs)     # BAAI/bge-reranker-base
```

召回是"问题、文档分开编码"（双塔，快但粗）；重排是"问题+文档拼在一起送进模型"
（交叉编码器，准但慢）。所以重排只作用在 20 条候选上，不作用在全量。

### 2.4 SimpleRAG（bigram 兜底）

`src/simple_rag.py` 是最早的实现，现在只在 `RETRIEVER=bigram` 或依赖缺失时使用。

当前打分公式（已不是 Jaccard）：

```
覆盖率   = |查询∩段落| / |查询|          × 0.55
精确率   = |查询∩段落| / |段落|          × 0.20
标题命中 = |查询∩标题| / |查询|          × 0.25
```

分母**以查询为基准**是有原因的：早期版本用 `min(查询词数, 段落词数)` 当分母，等于
**段落越短分越高**——"产品概述"这种两三行的段落稳拿第一，把真正的"电池续航"挤下去。
改成覆盖率为主之后就正常了。

### 2.5 这一部分的面试考点

- **Q:BM25 的 k1 和 b 各控制什么?** → k1 控制词频饱和（防堆词），b 控制长度归一化强度
  （防长文霸榜）。k1 越大，词频影响越久才饱和；b=0 完全不归一化，b=1 完全归一化。
- **Q:为什么不直接用向量检索?** → ① 离线私有化环境装不了模型；② 型号/数字要精确匹配；
  ③ 当前 159 段规模 BM25 够用。向量和混合都实现了，配置一改就能开。
- **Q:RRF 为什么用名次不用分数?** → 两路分数量纲不同，直接加权求和无意义；名次是天然
  归一化的，且对异常高分鲁棒。
- **Q:RRF 的 60 是怎么来的?** → 原论文的经验值，作用是压平名次差距。它是平滑项，不敏感，
  主要影响"头部是否过于集中"。
- **Q:为什么标题要加权?** → 手册类文档里，标题就是这一节的命题。正文可能只是顺带提到
  某个词，标题命中才说明这一节真在讲它。
- **Q:混合检索一定更好吗?** → 不一定。两路都弱的时候融合也救不回来；而且要额外下模型、
  增加延迟。必须用评测集对比，不能凭信仰。
- **Q:依赖缺失时怎么办?** → 工厂里 `_try_dense()` 捕获异常返回 None，自动降级 BM25，
  主流程不报错。**降级路径是必须设计的，客户现场什么情况都有。**

## 第三部分:`src/llm_client.py` 大模型调用层

### 3.1 整体职责

封装与大模型的所有交互:① 用检索到的文档拼 Prompt;② 调用 OpenAI 兼容 API;③ 处理 429 限流的指数退避重试。

### 3.2 逐行解读

```python
"""LLM大模型调用模块 - 支持OpenAI兼容API"""
import time
from typing import List, Tuple, Optional
from openai import OpenAI
import config
```
- `time`:用于重试时 `sleep` 等待。
- `from openai import OpenAI`:OpenAI 官方 SDK 的客户端类。**注意**:虽然叫 OpenAI,但只要服务商提供"OpenAI 兼容"接口(DeepSeek/通义/MiniMax 都提供),就能用这个 SDK 连过去——这就是"OpenAI 兼容格式"成为事实标准的好处。
- `Optional` 这里其实没用到(import 了没使用),属于可清理的冗余 import。面试官眼尖可能会问。

```python
class LLMClient:
    def __init__(self):
        self.client = OpenAI(
            base_url=config.LLM_BASE_URL,
            api_key=config.LLM_API_KEY
        )
        self.model = config.LLM_MODEL
        self.temperature = config.LLM_TEMPERATURE
```
- 构造时创建一个 `OpenAI` 客户端实例。`base_url` 指向 DeepSeek 而非 OpenAI,这就是"换服务商只改配置"的实现。
- 把 model 和 temperature 存成实例属性,后面调用时复用。

```python
    def _call_api_with_retry(self, messages: list, max_retries: int = 3) -> str:
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature
                )
                return response.choices[0].message.content
```
- 核心 API 调用:`chat.completions.create` 是 OpenAI 的标准对话接口。
- `messages`:对话消息列表,格式 `[{"role": "system/user/assistant", "content": "..."}]`。
- `response.choices[0].message.content`:取第一个候选回复的文本内容。`choices` 是列表是因为 API 支持一次返回多个候选(由 `n` 参数控制,默认 1)。
- **成功就 `return`,直接退出循环**——重试循环只在失败时才继续。

```python
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "rate" in error_str.lower():
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 3  # 3秒、6秒、9秒指数退避
                        print(f"遇到限流，等待{wait_time}秒后重试...")
                        time.sleep(wait_time)
                        continue
                raise e

        raise Exception("API调用失败，已达最大重试次数")
```
- `except Exception as e`:捕获所有异常。
- `if "429" in error_str or "rate" in error_str.lower()`:**只对限流错误重试**。其他错误(如密钥错误 401、参数错误 400)重试没意义,直接 `raise e` 抛出。这是关键判断——**不要对所有错误都盲目重试**。
- `if attempt < max_retries - 1`:还没到最后一次才等待重试;最后一次失败就不等了,直接抛。
- `wait_time = (attempt + 1) * 3`:attempt 从 0 开始,所以是 3、6、9 秒。
  - **严格来说这是"线性退避"不是"指数退避"**(指数应该是 3、9、27,即 `3 * 2**attempt`)。注释写"指数退避"略不准确。面试官若较真可以坦诚:"这里实现的是线性递增退避,叫指数退避不够严谨。"——**能指出自己代码的不精确,是加分项**。
- `time.sleep(wait_time)`:阻塞等待。
- `continue`:进入下一次循环重试。
- 循环正常结束(理论上不会走到)再兜底抛异常。

> **退避策略小知识**:
> - **固定退避**:每次等一样久(如都等 3 秒)。
> - **线性退避**:等待时间线性增长(3、6、9)——本项目实际用的。
> - **指数退避**:成倍增长(3、6、12 或 3、9、27)。
> - **指数退避 + 抖动(jitter)**:在指数基础上加随机,**避免大量客户端同时重试造成"重试风暴"**——这是生产环境最佳实践。
> 面试讲法:"我用的是线性退避,生产环境更推荐指数退避加随机抖动,防止惊群效应。"

```python
    def answer_question(self, question: str,
                        retrieved_docs: List[Tuple[str, str, float]]) -> str:
        context_parts = []
        for i, (device, content, score) in enumerate(retrieved_docs, 1):
            context_parts.append(f"[文档{i} - {device}]\n{content}")
        context = "\n\n".join(context_parts)
```
- `enumerate(retrieved_docs, 1)`:遍历检索结果,序号从 1 开始(第二个参数是起始值)。
- 元组解包 `(device, content, score)`:直接拆出设备名、内容、得分。
- 把每段格式化成 `[文档1 - 智能手环]\n内容...`,用 `\n\n` 拼接成完整上下文。
- **为什么标注文档来源**:让模型知道每段出自哪个设备,也为"可溯源"打基础。

```python
        system_prompt = """你是SmartGuide AI客服助手，专门回答智能硬件使用问题。

回答规则：
1. 只根据提供的文档内容回答，不能编造信息
2. 如果文档中没有相关信息，明确告知"该信息不在设备使用指南中，无法作答"
3. 回答要简洁准确，突出关键步骤和数字
4. 对于超出使用指南的问题(如价格、保修、购买渠道、维修、对比建议)，礼貌拒答
"""
```
- **System Prompt 是整个 AI 行为的"宪法"**。逐条看:
  - 规则1:防幻觉的核心约束——"只根据文档"。
  - 规则2:**给拒答提供了固定话术**,这句话里的"不在""无法作答"正好命中 `REFUSAL_SIGNALS`,让评测端能检测到。**Prompt 和评测逻辑是配合设计的**——这是个精妙的耦合点。
  - 规则3:体验约束(简洁、突出数字)。
  - 规则4:明确划定能力边界(超纲拒答)。

```python
        user_prompt = f"""参考文档：
{context}

用户问题：{question}

请根据参考文档回答用户问题。"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        return self._call_api_with_retry(messages)
```
- `user_prompt` 把检索到的上下文和用户问题组装起来。
- `messages` 数组:system 定规则,user 给具体任务。这是标准的两段式结构。
- 最后调用带重试的方法。
- **设计考点:为什么文档放 user 而不是 system?** 一种常见做法是 system 放"角色和规则"(相对固定),user 放"本次的检索内容+问题"(每次变化)。也有人把文档塞 system。没有绝对对错,但**把动态内容放 user、静态规则放 system** 有利于将来的 prompt caching(部分服务商对固定的 system 部分有缓存优惠)。

### 3.3 这个文件的面试考点

- **Q:为什么只对 429 重试,不对所有错误重试?** → 401(密钥错)、400(参数错)重试也不会成功,只会浪费时间。只有限流、网络抖动这类**临时性错误**才值得重试。
- **Q:你这个退避是指数退避吗?** → (诚实)实际是线性退避(3/6/9),注释写得不准。生产建议指数退避+抖动防重试风暴。
- **Q:`response.choices[0]` 为什么是数组?** → API 支持一次生成多个候选(n 参数),choices 是候选列表,默认取第一个。
- **Q:system prompt 和 user prompt 怎么分工?** → system 定角色和规则(相对固定),user 放本次的上下文和问题(动态)。
- **Q:这段代码有什么风险?** → ① 没有超时设置(API 卡住会一直挂),应加 `timeout`;② 没有 token 长度控制,文档太长可能超模型上下文窗口;③ `except Exception` 太宽泛,理想情况应捕获 SDK 的具体异常类型(如 `RateLimitError`)。

---

## 第四部分:`src/evaluator.py` 评测引擎

### 4.1 整体职责

给 AI 回答打分和分类。三个能力:① 有效性判断(`is_valid_question`);② 拒答判断(`is_refusal`);③ 有据可查率(`calculate_accuracy_a`,原名方案A),外加关键词提取(`extract_keywords_v2`)和总分类(`classify_question`)。

### 4.2 逐行解读

```python
class Evaluator:
    def __init__(self):
        self.refusal_signals = config.REFUSAL_SIGNALS
```
- 从配置读拒答词表,存成实例属性。

#### (1) 有效性判断 `is_valid_question`

```python
    def is_valid_question(self, question: str) -> bool:
        if len(question.strip()) < 3:
            return False

        device_keywords = ["手环", "手表", "音箱", "台灯", "门锁", "智能"]
        has_device = any(kw in question for kw in device_keywords)

        action_keywords = ["如何", "怎么", "怎样", "什么", "多少", "哪里",
                          "能否", "可以", "支持", "是否", "为什么"]
        has_action = any(kw in question for kw in action_keywords)

        return has_device or has_action
```
- 先长度过滤:小于 3 字符直接判无效("hi"、"在吗")。
- `any(kw in question for kw in keywords)`:**只要有一个关键词出现就 True**。`any()` 遇到第一个 True 就短路返回,高效。
- 逻辑:**包含设备词 或 包含疑问词** 就算有效。
- **设计取舍**:这是规则法(rule-based),简单可控、零成本、可解释。缺点是**召回有限**——比如"防水吗"不含设备词也不含上面的疑问词(没有"吗"),会被误判无效。更强的做法是用小模型做意图分类,但成本和复杂度高。又是"够用就好"。
- **坑点**:用 `or` 比较宽松。"今天天气怎么样"含"怎么"会被判有效(虽然和设备无关),要靠后面的检索+AI拒答兜底。这是**多级过滤**的设计——单级不完美没关系,后面还有关卡。

#### (2) 拒答判断 `is_refusal`

```python
    def is_refusal(self, answer: str) -> bool:
        return any(signal in answer for signal in self.refusal_signals)
```
- 检查 AI 回答里是否含任一拒答信号词。
- **依赖前提**:Prompt 里规定了拒答时要说"不在指南中""无法作答",这些词正好在信号表里。**Prompt 和这里是强耦合的**——改了 Prompt 的拒答话术,信号表也要同步改,否则检测失效。这是个隐藏的维护风险点,面试可以主动提。
- **局限**:误判风险。如果正常回答里出现"无法"(比如"无法连接时请重启"),会被误判成拒答。这是关键词法的固有缺陷。

#### (3) 关键词提取 `extract_keywords_v2`(本文件最核心)

```python
    def extract_keywords_v2(self, text: str) -> set:
        keywords = set()

        # 1. 提取"数字+单位"组合
        number_patterns = [
            r'\d+[天日小时分钟秒]',  # 时间: 7天、2小时
            r'\d+ATM',  # 防水等级: 5ATM
            r'\d+米',  # 距离: 5米
            r'\d+[KkMm]',  # 色温/存储: 2700K、100M
            r'\d+%',  # 百分比: 50%
            r'\d+-\d+[KkMm位]',  # 范围: 2700-6500K、6-12位
            r'\d+枚',  # 指纹数: 100枚
            r'\d+\.?\d*秒',  # 小数秒: 0.3秒
        ]
        for pattern in number_patterns:
            matches = re.findall(pattern, text)
            keywords.update(matches)
```
- `\d+` 匹配一个或多个数字。后面接单位字符。
- `re.findall(pattern, text)`:找出所有匹配,返回列表。
- `keywords.update(matches)`:把列表里所有元素加进集合(批量 add)。
- **为什么先抓"数字+单位"**:这些是说明书里最关键的技术参数(续航14天、防水5ATM)。如果用通用切词,"14天"可能被切成"14"和"天"分别处理,丢失语义。**先把这些"原子化的事实"整体提取出来**,保护它们不被切碎。
- **`\d+\.?\d*秒`**:`\.?` 表示小数点可有可无,`\d*` 表示小数部分零或多位。能匹配"0.3秒"也能匹配"3秒"。

```python
        # 2. 按标点和连接词切分短语
        clean_text = re.sub(r'[，。！？、；：""''（）\(\)《》\[\]]', ' ', text)
        filler_words = ['的', '了', '是', '在', '和', '与', '或', '即可',
                       '可以', '需要', '然后', '接着', '约', '大约']
        for word in filler_words:
            clean_text = clean_text.replace(word, ' ')
```
- 先用正则把各种标点替换成空格(注意中英文括号都列了)。
- 再把虚词/连接词替换成空格。这些词没有区分度("的""了"哪都有),留着会污染关键词。
- **效果**:"续航约14天的设置" → 切分点变多,"约""的"被去掉,剩下有意义的片段。

```python
        # 3. 提取2-5字中文短语
        phrases = re.findall(r'[一-龥]{2,5}', clean_text)
        stopwords = {'然后', '接着', '如果', '这个', '那个', '进行', '开始'}
        meaningful_phrases = [p for p in phrases if p not in stopwords]
        keywords.update(meaningful_phrases)

        return keywords
```
- `[一-龥]{2,5}`:匹配连续 2 到 5 个中文字符。被空格隔开的片段会分别匹配。
- `{2,5}` 是**贪婪匹配**:能匹配 5 个就不匹配 4 个。"睡眠监测功能"(6字)会被切成"睡眠监测功"(5字)——这其实是个小缺陷,但对算分影响不大。
- `stopwords` 集合再过滤一遍残留的无意义词。用 `set` 是因为 `in` 判断 O(1)。
- **这就是文档里说的"关键词提取演进"**:v1 用粗暴 2 字滑窗(会切出"约天""ma"),v2 改成"先抓数字单位 + 去虚词 + 抓有意义短语"三步法。**这是真实迭代踩坑的产物。**

#### (4) 有据可查率（groundedness）`calculate_accuracy_a`

```python
    def calculate_accuracy_a(self, answer: str,
                             retrieved_docs: List[Tuple[str, str, float]]) -> Tuple[float, dict]:
        answer_keywords = self.extract_keywords_v2(answer)

        if len(answer_keywords) == 0:
            return 0.0, {
                "total_keywords": 0,
                "supported_keywords": 0,
                "unsupported_keywords": [],
                "answer_keywords": []
            }
```
- 先提取 AI 回答的关键词。
- **边界处理**:如果一个关键词都没提到(空回答或全是数字标点),返回 0 分。**不处理这个边界会在下面除法时除零崩溃**——这是防御性编程的体现。

```python
        all_doc_content = " ".join([content for _, content, _ in retrieved_docs])
        doc_keywords = self.extract_keywords_v2(all_doc_content)

        supported = answer_keywords & doc_keywords
        unsupported = answer_keywords - doc_keywords

        accuracy = len(supported) / len(answer_keywords)

        return accuracy, {
            "total_keywords": len(answer_keywords),
            "supported_keywords": len(supported),
            "unsupported_keywords": list(unsupported),
            "answer_keywords": list(answer_keywords)
        }
```
- `[content for _, content, _ in retrieved_docs]`:只取元组中间的内容(`_` 是惯例的"忽略"变量),拼成一大段文档文本。
- 提取文档的关键词集合。
- `answer_keywords & doc_keywords`:交集 = **回答里有依据的词**。
- `answer_keywords - doc_keywords`:差集 = **回答里没依据的词(疑似幻觉)**。
- `accuracy = 有依据的词数 / 回答总词数`:这就是有据可查率的定义。
- **为什么用 `&` 和 `-` 集合运算**:简洁高效,O(min(n,m))。这是用对数据结构带来的代码优雅。
- 返回值是 `(准确率, 详情字典)` 元组,详情用于前端展示"哪些词没依据"。**算法不只给结论,还给可解释的中间结果**——好的评测要可追溯。

#### (5) 总分类 `classify_question`

```python
    def classify_question(self, question: str, answer: str) -> Tuple[str, str]:
        if not self.is_valid_question(question):
            return "无效", "问题不符合客服场景(过短或不包含设备关键词)"

        if self.is_refusal(answer):
            return "超纲", "AI拒答(问题超出知识库范围，如价格/保修/购买)"

        return "有效", "正常回答，计入准确率统计"
```
- **三级分类的总入口,顺序很重要**:
  1. 先判无效(不看 answer,因为无效问题压根不该调 AI)。
  2. 再判超纲(看 answer 是否拒答)。
  3. 都不是 → 有效。
- 返回 `(分类, 说明)` 元组,说明文字用于前端展示,提升可解释性。
- **设计考点:为什么是这个顺序?** 无效判断最便宜(纯本地、不依赖 AI 输出),放最前面。这和"调 AI 前先拦截无效问题"的成本优化是一致的——在 `问答系统.py` 里,无效问题根本不会走到调用 AI 这一步。

### 4.3 这个文件的面试考点

- **Q:三级分类为什么是这个顺序?** → 无效判断最便宜且不依赖 AI,放最前;超纲依赖 AI 输出,放后。成本和依赖关系决定顺序。
- **Q:`is_refusal` 有什么风险?** → 关键词误判:正常回答含"无法"(如"无法连接时…")会被误判拒答。且与 Prompt 话术强耦合。
- **Q:为什么关键词提取要先抓"数字+单位"?** → 技术参数是说明书核心事实,整体提取防止"14天"被切碎成"14"和"天"丢失语义。
- **Q:有据可查率为空回答时怎么处理?** → 返回 0,且提前 return 避免除零。
- **Q:有据可查率的局限是什么?** → ① 只看关键词重合,**不理解语义**(同义不同字算没依据);② 依赖关键词提取质量;③ 可能被"复读文档原词"刷高分(AI 照抄文档就能拿满分,但未必真正答对问题)。改进可引入语义相似度(embedding)做支撑判断。

---

## 第五部分:`问答系统.py` 主页

### 5.1 整体职责

Streamlit 主页面,编排层。把 RAG、LLM、Evaluator 三个能力串起来,处理用户输入,渲染结果。也是理解 Streamlit 运行机制的最佳样本。

### 5.2 先理解 Streamlit 的核心机制(必看)

> **Streamlit 最反直觉的一点:每次用户交互(点按钮、输入、切页),整个脚本从第一行到最后一行重新执行一遍。** 它不是传统的"事件回调"模型,而是"全脚本重跑"模型。
>
> 这带来两个后果:
> 1. **普通局部变量每次都被重置**——所以要用 `st.session_state` 存需要跨次保留的数据。
> 2. **重操作(加载文件、建索引)如果不缓存,每次重跑都重做**——所以要用 `@st.cache_resource` / `@st.cache_data`。
>
> 这两点是 Streamlit 所有"奇怪写法"的根源。理解了它,下面代码就全通了。

### 5.3 逐行解读

```python
"""SmartGuide AI - 主页:问答系统"""
import streamlit as st
from src.simple_rag import SimpleRAG
from src.llm_client import LLMClient
from src.evaluator import Evaluator
import config
```
- 导入三个能力模块和配置。

```python
@st.cache_resource
def load_knowledge_base():
    """加载知识库(只加载一次)"""
    documents = {}
    for device_name, file_path in config.MANUALS.items():
        with open(file_path, 'r', encoding='utf-8') as f:
            documents[device_name] = f.read()
    return SimpleRAG(documents)
```
- `@st.cache_resource`:**关键装饰器**。被它装饰的函数,结果会被缓存,后续调用直接返回缓存对象,**不重新执行函数体**。
- 所以这 5 个文件只在首次读一次,SimpleRAG 只建一次索引。后续所有重跑都复用同一个对象。
- `encoding='utf-8'`:**中文必须显式指定**,否则 Windows 默认 GBK 会乱码或报错。这是跨平台中文处理的硬性要求。
- `with open(...)`:上下文管理器,自动关文件,即使出异常也不会泄露文件句柄。
- **`cache_resource` vs `cache_data` 的区别**(高频考点):
  - `cache_resource`:缓存"全局资源/单例"(数据库连接、ML模型、这里的检索器),**所有用户/会话共享同一个**,不做拷贝。
  - `cache_data`:缓存"数据"(DataFrame、计算结果),每次返回**拷贝**,防止被意外修改。
  - 这里检索器是有状态的服务对象,且只读,用 `cache_resource` 正确。

```python
def init_session_state():
    if 'current_result' not in st.session_state:
        st.session_state.current_result = None
    if 'history' not in st.session_state:
        st.session_state.history = []
```
- `st.session_state`:**会话级**的持久字典,跨脚本重跑保留(但不同浏览器会话相互独立)。
- `if 'key' not in ...`:**只在第一次初始化**,后续重跑不覆盖已有数据。这是 session_state 的标准初始化范式。
- `current_result` 存当前结果,`history` 存历史记录列表。

```python
def main():
    st.set_page_config(
        page_title="SmartGuide AI",
        page_icon="🤖",
        layout="wide"
    )
```
- `set_page_config`:设置页面标题、图标、布局。**必须是第一个 st 命令**,否则报错。`layout="wide"` 用宽屏布局。

```python
    st.title("🤖 SmartGuide AI - 智能硬件客服助手")
    st.markdown("""...支持的设备...""")

    init_session_state()
    rag = load_knowledge_base()
    llm_client = LLMClient()
    evaluator = Evaluator()
```
- 渲染标题和说明。
- 初始化 session,加载(缓存的)知识库,创建 LLM 客户端和评测器。
- **注意**:`rag = load_knowledge_base()` 虽然每次重跑都"调用",但因为有 `@st.cache_resource`,实际只在首次真正执行,之后秒返回缓存。`LLMClient()` 和 `Evaluator()` 是轻量对象,每次新建无所谓。

```python
    col1, col2 = st.columns([4, 1])
    with col1:
        question = st.text_input(
            "请输入您的问题:",
            placeholder="例如: 智能手环如何连接手机?",
            key="question_input"
        )
    with col2:
        st.write("")  # 占位对齐
        submit = st.button("🔍 提问", type="primary", use_container_width=True)
```
- `st.columns([4, 1])`:把一行分成宽度比 4:1 的两列(输入框宽、按钮窄)。
- `with col1:` / `with col2:`:在对应列里放组件。
- `key="question_input"`:给组件一个唯一 key,Streamlit 用它在 session_state 里追踪这个组件的值。
- `st.write("")`:空占位,把按钮往下推一点和输入框对齐(因为输入框有 label 占了一行高度)。是个对齐小技巧。
- `submit` 是布尔值:这次重跑中按钮**是否被点击**。

```python
    if submit and question:
        if not evaluator.is_valid_question(question):
            st.warning("⚠️ 问题无效:不符合客服场景(问题过短或不包含设备关键词)")
            st.session_state.current_result = None
        else:
            with st.spinner("🔍 检索知识库..."):
                retrieved_docs = rag.retrieve(question, top_k=config.TOP_K)
```
- `if submit and question`:**点了按钮 且 输入框非空**才处理。`question` 空字符串是 falsy,顺便挡了空提交。
- **第一级过滤**:无效问题直接警告,清空结果,不往下走(不调 AI)。这就是产品文档说的"省钱"落地点。
- `with st.spinner("...")`:显示加载动画,`with` 块内的代码执行期间转圈,执行完自动消失。**分阶段 spinner** 让用户知道当前在哪一步。

```python
            if not retrieved_docs:
                st.warning("❌ 未找到相关内容")
                st.session_state.current_result = None
            else:
                with st.spinner("🤔 AI思考中..."):
                    try:
                        answer = llm_client.answer_question(question, retrieved_docs)

                        category, category_note = evaluator.classify_question(question, answer)

                        if category == "有效":
                            accuracy, detail = evaluator.calculate_accuracy_a(answer, retrieved_docs)
                        else:
                            accuracy, detail = None, None
```
- 检索为空也兜底处理。
- `try` 包住 AI 调用——网络/API 错误不能让整个页面崩。
- 拿到回答后分类。只有"有效"才算准确率,否则 accuracy/detail 设 None。**和评测引擎的"超纲不计入准确率"逻辑一致**。

```python
                        st.session_state.current_result = {
                            "question": question,
                            "answer": answer,
                            "retrieved_docs": retrieved_docs,
                            "category": category,
                            "category_note": category_note,
                            "accuracy": accuracy,
                            "detail": detail
                        }

                        st.session_state.history.append(st.session_state.current_result)

                    except Exception as e:
                        st.error(f"❌ 调用AI失败: {str(e)}")
                        st.session_state.current_result = None
```
- **把所有结果打包存进 `session_state.current_result`**——这是"截图后答案不丢"的核心。结果存在会话状态里,下次重跑从这里读,而不是依赖局部变量。
- `history.append(...)`:追加到历史,供"实时评测"页统计。
- `except`:出错显示友好提示,不抛红色堆栈给用户。

> **隐藏的坑**:`history.append` 存的是 `current_result` 的**引用**,不是拷贝。这里因为每次都新建字典所以没事,但如果后面修改了 current_result,history 里的也会变。面试官可能问"这里会不会有引用别名问题",答:当前安全,因为每次是全新字典。

```python
    if st.session_state.current_result:
        result = st.session_state.current_result
        st.success("✅ 回答生成完毕")
        st.markdown("### 💬 AI回答")
        st.info(result["answer"])
```
- **渲染块独立于处理块**:无论这次重跑是不是因为点了提问,只要 session 里有结果就渲染。**这就是为什么按其他键(触发重跑)答案还在**——渲染不依赖"这次是否点了按钮",只依赖 session 状态。这是 Streamlit 正确写法的精髓。

```python
        col1, col2, col3 = st.columns(3)
        with col1:
            category_color = {"有效": "🟢", "超纲": "🟡", "无效": "🔴"}
            st.metric("问题分类", f"{category_color[result['category']]} {result['category']}")
        with col2:
            if result["accuracy"] is not None:
                st.metric("有据可查率", f"{result['accuracy']:.1%}")
            else:
                st.metric("有据可查率", "不适用")
        with col3:
            st.metric("引用文档数", len(result["retrieved_docs"]))
```
- 三列指标卡。`st.metric` 是 Streamlit 的指标展示组件(大数字)。
- `f"{...:.1%}"`:把 0.667 格式化成 "66.7%"。`.1%` = 一位小数的百分比。
- `is not None` 判断:超纲/无效问题 accuracy 是 None,显示"不适用"。**注意不能用 `if result["accuracy"]`**,因为准确率可能是 0.0(falsy),会被误判。必须 `is not None`——这是个经典坑。

```python
        st.caption(f"📝 {result['category_note']}")

        with st.expander("📚 查看引用文档"):
            for i, (device, content, score) in enumerate(result["retrieved_docs"], 1):
                st.markdown(f"**[文档{i}] {device}** (相似度: {score:.3f})")
                st.text(content[:200] + "..." if len(content) > 200 else content)
                st.divider()
```
- `st.expander`:可折叠区域,默认收起,点击展开。用来放"次要但有用"的信息(引用文档),不占主界面。
- `content[:200] + "..." if len(content) > 200 else content`:三元表达式,超 200 字符截断加省略号,否则原样。**避免长文档撑爆界面**。
- `st.divider()`:分隔线。

```python
        if result["detail"]:
            with st.expander("🔍 关键词匹配详情"):
                detail = result["detail"]
                st.write(f"- 答案关键词总数: {detail['total_keywords']}")
                st.write(f"- 有文档支持的: {detail['supported_keywords']}")
                st.write(f"- 无文档支持的: {len(detail['unsupported_keywords'])}")
                if detail['unsupported_keywords']:
                    st.warning(f"缺乏依据的词: {', '.join(detail['unsupported_keywords'][:10])}")
```
- 只有有效问题(detail 非 None)才展示关键词详情。
- 把有据可查率的中间结果可视化:总词数、有依据、无依据。
- `[:10]`:无依据的词最多显示 10 个,防刷屏。
- **这是"可解释 AI"的落地**:不只给分数,还告诉你"哪些词没依据",让用户能判断这个准确率可不可信。

```python
    with st.sidebar:
        st.header("📊 功能导航")
        st.page_link("问答系统.py", label="🏠 问答系统", icon="🏠")
        st.page_link("pages/实时评测.py", label="📈 实时评测", icon="📈")
        st.page_link("pages/一键评测.py", label="🎯 一键评测", icon="🎯")
        ...
```
- `with st.sidebar:`:侧边栏容器。
- `st.page_link`:多页面导航链接。**Streamlit 多页面应用约定**:主文件在根目录,其他页面放 `pages/` 文件夹,会自动出现在导航里。

```python
if __name__ == "__main__":
    main()
```
- Python 标准入口保护。直接运行时执行 main()。(Streamlit 实际是 `streamlit run 问答系统.py` 启动,但保留这个是好习惯。)

### 5.4 这个文件的面试考点

- **Q:Streamlit 的执行模型是什么?** → 每次交互全脚本重跑,不是事件回调。
- **Q:为什么要用 session_state?** → 局部变量每次重跑被重置,跨次保留的数据(结果、历史)必须放 session_state。
- **Q:`cache_resource` 和 `cache_data` 区别?** → 前者缓存共享单例(连接/模型),不拷贝;后者缓存数据,返回拷贝防篡改。
- **Q:"截图后答案消失"是怎么修的?** → 把结果存 session_state,渲染逻辑只依赖 session 状态而非"是否点了按钮",所以任何重跑都能重新渲染出答案。
- **Q:`if result["accuracy"]` 有什么坑?** → 准确率为 0.0 时是 falsy 会误判,必须用 `is not None`。
- **Q:中文读文件要注意什么?** → 显式 `encoding='utf-8'`,否则 Windows GBK 乱码。

---

## 第六部分:`pages/实时评测.py`

### 6.1 整体职责

读 `session_state.history`,筛出有效问题,用 plotly 画柱状图+折线图,展示准确率趋势和统计。**纯展示页,不调用 AI**。

### 6.2 逐行解读

```python
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
```
- `plotly.graph_objects as go`:plotly 的底层 API,可精细控制每个图元(Bar、Scatter)。
- `make_subplots`:创建多子图布局。

```python
def init_session_state():
    if 'history' not in st.session_state:
        st.session_state.history = []
```
- **注意**:每个页面都各自初始化 history。因为用户可能直接打开这个页面(没先去主页),要防 `history` 不存在报错。**防御性初始化**。session_state 是跨页面共享的,所以主页存的 history 这里读得到。

```python
    if not st.session_state.history:
        st.warning("⚠️ 暂无提问记录，请先在主页进行提问")
        return
```
- **空状态处理**:没数据就提示去主页,`return` 提前退出。避免下面对空列表做统计出错。良好的空状态 UX。

```python
    valid_records = [
        r for r in st.session_state.history
        if r.get("category") == "有效" and r.get("accuracy") is not None
    ]
```
- 列表推导筛选**有效且有准确率**的记录。
- `r.get("category")`:用 `.get()` 而非 `r["category"]`,**key 不存在时返回 None 而非抛 KeyError**,更健壮。

```python
    if not valid_records:
        st.info("📊 当前所有问题均为超纲或无效，无准确率数据")
        categories = [r.get("category", "未知") for r in st.session_state.history]
        category_counts = pd.Series(categories).value_counts()
        ...
        return
```
- **第二层空状态**:有历史但全是超纲/无效,没有可画的准确率。这时改为展示分类统计。
- `pd.Series(categories).value_counts()`:pandas 经典用法,统计每个分类出现次数。
- **考点:两层空状态**——"完全没记录"和"有记录但无有效数据"是不同的空状态,分别处理。新手常只处理第一种。

```python
    questions = [r["question"] for r in valid_records]
    accuracies = [r["accuracy"] for r in valid_records]
    avg_accuracy = sum(accuracies) / len(accuracies)
```
- 提取问题列表、准确率列表,算平均。这里 valid_records 已非空(上面 return 挡了),除法安全。

```python
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("每次提问准确率 (柱状图)", "准确率趋势 (折线图)"),
        vertical_spacing=0.15,
        row_heights=[0.5, 0.5]
    )
```
- 创建 2 行 1 列的子图。上下各放一个图。`vertical_spacing` 控制间距,`row_heights` 控制高度比例。

```python
    colors = ['#1f77b4' if acc >= 0.7 else '#ff7f0e' if acc >= 0.5 else '#d62728'
              for acc in accuracies]
```
- **嵌套三元表达式做颜色编码**:≥0.7 蓝、0.5~0.7 橙、<0.5 红。
- 读法:`A if 条件1 else (B if 条件2 else C)`。
- **可读性考点**:嵌套三元省事但难读,面试官可能问"能不能写得更清晰"。可以抽成函数 `def color_of(acc): ...`。

```python
    fig.add_trace(
        go.Bar(
            x=list(range(1, len(accuracies) + 1)),
            y=accuracies,
            marker_color=colors,
            text=[f"{acc:.1%}" for acc in accuracies],
            textposition='outside',
            name="准确率",
            hovertemplate="<b>第%{x}次提问</b><br>准确率: %{y:.1%}<extra></extra>"
        ),
        row=1, col=1
    )
```
- `go.Bar`:柱状图。`x` 是序号 1~N,`y` 是准确率。
- `marker_color=colors`:每根柱用上面算的颜色。
- `text` + `textposition='outside'`:柱顶显示百分比标签。
- `hovertemplate`:鼠标悬停提示模板。`%{x}`/`%{y}` 是占位符,`<extra></extra>` 去掉默认的多余信息框。
- `row=1, col=1`:放第一个子图。

```python
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(accuracies) + 1)),
            y=accuracies,
            mode='lines+markers',
            ...
        ),
        row=2, col=1
    )
```
- `go.Scatter` + `mode='lines+markers'`:带数据点的折线图,放第二个子图。

```python
    fig.add_hline(
        y=avg_accuracy,
        line_dash="dash",
        line_color="red",
        annotation_text=f"平均: {avg_accuracy:.1%}",
        annotation_position="right",
        row=2, col=1
    )
```
- `add_hline`:加一条水平参考线(平均值)。虚线红色,带标注。**让人一眼看出每个点在平均线上还是下**。

```python
    fig.update_xaxes(title_text="提问序号", row=1, col=1)
    ...
    fig.update_yaxes(title_text="准确率", range=[0, 1.05], row=1, col=1)
    ...
    fig.update_layout(height=800, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
```
- 设置坐标轴标题。`range=[0, 1.05]`:**Y 轴固定 0~1**,留 0.05 余量让顶部标签不被截断。**固定量程很重要**——否则 plotly 自动缩放,两次数据看起来"尺度不一致",误导判断。
- `st.plotly_chart(..., use_container_width=True)`:渲染图表并自适应容器宽度。

```python
    df = pd.DataFrame({...})
    st.dataframe(df, use_container_width=True, hide_index=True)

    with st.expander("🔍 查看完整统计(含超纲/无效问题)"):
        all_categories = [r.get("category", "未知") for r in st.session_state.history]
        category_counts = pd.Series(all_categories).value_counts()
        st.write("**问题分类分布:**")
        for cat, count in category_counts.items():
            st.write(f"- {cat}: {count} 条")
```
- 用 DataFrame 展示明细表,`hide_index=True` 隐藏行号。
- 折叠区放完整统计(含超纲/无效),**主区只看有效问题,噪声折叠备查**——和评测引擎的指标纯净度一脉相承。

### 6.3 这个文件的面试考点

- **Q:为什么每个页面都初始化 history?** → 用户可能直接进入此页,防 KeyError;session_state 跨页共享,主页存的能读到。
- **Q:你处理了哪些空状态?** → 两层:完全无记录、有记录但无有效准确率数据,分别给不同提示。
- **Q:Y 轴为什么固定 range?** → 防止自动缩放导致不同数据尺度不一致,误导对比。
- **Q:`r.get("category")` 和 `r["category"]` 区别?** → `.get()` key 不存在返回 None,`[]` 抛 KeyError。前者更健壮。

---

## 附录 A:`src/telemetry.py` 埋点与自助解决率

### A.1 为什么要埋点

`groundedness` 这类技术指标回答不了一个问题:**这套系统到底有没有用**。
业务方要的是"省了多少人力",所以必须把每一次问答记下来,最后算成一个业务指标。

### A.2 存储结构

单文件 SQLite(`data/telemetry.db`),一张 `events` 表:

| 字段 | 含义 |
|---|---|
| `session_id` | 浏览器会话（UUID） |
| `question` / `effective_question` | 原问题 / 补全设备后的实际问题 |
| `answer` | 生成的回答 |
| `category` | 有效 / 超纲 / 无效 |
| `groundedness` | 有据可查率 |
| `retrieved_count` | 召回条数 |
| `latency_ms` | 端到端耗时 |
| `retriever` / `device` | 用的哪个检索器、识别到哪个设备 |
| `resolved` | 显式反馈：1 解决 / 0 未解决 / NULL 未表态 |
| `feedback` | helpful / unhelpful |
| `follow_up` | 是否被判定为"追问" |

`ts` 和 `session_id` 建了索引，因为看板永远按时间和会话查。

### A.3 三个核心动作

**① 写一条提问**（`log_query`）
插入之前先做一个判断：**同一个 session 内、10 分钟窗口内、上一条还没被标记解决的记录，标记为 `follow_up=1`**。

> 这一步是整个埋点设计的关键。它把"用户没点反馈"这个信息缺口，用行为补上了——
> 你接着问，说明上一条没解决。

**② 记反馈**（`log_feedback`）
用户点 👍/👎，更新目标记录的 `resolved` 和 `feedback`。

**③ 算自助解决率**（`compute_resolution_rate`）

```
显式解决 = resolved == 1
隐式解决 = resolved IS NULL AND follow_up == 0     ← 没表态，且之后没再问
自助解决率 = (显式解决 + 隐式解决) / 总提问
```

同时返回：显式反馈条数、反馈覆盖率、追问数、平均有据可查率、拒答率。

### A.4 这个指标的诚实之处

- 隐式解决是**近似**，不是真值。用户可能没解决但也没再问（放弃了）。
- 所以看板必须同时展示 **"显式反馈覆盖率"**——覆盖率低的时候，自助解决率的可信度就要打折扣。
- 面试说法：**"这是一个有明确假设的近似指标，我把假设写在页面的折叠说明里，而不是藏起来。"**

### A.5 面试考点

- **Q:为什么用 SQLite 不用 MySQL?** → 单机部署、零运维、够用。数据量上来再换。
- **Q:10 分钟窗口怎么定的?** → 经验值。太短会把"看完答案去试了一下再回来问"误判为追问；太长会把真的新问题算成追问。可配置。
- **Q:埋点失败了怎么办?** → 全部 try/except 吞掉，只记到 `st.session_state.last_telemetry_error`。**埋点不能影响主流程。**
- **Q:怎么防止刷数据?** → 目前没有。生产要加会话校验、频控、异常检测。

---

## 附录 B:`src/evaluation_runner.py` 可复现评测引擎

### B.1 它解决什么问题

页面上点一下跑评测，结果留在浏览器里——**不可复现、不可对比、不可回归**。
这个模块把评测变成命令行可跑的、落盘留档的工程动作。

### B.2 三件事

**① 读测试集**（`load_test_set`）
读 `evaluation/test_set.csv`，UTF-8 BOM，过滤空行和重复表头。

**② 逐题评测**（`evaluate_one`）
```
检索 → 没召回就直接判超纲拒答（不调模型）
     → 有召回则调模型 → 分类 → 判是否拒答
     → 有效题才算有据可查率
     → 有标准答案就算正确率
     → 超纲题：拒答了就算正确
```

**注意一个设计**：`correct = recall ≥ 0.6`（关键词判分）或 LLM judge 的判定。
两种口径可切换，因为**关键词判分便宜但死板，LLM judge 灵活但不稳定**，
评测报告里必须写清楚用的哪种（写进 `meta.json`）。

**③ 汇总落盘**（`summarize` + `save_results`）

| 文件 | 内容 |
|---|---|
| `results.jsonl` | 逐题：问题、回答、召回数、两指标、耗时、错误 |
| `results.csv` | 同上，表格版 |
| `summary.json` | 汇总指标 |
| `meta.json` | **模型、温度、检索器、top_k、判分方式、时间戳** |

> `meta.json` 是这个模块最重要的产出。没有它，几个月后你看到一组数字，
> 根本不知道当时跑的什么配置，也就没法解释"上次是 74% 这次怎么 60% 了"。

### B.3 汇总里有哪些指标

`total / valid / out_of_scope / invalid_or_error / errors /
groundedness_avg / correctness_recall_avg / correct_rate /
refusal_rate / task_success_rate`

**`task_success_rate` 的定义**：（答对的有效题 + 正确拒答的超纲题）/ 总题数。
这是唯一一个"把正面和负面题放在一起算"的指标，最接近业务视角。

### B.4 面试考点

- **Q:为什么评测要能命令行跑?** → 要进 CI、要能回归、要能对比不同配置。
- **Q:关键词判分和 LLM judge 怎么选?** → 关键词判分快且确定，适合回归；LLM judge 适合评估语义，但同一份数据跑两次结果可能不同，不能做严格回归。
- **Q:为什么记录 meta?** → 没有配置元数据的指标是不可比的。
- **Q:样本 100 题够吗?** → 工程上够用，统计上不够。置信区间会很宽，不能用于对外承诺精度。

---

## 附录 C:`src/auth.py` 与 `src/admin_panel.py` 服务端

### C.1 鉴权

`require_login()`：单密码门。密码在 `config.ADMIN_PASSWORD`，登录后写
`st.session_state.admin_authed`，页面顶部渲染退出按钮。

**明确说清它不是什么**：不是用户体系、没有角色、没有哈希存储、没有会话过期。
面试主动交代，并给生产方案（接 SSO / OAuth，角色权限，审计日志）。

### C.2 服务端面板

`pages/` 下三个页面都在登录后可用：

| 页面 | 内容 |
|---|---|
| `实时评测.py` | 当前会话内的逐题有据可查率趋势（读 session_state，不落库） |
| `一键评测.py` | 100 题批量评测，可选 judge（keyword/llm）、检索器（bm25/bigram/hybrid），调用 `evaluation_runner` |
| `自助解决率.py` | 读 SQLite，展示自助解决率、提问量、反馈覆盖率、追问率、趋势、分类分布、高频未解决 TOP8、CSV 导出 |

`src/admin_panel.py` 还提供**知识库管理**：按设备预览段落、追加新块、删除选中段，
写回 txt 后 `st.cache_resource.clear()` 重建索引。

> **这是"知识库运营"的雏形**：客户自己能维护内容，不用改代码、不用发版。
> 对交付型岗位来说，这一点比模型效果更值钱。

### C.3 面试考点

- **Q:知识库改了为什么要清缓存?** → 检索器的索引是在 `@st.cache_resource` 里构建的，
  不清缓存就一直用旧索引。
- **Q:删除段落有什么风险?** → 索引重建是同步的，大文档会卡；而且没有版本回滚。
  生产要加备份和灰度。
- **Q:管理员改了文档，怎么证明没改坏?** → 改完重跑评测，对比 `summary.json`。
  这就是评测集存在的第二个价值——**变更回归**。

---

## 附录 D:文档同步记录

- **2026-10-26**：按当前代码同步本文件。主要修正：
  - 检索层由 `simple_rag.py`（bigram + Jaccard）改为 `retrieval.py`（默认 BM25，可选向量与混合检索）；
  - `top_k` 由 3 改为 5；
  - 大模型由 MiniMax 改为 DeepSeek（`deepseek-chat`，temperature 0.3）；
  - 指标名"方案A准确率"统一为"有据可查率（groundedness）"；
  - 补充埋点、可复现评测、鉴权与服务端面板三个模块。





