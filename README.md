# SmartGuide AI - 智能硬件客服助手

基于 RAG 的智能客服系统，回答智能手环、手表、音箱、台灯、门锁的使用问题。

## 项目特点

- **三级问题分类**: 无效/超纲/有效问题自动识别
- **方案A准确率**: 衡量防幻觉能力，而非标准答案匹配
- **API限流重试**: 自动处理429错误，指数退避重试
- **实时评测可视化**: 柱状图+折线图展示准确率趋势
- **一键批量评测**: 42题测试集快速验证Prompt优化效果

## 技术栈

- **语言**: Python 3.12
- **大模型**: MiniMax (支持任何OpenAI兼容API)
- **界面**: Streamlit
- **检索**: Bigram关键词匹配 (无需向量数据库)
- **可视化**: Plotly

## 快速开始

### 1. 克隆仓库

```bash
git clone git@github.com:xhygaaaaa/smart-ai.git
cd smart-ai
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

国内用户建议使用镜像加速:
```bash
pip install -i https://mirrors.aliyun.com/pypi/simple/ -r requirements.txt
```

### 3. 配置API密钥

编辑 `.env` 文件，填入你的 MiniMax API Key:

```env
LLM_BASE_URL=https://api.minimax.chat/v1
LLM_API_KEY=你的-minimax-api-key
LLM_MODEL=abab6.5-chat
LLM_TEMPERATURE=0.3
TOP_K=3
```

**MiniMax API 申请**: https://api.minimax.chat/

也可换成其他兼容OpenAI格式的API:
- 通义千问: `https://dashscope.aliyuncs.com/compatible-mode/v1`
- DeepSeek: `https://api.deepseek.com`
- 智谱GLM: `https://open.bigmodel.cn/api/paas/v4`

### 4. 启动应用

```bash
streamlit run 问答系统.py
```

浏览器自动打开 `http://localhost:8501`

## 项目结构

```
smartguide-ai/
├── 问答系统.py           # 主页 - 问答功能
├── pages/
│   ├── 实时评测.py       # 实时准确率统计与可视化
│   └── 一键评测.py       # 批量跑42题测试集
├── src/
│   ├── simple_rag.py     # RAG检索模块
│   ├── llm_client.py     # LLM调用(含429重试)
│   └── evaluator.py      # 评测逻辑(三级分类+方案A)
├── data/                 # 5份产品使用指南
├── evaluation/           # 42题测试集
├── config.py             # 项目配置
└── requirements.txt      # 依赖清单
```

## 使用说明

### 问答系统

1. 在主页输入问题，如 "智能手环如何连接手机?"
2. 系统自动检索知识库并生成回答
3. 显示问题分类、准确率、引用文档

### 实时评测

- 自动统计所有有效问题的准确率
- 可视化展示准确率变化趋势
- 支持柱状图和折线图双重展示

### 一键评测

- 批量跑完整个42题测试集
- 查看各类问题(连接/功能/规格/故障/超纲)的表现
- 导出CSV评测报告

## 核心设计

### 三级问题分类

1. **无效问题**: 完全不沾边，直接拦截不调用AI
2. **超纲问题**: 沾边但知识库无内容(如价格/保修)，AI拒答，不计入准确率
3. **有效问题**: 正常回答，计入准确率统计

### 方案A准确率

不是跟标准答案对比，而是检查**AI回答的关键词有多少能在引用文档中找到依据**。

衡量的是"防幻觉能力"，避免AI编造知识库中不存在的内容。

### 关键词提取算法

避免粗暴的2字滑窗(会切出"约天""ma"等垃圾词)，采用：

1. 正则提取"数字+单位"组合 (如 "7天", "5ATM")
2. 按标点和连接词切分短语
3. 保留2-5字的有意义中文短语

### API限流处理

撞到429错误时，自动指数退避重试(等待3/6/9秒)，最多3次。

如仍频繁超限，可修改 `pages/一键评测.py` 中的 `time.sleep(1)` 为更长延时。

## 常见问题

### 依赖安装报错

**症状**: 提示 DLL 加载失败  
**原因**: 误装了 chromadb/torch 等重量级包  
**解决**: 卸载后只安装 requirements.txt 中的6个依赖

```bash
pip uninstall -y torch sentence-transformers chromadb
pip install -r requirements.txt
```

### 答案截图后消失

**原因**: Streamlit 全脚本重跑模式  
**解决**: 已用 `session_state` 持久化存储，正常使用不会出现此问题

### API限流429

**症状**: 一键评测时多次报错  
**解决**: 已实现自动重试。如仍超限，加长评测间隔:

```python
# pages/一键评测.py 第44行
time.sleep(2)  # 原来是1秒，改成2或3秒
```

### 中文乱码

**症状**: 终端输出乱码  
**解决**: Streamlit 网页里显示正常即可。如需命令行输出，在脚本开头加:

```python
import sys, io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
```

## 开发说明

### 添加新设备

1. 在 `data/` 下创建设备使用指南 txt 文件
2. 在 `config.py` 的 `MANUALS` 字典中添加路径
3. 在 `evaluation/test_set.csv` 中添加测试题

### 调整准确率算法

修改 `src/evaluator.py` 中的 `calculate_accuracy_a` 方法。

### 更换大模型

只需修改 `.env` 中的 `LLM_BASE_URL` 和 `LLM_MODEL`，无需改代码。

## 许可证

MIT License

## 作者

Created with Claude Opus 4.8