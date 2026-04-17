# 食谱 RAG（Programmers_cook）

基于 **RAG（检索增强生成）** 的中文食谱问答小项目：从 **`data/` 下 Markdown** 菜谱建库，结合 **BM25 + 向量混合检索** 与 **大模型** 回答「怎么做」「推荐几道菜」等问题。

## 功能概览

- **数据**：递归加载 `data/` 下所有 `.md`，解析路径中的分类目录并写入元数据（菜品名、难度等）。
- **分块**：`MarkdownHeaderTextSplitter` 按标题结构切分，保留父子文档关系以便检索子块、生成时用整份菜谱。
- **索引**：FAISS + HuggingFace 嵌入（默认 **BAAI/bge-large-zh**，支持本地 `hf_models` 目录）。若 `vector_index/` 下已有索引则启动时加载，否则构建并保存。**支持增量更新**，无需每次重建完整索引。
- **检索**：向量相似度 + BM25，RRF 融合；BM25 使用 **jieba**（`lcut_for_search`）中文分词；菜名出现在问句时的前置加权；列表类查询支持按父文档去重以推荐多道菜。
- **生成**：OpenAI 兼容接口（如阿里云 DashScope）调用对话模型；查询路由含 **list / detail / general / ingredient / compare_difficulty**，可选流式输出。
- **缓存**：基于文件的查询缓存机制，支持 TTL 过期，大幅提高重复查询的响应速度。
- **多轮对话**：支持上下文记忆、查询重写、代词消解，实现连贯的多轮问答体验。
- **历史对话侧边栏**：Web 界面支持查看和切换历史对话，支持新建和删除会话。
- **RAG 评估**：使用大模型对系统回答进行多维度评估（相关性、准确性、完整性、清晰度、幻觉程度）。
- **测试**：完整的单元测试覆盖，确保代码质量。

## 环境要求

- Python 3.10+（建议；项目使用 `str | None` 等写法）
- 可访问所选大模型 API（或自建兼容 OpenAI 的网关）
- 嵌入模型：首次可从 Hugging Face 拉取，或预先下载到本地（见下文）

## 快速开始

### 1. 安装依赖

```bash
cd Programmers_cook
pip install -r requirements.txt
```

### 2. 配置环境变量

在项目根目录创建 `.env`（不要将含密钥的文件提交到版本库）：

```env
OPENAI_API_KEY=你的API密钥
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen3.5-35b-a3b
```

说明：`GenerationIntegrationModule` 使用 `ChatOpenAI`，通过 `OPENAI_BASE_URL` 指向兼容 OpenAI 的服务即可切换厂商与模型名。

### 3. 准备嵌入模型（推荐本地）

默认配置为 **`bge-large-zh`**。二选一：

- **本地目录**（优先）：将模型完整下载到 **`hf_models/bge-large-zh/`**（与 `hf_models/bge-large-en-v1.5` 同级），程序会自动使用该路径。
- **在线**：若无上述目录，则使用 Hub 上的 **`BAAI/bge-large-zh`**（需网络）。

下载示例（在项目根目录执行）：

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download BAAI/bge-large-zh --local-dir hf_models/bge-large-zh
```

国内网络可使用镜像，例如（CMD）：

```bash
set HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download BAAI/bge-large-zh --local-dir hf_models/bge-large-zh
```

PowerShell：

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
huggingface-cli download BAAI/bge-large-zh --local-dir hf_models/bge-large-zh
```

### 4. 命令行交互

```bash
python main.py
```

首次运行若不存在 `vector_index/`，会自动构建 FAISS 索引并保存；之后默认加载已有索引以加快启动。

**增量更新**：若修改了 `data/` 下菜谱，系统会自动检测变更并执行增量更新，无需删除整个索引目录。

### 5. Web 界面（FastAPI）

在项目根目录执行（启动时会加载嵌入模型并构建/加载向量索引，**首次较慢**）：

```bash
uvicorn web_server:app --host 0.0.0.0 --port 8000
```

浏览器打开 **http://127.0.0.1:8000/** 。页面会轮询 `/api/health` 直至知识库就绪后再允许提问；支持**流式**与**非流式**回答。

**Web 界面功能：**
- 左侧**历史对话侧边栏**：查看所有历史会话
- **新建对话**：点击「+ 新建对话」开始新会话
- **切换对话**：点击历史会话条目加载该对话
- **删除对话**：点击会话右侧 × 按钮删除

也可执行：`python web_server.py`（默认监听 `0.0.0.0:8000`，无热重载）。

**更换嵌入模型后**，请删除旧的 **`vector_index/`** 再运行，否则向量维度或语义空间不一致会导致加载失败或检索异常。

### 6. 运行测试

```bash
python run_tests.py
```

或使用 pytest：

```bash
python -m pytest tests/ -v
```

### 7. RAG 系统评估

使用大模型对 RAG 系统进行多维度评估：

```bash
python evaluate_rag.py
```

评估维度（0-5 分）：
- **相关性**：回答是否与问题相关
- **准确性**：回答内容是否准确，基于检索文档
- **完整性**：回答是否完整覆盖问题
- **清晰度**：回答是否清晰易懂
- **幻觉程度**：是否存在虚构内容（越低越好）
- **综合得分**：整体表现

评估结果会保存到 `evaluation_results/` 目录。

**自定义测试用例**：编辑 `evaluation_test_cases.json` 添加你自己的问题和参考回答。

**使用示例**：查看 `evaluation_example.py` 了解更多使用方式。

## 项目结构

```
Programmers_cook/
├── main.py                      # 入口：RecipeRAGSystem 交互问答
├── web_server.py                # FastAPI：Web UI + /api/chat、/api/health、会话管理 API
├── run_tests.py                 # 运行所有单元测试
├── evaluate_rag.py              # RAG 系统评估脚本
├── evaluation_example.py        # RAG 评估使用示例
├── evaluation_test_cases.json   # 评估测试用例
├── static/
│   └── index.html               # 前端单页（含历史对话侧边栏）
├── config.py                    # RAGConfig：数据路径、索引路径、嵌入模型、缓存配置等
├── requirements.txt
├── .env                         # 本地创建：API 与模型（勿提交密钥）
├── data/                        # 默认菜谱 Markdown（按子目录分类）
├── hf_models/                   # 可选：本地 HuggingFace 嵌入模型目录
├── vector_index/                # 构建后生成：FAISS 索引（可删除以强制重建）
│   └── index_metadata.json      # 索引元数据（用于增量更新）
├── query_cache/                 # 查询缓存目录
├── conversation_history/        # 对话历史目录
├── evaluation_results/          # 评估结果目录
├── rag_modules/
│   ├── __init__.py
│   ├── data_preparation.py      # 加载文档、元数据、Markdown 分块、父文档聚合
│   ├── index_construction.py    # 嵌入模型解析、FAISS 构建/保存/加载、增量更新
│   ├── retrieval_optimization.py  # 混合检索、元数据过滤、列表去重
│   ├── generation_integration.py # LLM、路由、改写、列表/详细回答、多轮对话
│   ├── query_cache.py           # 查询缓存模块
│   ├── index_incremental.py     # 索引增量更新模块
│   ├── conversation_manager.py  # 对话历史管理模块
│   └── rag_evaluator.py         # RAG 系统评估模块
└── tests/                       # 单元测试目录
    ├── __init__.py
    ├── test_data_preparation.py
    ├── test_retrieval_optimization.py
    ├── test_index_construction.py
    └── test_conversation_manager.py
```

## 配置说明（`config.py`）

| 字段 | 含义 |
|------|------|
| `data_path` | 本地 Markdown 菜谱根目录，默认 `./data` |
| `index_save_path` | FAISS 索引目录，默认 `./vector_index` |
| `embedding_model` | 短名如 `bge-large-zh`：优先 `./hf_models/<名>/`，否则 `BAAI/<名>`；若含 `/` 则视为完整 Hub repo id |
| `llm_model` | 逻辑上的模型标识（实际调用以 `.env` 中 `LLM_MODEL` 为准） |
| `top_k` | 每次检索返回的 chunk 数量上限（可按需要调大） |
| `temperature` / `max_tokens` | 生成参数 |
| `enable_cache` | 是否启用查询缓存，默认 `True` |
| `cache_dir` | 查询缓存目录，默认 `./query_cache` |
| `cache_ttl` | 缓存过期时间（秒），默认 `3600`（1小时） |
| `enable_incremental_update` | 是否启用索引增量更新，默认 `True` |
| `index_metadata_path` | 索引元数据路径，默认 `./vector_index/index_metadata.json` |
| `enable_conversation` | 是否启用多轮对话，默认 `True` |
| `conversation_history_dir` | 对话历史目录，默认 `./conversation_history` |
| `max_history_length` | 最大历史消息数量，默认 `20` |
| `context_window_turns` | 上下文窗口轮数，默认 `3` |

## 交互式命令

在命令行交互模式下，支持以下特殊命令：

| 命令 | 说明 |
|------|------|
| `/stats` 或 `stats` | 查看缓存统计信息 |
| `/clear` 或 `clear` | 清空查询缓存 |
| `/update` 或 `update` | 检查文档更新 |
| `/apply` 或 `apply` | 应用文档更新 |
| `/conv` 或 `conv` | 查看对话统计信息 |
| `/new` 或 `new` | 开始新对话 |
| `/reset` 或 `reset` | 清空当前对话历史 |
| `退出` / `quit` / `exit` | 退出程序 |

## 核心功能说明

### 查询缓存

- 自动缓存非流式查询的结果
- 支持 TTL 过期机制，默认 1 小时
- 文档更新时自动清空相关缓存
- 可通过配置开关控制

### 索引增量更新

- 基于文件哈希检测文档变更
- 支持新增、修改、删除文档的增量处理
- 无需每次重建完整索引，大幅提升效率
- 变更时自动清空缓存保证数据一致性

### 多轮对话

- **上下文记忆**：自动保存对话历史，支持跨轮次引用
- **查询重写**：根据对话历史重写查询，解决代词指代问题
- **代词消解**：自动将"它"、"那道菜"等代词替换为具体内容
- **会话管理**：支持创建、加载、保存、删除会话
- **历史限制**：可配置最大历史消息数量和上下文窗口

**多轮对话示例：**
```
用户: 白灼虾怎么做？
助手: 白灼虾的做法是...[详细步骤]

用户: 它需要什么食材？
助手: 白灼虾需要的食材有...[自动识别"它"指代"白灼虾"]

用户: 那道菜难吗？
助手: 白灼虾的难度是简单...[自动识别"那道菜"指代"白灼虾"]
```

### 历史对话侧边栏

Web 界面新增左侧历史对话侧边栏功能：
- **会话列表**：按时间倒序显示所有历史会话
- **新建对话**：点击「+ 新建对话」按钮创建新会话
- **切换会话**：点击会话条目加载该会话的完整对话历史
- **删除会话**：点击会话右侧的 × 按钮删除不需要的会话
- **会话信息**：显示消息数量和最后更新时间

后端新增 API：
- `GET /api/sessions` - 获取所有会话列表
- `GET /api/sessions/{session_id}` - 获取指定会话详情
- `POST /api/sessions` - 创建新会话
- `DELETE /api/sessions/{session_id}` - 删除会话

### RAG 系统评估

使用大模型对 RAG 系统回答进行自动评估，支持多维度评分：

**评估维度：**
1. **相关性** - 回答与问题的相关程度
2. **准确性** - 内容准确性，是否基于检索文档
3. **完整性** - 是否完整覆盖问题各方面
4. **清晰度** - 回答是否清晰易懂
5. **幻觉程度** - 是否存在未检索到的虚构内容（越低越好）
6. **综合得分** - 整体表现评分

**使用方式：**
```bash
# 一键评估
python evaluate_rag.py

# 自定义测试用例
# 编辑 evaluation_test_cases.json
```

**输出结果：**
- 控制台打印评估汇总
- 详细结果保存到 `evaluation_results/` 目录
- 包含每个问题的评分和详细反馈

### 单元测试

- 数据准备模块测试：文档加载、分块、元数据提取
- 检索优化模块测试：BM25 预处理、RRF 重排、过滤逻辑
- 索引构建模块测试：索引构建、保存、加载、缓存操作
- 对话管理模块测试：消息管理、会话管理、历史序列化

## 使用提示

- **问具体菜名**（如「白灼虾怎么做」）：问句中包含 `dish_name`（与文件名一致）时检索会优先相关菜谱块。
- **推荐多道菜**：列表类问题在带分类/难度过滤时会扩大候选并按「一父文档一块」去重，避免多条结果来自同一道菜。
- **缓存命中**：相同问题会直接返回缓存结果，大幅提升响应速度。
- **增量更新**：修改菜谱后无需手动删除索引，系统会自动检测并更新。
- **多轮对话**：系统会自动保存对话上下文，支持追问和引用之前讨论的内容。
- **历史对话**：Web 界面左侧显示历史会话，可随时切换、新建或删除会话。
- **RAG 评估**：使用 `python evaluate_rag.py` 对系统进行自动化评估，了解各维度表现。

## 许可证与数据

菜谱数据位于 `data/`，结构与版权请遵循原数据提供方说明；本仓库代码以你本地使用与二次开发为准。
需要注意的是，data数据可以通过https://github.com/Anduin2017/HowToCook.git获取。
将dishes数据修改为data存放在项目下即可构建知识库。
本项目基于https://github.com/datawhalechina/all-in-rag.git 第八章源码二次开发。
