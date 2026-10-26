# Python Multi-format Document RAG

一个面向学习、校招面试讲解和 GitHub 展示的多格式文档检索增强问答项目。项目使用统一接口读取 TXT、CSV、PDF、PPTX，保留页码、幻灯片号、CSV 行号与分块编号，并在语义检索和 LLM 回答中返回可追溯来源。

本仓库为独立实现，不包含课程源码、品牌素材、个人数据或第三方测试文档。

## 技术栈

- Python 3.10–3.12
- LlamaIndex `Document`、`SentenceSplitter`、`VectorStoreIndex`
- HuggingFace Embedding，默认 `BAAI/bge-small-zh-v1.5`
- LlamaIndex 本地 `SimpleVectorStore` 持久化
- DashScope LLM，以及可配置 Base URL 的 OpenAI-compatible LLM（可接 OpenLux）
- pypdf、python-pptx、Python 标准库 CSV
- pytest 与 Mock Embedding/Mock LLM

## RAG 工作流程

```text
TXT / CSV / PDF / PPTX
        │
        ▼
按文件 / 行 / 页 / 幻灯片解析
        │
        ▼
Unicode 与空白清洗（保留段落）
        │
        ▼
LlamaIndex SentenceSplitter
        │
        ▼
Embedding → VectorStoreIndex → 本地持久化
        │
        ▼
Top-K 检索 → 相似度阈值 → LLM 回答
        │
        ▼
答案 + 文件名 / 页码 / 幻灯片号 / 行号 / 分块编号
```

详细设计见 [docs/architecture.md](docs/architecture.md)。

## 文档与元数据

| 格式 | 解析粒度 | 特有元数据 |
|---|---|---|
| TXT | 每个文件 | — |
| CSV | 每条数据行 | `row_number` |
| PDF | 每一页 | `page_number` |
| PPTX | 每张幻灯片 | `slide_number` |

所有分块都会包含：

- `source_file`
- `file_type`
- `source_id`
- `chunk_index`

索引中存储相对文件路径，而不是开发者机器上的绝对路径。

## Ubuntu 安装

以下命令适用于 Ubuntu 22.04 和 Python 3.10。建议使用独立虚拟环境：

```bash
cd /path/to/multiformat-rag-app
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
```

以上是轻量级开发和 Mock 测试环境。要运行真实 HuggingFace Embedding，建议先安装 CPU 版 PyTorch，再安装可选依赖，避免默认解析到不需要的 CUDA 包：

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[local-embedding,test]'
```

首次执行真实索引命令时，HuggingFace 可能需要从网络下载 Embedding 模型。测试套件使用 Mock Embedding，不下载模型，也不调用付费 API。

默认模型 `BAAI/bge-small-zh-v1.5` 为 MIT 许可、约 24M 参数、512 维。其
`model.safetensors` 文件约 95.8 MB；模型仓库同时提供另一份约 95.8 MB 的
PyTorch 权重，因此浏览整个仓库时会看到约 192 MB 的权重文件。实际加载器通常
只选择一种权重格式，此外还会下载配置和 tokenizer 小文件。

## 环境变量

```bash
cp .env.example .env
```

配置项：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `LLM_PROVIDER` | `dashscope` | `dashscope` 或 `openai_compatible` |
| `LLM_BASE_URL` | `https://api.openlux.ai/v1` | OpenAI-compatible Base URL |
| `LLM_API_KEY` | 空 | OpenAI-compatible provider 的本地密钥 |
| `LLM_MODEL` | 空 | 可选模型覆盖；应填写账户实际可用的模型 ID |
| `LLM_TIMEOUT_SECONDS` | `30` | OpenAI-compatible 请求超时秒数 |
| `DASHSCOPE_API_KEY` | 空 | DashScope provider 的本地密钥 |
| `RAG_EMBEDDING_MODEL` | `BAAI/bge-small-zh-v1.5` | HuggingFace 模型名或本地路径 |
| `RAG_LLM_MODEL` | `qwen-turbo` | 向后兼容的默认模型名；`LLM_MODEL` 非空时优先 |
| `RAG_INDEX_DIR` | `.rag_index` | 本地索引目录 |
| `RAG_CHUNK_SIZE` | `512` | SentenceSplitter token 数 |
| `RAG_CHUNK_OVERLAP` | `64` | 相邻分块重叠 token 数 |
| `RAG_TOP_K` | `5` | 默认检索数量 |
| `RAG_MIN_SIMILARITY` | `0.35` | 最低余弦相似度，数值越高越相似 |

程序不会输出完整 API Key。不要把 `.env` 提交到 Git。

## CLI 使用

查看帮助：

```bash
multiformat-rag --help
multiformat-rag index --help
multiformat-rag retrieve --help
multiformat-rag ask --help
```

### 构建索引

```bash
multiformat-rag index ./data \
  --persist-dir ./.rag_index \
  --chunk-size 512 \
  --chunk-overlap 64
```

若目标目录已经包含索引，可显式增加 `--overwrite`。该选项覆盖同名索引文件，但不会删除目录。
默认拒绝写入非空目录，用于避免误覆盖。请勿把日志或其他文件放进索引目录；重要
索引应先备份，因为进程在覆盖写入期间异常退出仍可能留下不完整文件。

### 仅检索，不调用 LLM

```bash
multiformat-rag retrieve "产品支持哪些文档格式？" \
  --persist-dir ./.rag_index \
  --top-k 5
```

JSON 输出：

```bash
multiformat-rag retrieve "产品支持哪些文档格式？" --json
```

### 检索增强问答

`ask` 会调用 `LLM_PROVIDER` 选择的真实接口，可能产生费用。默认仍使用
DashScope；程序不会在 provider 之间自动回退。缺少对应 Key 时会在加载模型和索引
之前给出安全错误。

DashScope 配置：

```dotenv
LLM_PROVIDER=dashscope
DASHSCOPE_API_KEY=在本地填写
RAG_LLM_MODEL=qwen-turbo
```

OpenLux 使用 OpenAI-compatible Chat Completions。模型权限因账户而异，应先在
OpenLux 控制台或通过认证后的 `GET /v1/models` 确认实际模型 ID，不要猜测：

```dotenv
LLM_PROVIDER=openai_compatible
LLM_BASE_URL=https://api.openlux.ai/v1
LLM_API_KEY=在本地填写
LLM_MODEL=账户实际可用的模型ID
LLM_TIMEOUT_SECONDS=30
```

```bash
multiformat-rag ask "产品支持哪些文档格式？" \
  --persist-dir ./.rag_index
```

回答之后会列出来源文件、页码或幻灯片号、CSV 行号、分块编号和相似度。

## 测试

```bash
python -m pytest
```

测试运行时在 pytest 临时目录中生成无品牌、无个人隐私的 TXT、CSV、PDF 和 PPTX 数据，覆盖：

- 四种格式解析
- Unicode、控制字符和段落清洗
- 分块边界与非法参数
- 元数据传递
- 索引构建、持久化和重新加载
- 检索来源追溯
- 空文档过滤
- Mock LLM 问答与无依据回退
- 显式覆盖建库不残留旧节点
- 检索文档不能伪造 Prompt 结构标签

### 本机真实 Embedding 验证

2026-10-18 在 Ubuntu 22.04、Python 3.10.12、CPU 环境完成了独立验证。测试使用
脚本生成的三份非敏感中文 TXT，不读取仓库外的业务文档：

```bash
python scripts/validate_real_embedding.py \
  --output-dir .validation/real-embedding
```

验证环境与结果：

- 模型：`BAAI/bge-small-zh-v1.5`
- PyTorch：`2.14.1+cpu`，CUDA 不可用且未安装 CUDA wheel
- 关键依赖：`llama-index-core 0.14.25`、
  `llama-index-embeddings-huggingface 0.8.0`、
  `sentence-transformers 6.1.0`、`transformers 5.20.1`
- 安装前可用磁盘约 252 GiB、可用内存约 20 GiB且无 swap；安装完成后的整个
  `.venv` 约 1.4 GiB
- 向量维度：512
- 文档 / 分块：3 / 3
- 首次下载、建库、持久化、同进程重载与检索总耗时：23.30 秒
- 进程峰值常驻内存：约 632 MiB
- 独立进程在 `HF_HUB_OFFLINE=1` 下重新加载索引并检索：4.75 秒，峰值约 590 MiB
- 相关文档在 Top-3 中排名第 1，得分 `0.5963`；另两项得分分别为
  `0.2514`、`0.2460`
- 使用阈值 `0.35` 独立检索时仅保留正确文档，得分 `0.5481`
- 返回结果包含 `source_file`、`file_type`、`source_id`、`chunk_index`

两次首条得分不同不是索引重载漂移：建库验证使用的问题是“怎样让语义检索结果
能够追踪到原始文档的位置？”，独立进程重载验证使用的是“如何追踪检索结果对应的
原始文件？”。两次使用相同模型、相同持久化索引和相同归一化配置，但查询文本不同，
因此查询向量和余弦相似度分别为 `0.5963` 与 `0.5481`；两次首条结果均为同一正确文档。

这些数值是本机一次小样本验证记录，不是性能基准，不保证其他硬件、依赖版本或
文档集得到相同结果。`.validation/` 已加入 `.gitignore`，不会作为项目素材发布。

### DashScope 验证状态

DashScope 适配器、默认 `qwen-turbo` 配置、环境变量读取、缺少 Key 的安全报错和
Mock LLM 问答均已验证。验收环境中没有 `DASHSCOPE_API_KEY`，因此没有发起真实
LLM 请求，也没有验证账户权限、额度、真实回答质量或真实拒答表现。配置合法 Key
后应使用非敏感合成文档分别测试有依据问题和无依据问题，再记录结果；不得把 Key
或完整请求日志提交到仓库。

### OpenLux 验证状态

项目已实现并用本地 Mock HTTP 传输验证 OpenAI-compatible Chat Completions
适配器，包括自定义 Base URL/模型、Bearer 认证、超时、鉴权失败、模型或端点不可用、
限流、服务错误、畸形响应和空回答处理。2026-10-26 在明确授权后，使用
`gemini-3.1-flash-lite`、既有真实 BGE 索引和四格式合成文档执行了 6 次 OpenLux
真实请求；A–F 六个场景全部通过，覆盖多来源综合、无依据拒答、相似但无关内容、
Prompt Injection 和引用映射。该结果仅代表这一次受控验证，不代表 DashScope 或
未来路由结果。Key 未写入报告或仓库。详细状态见
[docs/V0.2_REAL_LLM_VALIDATION.md](docs/V0.2_REAL_LLM_VALIDATION.md)。

### GitHub Actions

`.github/workflows/tests.yml` 在 Ubuntu 和 Python 3.10 上仅安装核心依赖与测试依赖，
设置 `HF_HUB_OFFLINE=1`，并把两类 LLM Key 置空后执行 Mock 测试。工作流不安装
本地 Embedding 可选依赖、不下载模型，也不调用任何收费接口。

## 项目限制

- 当前 PDF 解析只读取文本层，不包含 OCR；扫描 PDF 需要额外 OCR 方案。
- PPTX 读取文本框和表格，不处理图片、图表或复杂版式语义。
- CSV 将第一行视为表头，不适用于无表头文件。
- 本地 SimpleVectorStore 适合演示和中小规模数据，不适合大规模并发服务。
- 当前是单轮问答，不保存多轮会话记忆。
- 默认阈值需要针对实际模型和文档集评估后调整。
- Mock 自动化测试不证明真实账户状态；V0.2 的单次 OpenLux 验证也不保证未来网络、
  模型路由、权限、额度或输出保持不变。真实 DashScope 仍未验证。
- 当前没有对 LLM 输出中的引用编号做二次事实校验；来源列表表示送入模型的上下文，
  不代表模型一定正确使用了每条来源。
- Prompt 对检索文本和问题使用不可信输入声明与标签转义，可降低直接提示注入风险，
  但任何基于提示词的防护都不能彻底消除间接提示注入。不要索引不可信的机密文档。
- 调用 DashScope 或 OpenAI-compatible provider 时，检索到的文本会发送给对应
  第三方服务。文件名、页码等来源元数据也会进入 Prompt；涉及隐私、商业秘密或
  受限材料时应停止使用或先做脱敏。

## 安全与公开发布

- 提交前运行凭据扫描，并确认 `.env`、本地索引和模型目录没有被 Git 跟踪。
- 仅使用自己创建或拥有明确分发许可的示例文档。
- 调用外部 LLM 前确认文档允许发送给对应服务商。
- 当前仓库不打包模型权重或第三方源码。依赖许可摘要见
  [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)，发布二进制包前应再次核对锁定版本。

## License

项目代码采用 MIT License。版权行使用通用的 contributors 标识，不虚构个人作者；
发布前可由仓库所有者补充自己的姓名或 GitHub 用户名。
