# AI Agent Course Code

这是一个围绕 AI Agent、LLM 应用开发与相关工程实践的示例代码仓库。

仓库内包含多个相对独立的实验项目，覆盖以下主题：

- LangChain 基础用法
- Prompt 模板与 Runnable
- Output Parser 与结构化输出
- Tool Calling 与 Agent 工具调用
- Memory 与对话记忆管理
- RAG、向量检索与知识库相关实验
- LangGraph、LangSmith 等 Agent 生态工具
- NestJS / 前后端集成类示例
- Redis、PostgreSQL、Milvus、Elasticsearch、Neo4j 等配套存储实验

## 仓库结构

每个目录通常都是一个单独的小项目，可以独立安装依赖、单独运行。

部分目录示例：

- `prompt-template-test/`：Prompt 模板相关示例
- `output-parser-test/`：结构化输出、JSON 解析、Tool Call 解析示例
- `tool-test/`：工具调用相关示例
- `memory-test/`：对话记忆、总结记忆、检索记忆示例
- `rag-test/`：文档加载、切分与 RAG 基础实验
- `langgraph-test/`：LangGraph 相关示例
- `langsmith-test/`：LangSmith 评测与追踪相关示例
- `deep-research-assistant/`：研究型 Agent 示例
- `agui-backend/`、`agui-frontend/`：前后端配套示例
- `nest-feature/`、`hello-nest-langchain/`：NestJS 结合 LLM 的示例项目
- `redis-test/`、`pgsql-test/`、`milvus-test/`、`neo4j-graphrag/`：各类存储与检索实验

## 使用方式

### 1. 进入具体项目目录

```bash
cd output-parser-test
```

### 2. 安装依赖

仓库中的项目大多是独立管理的，请在对应目录内安装依赖：

```bash
pnpm install
```

如果某个子项目使用 `npm`，也可以按其目录下的说明执行。

### 3. 配置环境变量

多数 LLM 相关示例会使用以下环境变量：

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `MODEL_NAME`

如项目需要额外配置，请查看对应目录下的代码或说明文件。

### 4. 运行示例

不同子项目的运行方式不同，通常直接执行对应脚本文件，例如：

```bash
node src/xxx.mjs
```

或使用项目内定义的脚本命令。

## 说明

- 本仓库以实验和学习代码为主，很多目录彼此独立，不一定共用同一套工程规范。
- 某些示例依赖本地基础设施，例如 MySQL、Redis、Milvus、Elasticsearch、Neo4j 或 PostgreSQL，请先确保相关服务可用。
- 部分目录下已经包含单独的 `README.md` 或学习笔记，可优先查看对应文档。

## 建议阅读顺序

如果你是第一次看这个仓库，可以按下面顺序阅读：

1. `prompt-template-test/`
2. `output-parser-test/`
3. `tool-test/`
4. `memory-test/`
5. `rag-test/`
6. `langgraph-test/`
7. 其他与业务集成或存储相关的目录

## 目标

这个仓库的目标是把 AI Agent 开发中常见的关键能力拆成一个个可运行、可验证的小实验，方便逐步理解和复用。
