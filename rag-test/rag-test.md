# rag-test 学习笔记

## 概述

本模块是一个完整的 **RAG（Retrieval-Augmented Generation，检索增强生成）** 技术学习项目，基于 LangChain 生态系统构建。涵盖从文档加载、文本分割、向量化存储、相似度检索到最终 LLM 问答的完整 RAG 流程，同时深入探讨了多种文本分割策略的实现与对比。

核心技术栈：`LangChain` + `OpenAI Embeddings` + `MemoryVectorStore` + `js-tiktoken` + `Cheerio`

---

## 技术栈

| 库 | 版本 | 用途 |
|---|---|---|
| `@langchain/openai` | ^1.2.0 | OpenAI 模型与嵌入向量接口 |
| `@langchain/classic` | ^1.0.7 | 内存向量存储 `MemoryVectorStore` |
| `@langchain/community` | ^1.1.1 | 社区加载器，如 `CheerioWebBaseLoader` |
| `@langchain/core` | ^1.1.8 | 核心抽象，`Document`、`Runnable` 等 |
| `@langchain/textsplitters` | ^1.0.1 | 文本分割器集合 |
| `cheerio` | ^1.1.2 | HTML 解析（供网页加载器使用） |
| `js-tiktoken` | ^1.0.21 | Token 计数工具，与 OpenAI 编码对齐 |
| `dotenv` | ^17.2.3 | 环境变量管理 |

---

## 文件结构

```
rag-test/
├── package.json                         # 项目依赖配置
├── src/
│   ├── hello-rag.mjs                    # 完整 RAG 流程演示（手动构建文档）
│   ├── loader-and-splitter.mjs          # 网页加载 + 文本分割基础演示
│   ├── loader-and-splitter2.mjs         # 网页加载 + 分割 + 向量检索完整流程
│   ├── tiktoken-test.mjs                # Token 计数工具对比测试
│   └── splitters/
│       ├── CharacterTextSplitter-test.mjs          # 字符分割器测试
│       ├── RecursiveCharacterTextSplitter-test.mjs # 递归字符分割器测试
│       ├── TokenTextSplitter-test.mjs              # Token 分割器测试
│       ├── recursive-splitter-code.mjs             # 针对代码的递归分割
│       ├── recursive-splitter-latex.mjs            # 针对 LaTeX 公式的分割
│       └── recursive-splitter-markdown.mjs        # 针对 Markdown 文档的分割
```

---

## 核心知识点

### 知识点 1：RAG（检索增强生成）基本架构

- **概念解释**：RAG 是一种在 LLM 回答问题前，先从知识库中检索相关文档片段，再将其作为上下文注入到 Prompt 中，从而让模型基于真实数据回答的技术。它有效解决了 LLM 知识截止和幻觉问题。

- **数据流**：
  ```
  文本/网页 → Document Loader → 文本分割 → Embedding 向量化 → 向量数据库
                                                                    ↓
  用户问题 → Embedding → 相似度检索 → 召回文档 → 构建 Prompt → LLM → 回答
  ```

- **代码示例**（来自 `hello-rag.mjs`）：
  ```js
  // 1. 创建向量存储（存入所有文档的向量）
  const vectorStore = await MemoryVectorStore.fromDocuments(documents, embeddings);

  // 2. 创建检索器，每次返回 top-3
  const retriever = vectorStore.asRetriever({ k: 3 });

  // 3. 检索相关文档
  const retrievedDocs = await retriever.invoke(question);

  // 4. 构建上下文 + Prompt，调用 LLM
  const context = retrievedDocs.map((doc, i) => `[片段${i + 1}]\n${doc.pageContent}`).join("\n\n━━━━━\n\n");
  const response = await model.invoke(prompt);
  ```

- **注意事项**：`retriever.invoke()` 和 `vectorStore.similaritySearchWithScore()` 返回的文档排序可能不完全对应，建议统一使用 `similaritySearchWithScore` 同时获取文档和评分。

---

### 知识点 2：Document 对象与 Metadata

- **概念解释**：LangChain 中所有文本都被包装为 `Document` 对象，包含 `pageContent`（正文）和 `metadata`（元数据）两部分。元数据用于过滤、追溯、展示来源等。

- **代码示例**（来自 `hello-rag.mjs`）：
  ```js
  import { Document } from "@langchain/core/documents";

  const doc = new Document({
    pageContent: `光光是一个活泼开朗的小男孩...`,
    metadata: { 
      chapter: 1, 
      character: "光光", 
      type: "角色介绍", 
      mood: "活泼" 
    },
  });
  ```

- **注意事项**：`metadata` 是自由格式的对象，可以存储任意信息，但不参与向量检索，只用于结果过滤和展示。

---

### 知识点 3：OpenAI Embeddings（嵌入向量）

- **概念解释**：Embedding 是将文本转换为高维数值向量的技术，语义相近的文本在向量空间中距离更近。RAG 的相似度检索就是基于向量余弦距离实现的。

- **代码示例**（来自 `hello-rag.mjs`）：
  ```js
  import { OpenAIEmbeddings } from "@langchain/openai";

  const embeddings = new OpenAIEmbeddings({
    apiKey: process.env.OPENAI_API_KEY,
    model: process.env.EMBEDDINGS_MODEL_NAME,  // 如 "text-embedding-3-small"
    configuration: {
      baseURL: process.env.OPENAI_BASE_URL  // 支持自定义代理地址
    },
  });
  ```

- **注意事项**：Embedding 模型和 Chat 模型是两个独立的模型，不能混用。`EMBEDDINGS_MODEL_NAME` 在 `.env` 中单独配置。

---

### 知识点 4：MemoryVectorStore（内存向量存储）

- **概念解释**：`MemoryVectorStore` 是 LangChain 提供的最简单的向量存储，将所有向量保存在内存中，无需额外的数据库服务，适合学习和小规模测试。生产环境通常换为 Milvus、Pinecone、Elasticsearch 等。

- **代码示例**（来自 `loader-and-splitter2.mjs`）：
  ```js
  import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";

  // 批量将文档向量化并存入内存
  const vectorStore = await MemoryVectorStore.fromDocuments(splitDocuments, embeddings);

  // 带相似度分数的检索（distance 越小越相似，similarity = 1 - distance）
  const scoredResults = await vectorStore.similaritySearchWithScore(question, 2);
  scoredResults.forEach(([doc, score]) => {
    const similarity = (1 - score).toFixed(4);
    console.log(`相似度: ${similarity}`);
  });
  ```

- **注意事项**：`similaritySearchWithScore` 返回的 `score` 是**距离值**（越小越相似），而非相似度，计算相似度需用 `1 - score`。

---

### 知识点 5：CheerioWebBaseLoader（网页文档加载器）

- **概念解释**：`CheerioWebBaseLoader` 是基于 `cheerio`（服务端 jQuery）实现的网页内容加载器，可以抓取指定 URL 页面并用 CSS 选择器提取正文内容，转为 `Document` 对象。

- **代码示例**（来自 `loader-and-splitter.mjs`）：
  ```js
  import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";

  const cheerioLoader = new CheerioWebBaseLoader(
    "https://juejin.cn/post/7233327509919547452",
    {
      selector: '.main-area p'  // 只提取 .main-area 下的 <p> 标签文本
    }
  );

  const documents = await cheerioLoader.load();
  // documents[0].pageContent 是提取到的所有段落文本拼接
  ```

- **注意事项**：`cheerio` 需要单独 `import "cheerio"` 导入以确保 peer dependency 正确加载。`selector` 选项让你精准控制抓取范围，避免导航、页脚等噪声内容。

---

### 知识点 6：CharacterTextSplitter（字符文本分割器）

- **概念解释**：最基础的文本分割器，按指定的**单个分隔符**切分文本。当超过 `chunkSize` 时才进行分割，切割点优先在分隔符处。

- **代码示例**（来自 `splitters/CharacterTextSplitter-test.mjs`）：
  ```js
  import { CharacterTextSplitter } from "@langchain/textsplitters";

  const logTextSplitter = new CharacterTextSplitter({
    separator: '\n',       // 单个分隔符（换行符）
    chunkSize: 200,        // 每块最大字符数
    chunkOverlap: 20       // 块间重叠字符数，保留上下文连续性
  });

  const splitDocuments = await logTextSplitter.splitDocuments([logDocument]);
  ```

- **注意事项**：如果一行内容超过 `chunkSize`，该行不会被二次切分（这是与 `RecursiveCharacterTextSplitter` 的关键区别）。块间重叠 `chunkOverlap` 可防止关键信息被切断在块边界。

---

### 知识点 7：RecursiveCharacterTextSplitter（递归字符分割器）

- **概念解释**：最常用的文本分割器，按**优先级列表**递归尝试多个分隔符。先尝试第一个分隔符（如 `\n`），若分割后仍超过 `chunkSize`，再用第二个（如 `。`），以此类推，保证分块尽量在语义边界处切割。

- **代码示例**（来自 `splitters/RecursiveCharacterTextSplitter-test.mjs`）：
  ```js
  import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

  const logTextSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 150,
    chunkOverlap: 20,
    separators: ['\n', '。', '，'],   // 按优先级递归尝试
    // lengthFunction: (text) => enc.encode(text).length,  // 可自定义长度计算（Token 数）
  });
  ```

- **注意事项**：`lengthFunction` 参数可以传入 tiktoken 的 Token 计数函数，使切割基于 Token 数而非字符数，更精确控制输入给 LLM 的内容量。

---

### 知识点 8：TokenTextSplitter（Token 分割器）

- **概念解释**：直接按照 Token 数量切分文本，而非字符数。因为 LLM 的上下文窗口是以 Token 计算的，Token 分割器可以最精确地控制每个分块的大小。

- **代码示例**（来自 `splitters/TokenTextSplitter-test.mjs`）：
  ```js
  import { TokenTextSplitter } from "@langchain/textsplitters";

  const logTextSplitter = new TokenTextSplitter({
    chunkSize: 50,                   // 每个块最多 50 个 Token
    chunkOverlap: 10,                // 块之间重叠 10 个 Token
    encodingName: 'cl100k_base',     // OpenAI GPT-4/GPT-3.5 使用的编码
  });
  ```

- **注意事项**：中文字符通常 1 个字符对应 2～3 个 Token（见 `tiktoken-test.mjs`），因此同样的 `chunkSize` 值，中文文本的实际字符数约为英文的 1/3。

---

### 知识点 9：语言感知的代码分割器

- **概念解释**：`RecursiveCharacterTextSplitter.fromLanguage()` 工厂方法可以创建针对特定编程语言的分割器，它内置了该语言的语法分隔符（如函数边界、类定义等），使切割点更符合代码结构。

- **代码示例**（来自 `splitters/recursive-splitter-code.mjs`）：
  ```js
  import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

  // 使用语言感知的工厂方法，自动选择 JS 语法分隔符
  const codeSplitter = RecursiveCharacterTextSplitter.fromLanguage('js', {
    chunkSize: 300,
    chunkOverlap: 60,
  });

  const splitDocuments = await codeSplitter.splitDocuments([jsCodeDoc]);
  ```

- **注意事项**：支持的语言包括 `js`、`ts`、`python`、`java`、`cpp`、`go` 等。代码分割时 `chunkOverlap` 建议设大一些，避免切断函数定义与调用。

---

### 知识点 10：MarkdownTextSplitter 和 LatexTextSplitter

- **概念解释**：专门针对 Markdown 和 LaTeX 文档结构设计的分割器，能识别标题层级（`#`、`##` 等）和 LaTeX 命令边界，在语义单元处切割，而不是盲目按字符数切断。

- **代码示例**（来自 `splitters/recursive-splitter-markdown.mjs`）：
  ```js
  import { MarkdownTextSplitter } from "@langchain/textsplitters";

  const markdownTextSplitter = new MarkdownTextSplitter({
    chunkSize: 400,
    chunkOverlap: 80
  });
  ```

- **代码示例**（来自 `splitters/recursive-splitter-latex.mjs`）：
  ```js
  import { LatexTextSplitter } from "@langchain/textsplitters";

  const markdownTextSplitter = new LatexTextSplitter({
    chunkSize: 200,
    chunkOverlap: 40
  });
  ```

- **注意事项**：这两个分割器本质上都是 `RecursiveCharacterTextSplitter` 的预设版本，只是分隔符列表不同，针对对应格式做了优化。

---

### 知识点 11：js-tiktoken Token 计数

- **概念解释**：`js-tiktoken` 是 OpenAI 官方 tiktoken 的 JS 实现，用于在不调用 API 的情况下，本地计算文本的 Token 数量。这对估算 API 费用、控制上下文长度非常重要。

- **代码示例**（来自 `tiktoken-test.mjs`）：
  ```js
  import { getEncoding, getEncodingNameForModel } from "js-tiktoken"; 

  // 根据模型名获取编码方案名称
  const encodingName = getEncodingNameForModel("gpt-4");  // → "cl100k_base"

  const enc = getEncoding("cl100k_base");
  console.log('apple', enc.encode("apple").length);      // 1 token
  console.log('pineapple', enc.encode("pineapple").length); // 1 token
  console.log('苹果', enc.encode("苹果").length);          // 2 tokens
  console.log('吃饭', enc.encode("吃饭").length);          // 2 tokens
  console.log('一二三', enc.encode("一二三").length);       // 3 tokens
  ```

- **注意事项**：`cl100k_base` 是 GPT-3.5/GPT-4 使用的编码，中文通常每个字符 1～2 个 Token，英文单词通常 1 个 Token（常见词）或多个 Token（长词）。

---

## 代码解析

### `hello-rag.mjs` — 手动构建文档的完整 RAG 演示

这是最基础的 RAG 示例，使用手动创建的文档而非从外部加载：

```js
// Step 1: 手动创建带 metadata 的文档
const documents = [
  new Document({ pageContent: "...", metadata: { chapter: 1, character: "光光" } }),
  // ...共 7 个文档
];

// Step 2: 向量化存储
const vectorStore = await MemoryVectorStore.fromDocuments(documents, embeddings);

// Step 3: 检索（k=3 表示返回最相似的 3 个文档）
const retriever = vectorStore.asRetriever({ k: 3 });
const retrievedDocs = await retriever.invoke(question);

// Step 4: 同时获取相似度评分
const scoredResults = await vectorStore.similaritySearchWithScore(question, 3);

// Step 5: 将检索结果组装进 Prompt，调用 LLM
const context = retrievedDocs.map((doc, i) => `[片段${i+1}]\n${doc.pageContent}`).join("\n\n━━━━━\n\n");
const response = await model.invoke(prompt);
```

---

### `loader-and-splitter.mjs` — 网页加载与文本分割基础

展示了如何加载真实网页并分割：

```js
// 用 CSS 选择器精准提取页面主要内容
const cheerioLoader = new CheerioWebBaseLoader(url, { selector: '.main-area p' });
const documents = await cheerioLoader.load();

// 递归分割：先按句号，再按其他符号
const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 400,
  chunkOverlap: 50,
  separators: ["。","！","？"],
});
const splitDocuments = await textSplitter.splitDocuments(documents);
```

---

### `loader-and-splitter2.mjs` — 完整网页 RAG 流程

在 `loader-and-splitter.mjs` 基础上加入了向量存储和问答：

```js
// 加载 → 分割 → 向量化
const documents = await cheerioLoader.load();
const splitDocuments = await textSplitter.splitDocuments(documents);
const vectorStore = await MemoryVectorStore.fromDocuments(splitDocuments, embeddings);

// 检索 + 评分（合并为一次调用，更高效）
const scoredResults = await vectorStore.similaritySearchWithScore(question, 2);
const retrievedDocs = scoredResults.map(([doc]) => doc);
```

关键优化：用 `similaritySearchWithScore` 一次调用同时拿到文档和评分，替代了 `hello-rag.mjs` 中的两次调用。

---

### `splitters/` 目录 — 分割器对比测试

该目录集中测试了各种分割器，统一用如下方式输出结果以便对比：

```js
splitDocuments.forEach(document => {
  console.log(document);
  console.log('charater length:', document.pageContent.length);  // 字符数
  console.log('token length:', enc.encode(document.pageContent).length);  // Token 数
});
```

这种对比输出让你直观看到同一段文本按字符数和 Token 数分割的区别。

---

## 知识点总结表

| 知识点 | 说明 | 所在文件 |
|--------|------|----------|
| RAG 完整流程 | 加载→分割→向量化→检索→问答 | `hello-rag.mjs`, `loader-and-splitter2.mjs` |
| Document 对象 | LangChain 文档基本单元，含 pageContent 和 metadata | `hello-rag.mjs` |
| OpenAI Embeddings | 文本向量化，用于相似度检索 | `hello-rag.mjs`, `loader-and-splitter2.mjs` |
| MemoryVectorStore | 内存向量数据库，适合学习测试 | `hello-rag.mjs`, `loader-and-splitter2.mjs` |
| similaritySearchWithScore | 向量检索 + 相似度分数，score 是距离值 | `hello-rag.mjs`, `loader-and-splitter2.mjs` |
| CheerioWebBaseLoader | 网页内容加载器，支持 CSS 选择器 | `loader-and-splitter.mjs`, `loader-and-splitter2.mjs` |
| CharacterTextSplitter | 单分隔符字符切割，不会二次切分超长块 | `splitters/CharacterTextSplitter-test.mjs` |
| RecursiveCharacterTextSplitter | 递归多分隔符切割，最常用 | `loader-and-splitter.mjs`, `splitters/RecursiveCharacterTextSplitter-test.mjs` |
| TokenTextSplitter | 按 Token 数切割，最精确控制 LLM 输入量 | `splitters/TokenTextSplitter-test.mjs` |
| fromLanguage() 工厂方法 | 语言感知代码分割，按代码结构切割 | `splitters/recursive-splitter-code.mjs` |
| MarkdownTextSplitter | 按 Markdown 标题层级切割 | `splitters/recursive-splitter-markdown.mjs` |
| LatexTextSplitter | 按 LaTeX 语法结构切割 | `splitters/recursive-splitter-latex.mjs` |
| js-tiktoken | 本地 Token 计数，中文每字约 1～2 Token | `tiktoken-test.mjs` |
| chunkOverlap | 块间重叠，保留跨块的上下文连续性 | 所有分割器示例 |

---

## 扩展学习

1. **向量数据库进阶**：学习 `Milvus`、`Pinecone`、`Qdrant` 等生产级向量数据库，替换 `MemoryVectorStore`（参考项目中的 `milvus-test` 目录）

2. **文档加载器扩展**：探索 `PDFLoader`、`DocxLoader`、`CSVLoader`、`GitHubLoader` 等更多加载器场景

3. **Reranking（重排序）**：在初次向量检索后加入 Cohere Rerank 等精排模型，提升检索质量（参考项目中的 `es-test/src/rerank` 目录）

4. **Metadata Filtering（元数据过滤）**：在向量检索时加入 `filter` 条件，先按元数据筛选再做向量搜索

5. **高级 RAG 策略**：了解 HyDE（假设文档嵌入）、多查询检索、Parent-Child Chunking 等高级 RAG 技巧

6. **Token 费用估算**：结合 `js-tiktoken` 和 OpenAI 定价表，在调用前预估 API 费用

7. **LangSmith 追踪**：为 RAG 流程添加 LangSmith 追踪，可视化每个检索步骤的结果（参考项目中的 `langsmith-test` 目录）
