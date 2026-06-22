# memory-test 学习笔记

## 概述

`memory-test` 是一个专注于 **LLM 对话记忆管理** 的实验项目，涵盖了 AI 对话系统中"记忆"这一核心问题的多种解决方案。项目演示了从最基础的内存存储，到基于文件持久化、消息截断、总结压缩，以及最高级的向量数据库语义检索等多种记忆策略。

核心技术栈：LangChain.js + OpenAI + Milvus 向量数据库 + js-tiktoken。

---

## 技术栈

| 库 / 工具 | 版本 | 用途 |
|-----------|------|------|
| `@langchain/core` | ^1.1.16 | 核心消息类型、历史记录接口 |
| `@langchain/openai` | ^1.2.3 | ChatOpenAI 模型、Embeddings |
| `@langchain/community` | ^1.1.5 | FileSystemChatMessageHistory（文件持久化） |
| `@langchain/classic` | ^1.0.10 | LangChain 经典 API 兼容层 |
| `@zilliz/milvus2-sdk-node` | ^2.6.9 | Milvus 向量数据库 SDK |
| `js-tiktoken` | ^1.0.21 | OpenAI tokenizer，精确 token 计数 |
| `dotenv` | ^17.2.3 | 环境变量加载 |

---

## 文件结构

```
memory-test/
├── package.json                     # 项目依赖配置
├── chat_history.json                # FileSystem 历史记录持久化文件
├── src/
│   ├── history-test.mjs             # 策略一：InMemory 内存历史记录
│   ├── history-test2.mjs            # 策略二：FileSystem 文件持久化历史记录（写入）
│   ├── history-test3.mjs            # 策略二续：FileSystem 文件持久化历史记录（恢复）
│   └── memory/
│       ├── insert-conversations.mjs # 向量数据库准备：插入历史对话到 Milvus
│       ├── retrieval-memory.mjs     # 策略三：Retrieval 向量检索记忆（RAG 记忆）
│       ├── summarization-memory.mjs # 策略四：按消息数量触发总结压缩
│       └── summarization-memory2.mjs# 策略四变体：按 token 数量触发总结压缩
│       └── truncation-memory.mjs    # 策略五：截断策略（按数量 + 按 token 双模式）
```

---

## 核心知识点

### 知识点 1：InMemoryChatMessageHistory —— 内存对话历史

- **概念解释**：最简单的对话历史存储方案，所有消息保存在 Node.js 进程内存中。进程重启后数据消失，适合开发调试或单次会话场景。
- **核心 API**：
  - `new InMemoryChatMessageHistory()` — 创建内存历史记录实例
  - `history.addMessage(message)` — 追加一条消息
  - `history.getMessages()` — 获取所有历史消息（返回 Promise）
  - `history.clear()` — 清空所有历史消息

- **代码示例**（来自 `history-test.mjs`）：

```javascript
const history = new InMemoryChatMessageHistory();
const systemMessage = new SystemMessage("你是一个友好的做菜助手。");

// 第一轮对话
await history.addMessage(new HumanMessage("你今天吃的什么？"));
const messages1 = [systemMessage, ...(await history.getMessages())];
const response1 = await model.invoke(messages1);
await history.addMessage(response1);

// 第二轮对话（自动携带上轮历史）
await history.addMessage(new HumanMessage("好吃吗？"));
const messages2 = [systemMessage, ...(await history.getMessages())];
const response2 = await model.invoke(messages2);
```

- **注意事项**：
  - `SystemMessage` **不存入** history，每次 invoke 手动拼到消息数组最前面
  - `getMessages()` 是异步的，必须 `await`
  - 每次调用模型前需将 systemMessage 与历史消息重新组合

---

### 知识点 2：FileSystemChatMessageHistory —— 文件持久化历史记录

- **概念解释**：来自 `@langchain/community` 的实现，将对话历史序列化为 JSON 文件保存到磁盘。程序重启后可以读取文件恢复历史，支持多 session 隔离。
- **核心参数**：
  - `filePath`：JSON 文件路径
  - `sessionId`：会话 ID，同一文件可存储多个 session 的历史

- **代码示例**（来自 `history-test2.mjs` & `history-test3.mjs`）：

```javascript
// 写入历史（history-test2.mjs）
const filePath = path.join(process.cwd(), "chat_history.json");
const history = new FileSystemChatMessageHistory({
  filePath: filePath,
  sessionId: "user_session_001",
});
await history.addMessage(new HumanMessage("红烧肉怎么做"));

// 恢复历史（history-test3.mjs —— 新进程中恢复）
const restoredHistory = new FileSystemChatMessageHistory({
  filePath: filePath,
  sessionId: "user_session_001",
});
const restoredMessages = await restoredHistory.getMessages();
console.log(`从文件恢复了 ${restoredMessages.length} 条历史消息`);
```

- **注意事项**：
  - 需要 `import path from "node:path"` 处理跨平台路径
  - 同一 `filePath` + 不同 `sessionId` 可实现多用户隔离
  - `history-test2.mjs` 负责写入，`history-test3.mjs` 在新进程中恢复并继续对话，两者配合演示完整持久化流程

---

### 知识点 3：消息截断（Truncation）—— 按数量和按 Token 两种方式

- **概念解释**：当对话历史过长时，直接截断并只保留最近 N 条（或最近 M 个 Token）的消息，是最简单的上下文窗口管理方式。
- **方式一：按消息数量截断**（手动 `slice`）

```javascript
// truncation-memory.mjs
const maxMessages = 4;
const trimmedMessages = allMessages.slice(-maxMessages); // 保留最后 4 条
```

- **方式二：按 Token 数量截断**（使用 `trimMessages` API + `js-tiktoken`）

```javascript
// truncation-memory.mjs
import { trimMessages } from "@langchain/core/messages";
import { getEncoding } from "js-tiktoken";

const enc = getEncoding("cl100k_base");  // GPT-4/GPT-3.5 使用的编码

function countTokens(messages, encoder) {
  let total = 0;
  for (const msg of messages) {
    const content = typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content);
    total += encoder.encode(content).length;
  }
  return total;
}

const trimmedMessages = await trimMessages(allMessages, {
  maxTokens: 100,
  tokenCounter: async (msgs) => countTokens(msgs, enc),
  strategy: "last",  // 保留最近的消息
});
```

- **`js-tiktoken` 说明**：
  - `getEncoding("cl100k_base")` 对应 GPT-4 / GPT-3.5-turbo / text-embedding-ada-002 系列使用的 tokenizer
  - `encoder.encode(text)` 返回 token ID 数组，`.length` 即 token 数
  - 相比直接用字符数，token 计数更精准，能精确控制 LLM 上下文成本

- **注意事项**：
  - `trimMessages` 的 `strategy: "last"` 表示优先保留最新消息，`"first"` 表示保留最早消息
  - `tokenCounter` 接收消息数组，必须返回 `Promise<number>`

---

### 知识点 4：总结压缩记忆（Summarization Memory）

- **概念解释**：当历史消息超过阈值时，不是简单丢弃旧消息，而是先用 LLM 对旧消息做摘要，再用摘要替代原始长对话，实现"有损但保留语义"的压缩。
- **`getBufferString` API**：将消息数组格式化为可读字符串，供总结 prompt 使用。

```javascript
// summarization-memory.mjs
import { getBufferString } from "@langchain/core/messages";

const conversationText = getBufferString(messagesToSummarize, {
  humanPrefix: "用户",
  aiPrefix: "助手",
});
// 输出示例:
// 用户: 我想学做红烧肉，你能教我吗？
// 助手: 当然可以！红烧肉是一道经典的中式菜肴...
```

- **完整总结流程**（来自 `summarization-memory.mjs`）：

```javascript
const maxMessages = 6;
const keepRecent = 2;

if (allMessages.length >= maxMessages) {
  const recentMessages = allMessages.slice(-keepRecent);        // 保留最近 2 条
  const messagesToSummarize = allMessages.slice(0, -keepRecent); // 其余旧消息准备总结

  // 调用 LLM 生成摘要
  const summary = await summarizeHistory(messagesToSummarize);

  // 清空历史，只恢复保留的消息
  await history.clear();
  for (const msg of recentMessages) {
    await history.addMessage(msg);
  }
  // summary 可以作为 SystemMessage 注入到下轮对话
}
```

- **变体：基于 Token 数量触发**（`summarization-memory2.mjs`）：

```javascript
const maxTokens = 200;
const keepRecentTokens = 80;

// 从后往前累加消息，直到达到 keepRecentTokens 限制
const recentMessages = [];
let recentTokens = 0;

for (let i = allMessages.length - 1; i >= 0; i--) {
  const msgTokens = enc.encode(allMessages[i].content).length;
  if (recentTokens + msgTokens <= keepRecentTokens) {
    recentMessages.unshift(allMessages[i]);
    recentTokens += msgTokens;
  } else {
    break;
  }
}
```

- **注意事项**：
  - 总结本身也消耗 LLM 调用，需权衡成本与记忆质量
  - 总结结果通常以 `SystemMessage` 形式注入，告知模型"之前发生了什么"
  - 按 Token 触发比按条数触发更精准，避免单条超长消息造成误差

---

### 知识点 5：Retrieval 检索式记忆（RAG Memory）

- **概念解释**：最高级的记忆策略。将历史对话向量化后存入 Milvus，每次新对话前根据当前输入语义检索最相关的历史片段，注入到 prompt 中。这是 RAG（检索增强生成）在对话记忆场景的应用。

- **整体数据流**：
  ```
  历史对话文本 → OpenAI Embeddings → 1024维向量 → 存入 Milvus
       ↑
  新用户输入 → Embeddings → 查询向量 → Milvus 相似度搜索 → Top-K 相关历史
       ↓
  相关历史 + 新问题 → 组装 Prompt → LLM → 回答
  ```

- **Milvus Collection 结构**（来自 `insert-conversations.mjs`）：

```javascript
await client.createCollection({
  collection_name: 'conversations',
  fields: [
    { name: 'id',        data_type: DataType.VarChar,     max_length: 50,   is_primary_key: true },
    { name: 'vector',    data_type: DataType.FloatVector, dim: 1024 },
    { name: 'content',   data_type: DataType.VarChar,     max_length: 5000 },
    { name: 'round',     data_type: DataType.Int64 },
    { name: 'timestamp', data_type: DataType.VarChar,     max_length: 100 }
  ]
});
```

- **创建索引**（IVF_FLAT + COSINE 相似度）：

```javascript
await client.createIndex({
  collection_name: COLLECTION_NAME,
  field_name: 'vector',
  index_type: IndexType.IVF_FLAT,
  metric_type: MetricType.COSINE
});
```

- **向量检索**（来自 `retrieval-memory.mjs`）：

```javascript
async function retrieveRelevantConversations(query, k = 2) {
  const queryVector = await embeddings.embedQuery(query); // 生成查询向量

  const searchResult = await client.search({
    collection_name: COLLECTION_NAME,
    vector: queryVector,
    limit: k,  // 返回最相似的 k 条
    metric_type: MetricType.COSINE,
    output_fields: ['id', 'content', 'round', 'timestamp']
  });

  return searchResult.results; // 包含 score（相似度分数）
}
```

- **构建 RAG Prompt**：

```javascript
const relevantHistory = retrievedConversations
  .map((conv, idx) => `[历史对话 ${idx + 1}]\n轮次: ${conv.round}\n${conv.content}`)
  .join('\n\n━━━━━\n\n');

const contextMessages = [
  new HumanMessage(`相关历史对话：\n${relevantHistory}\n\n用户问题: ${input}`)
];
const response = await model.invoke(contextMessages);
```

- **注意事项**：
  - 使用前需先运行 `insert-conversations.mjs` 将初始历史数据写入 Milvus
  - `client.connectPromise` 需要 `await` 确保连接建立
  - `embedQuery` 用于单个文本查询，`embedDocuments` 用于批量文档
  - `COSINE` 相似度分数越接近 1 表示越相关
  - 每轮对话结束后，新对话也会被 embed 并插入 Milvus，实现动态记忆积累

---

### 知识点 6：消息类型体系

- **概念解释**：LangChain 的消息系统由多种类型组成，代表不同的对话角色。

| 类型 | 导入路径 | 用途 |
|------|---------|------|
| `HumanMessage` | `@langchain/core/messages` | 用户输入消息 |
| `AIMessage` | `@langchain/core/messages` | AI/模型输出消息 |
| `SystemMessage` | `@langchain/core/messages` | 系统指令（不存入 history） |

- **代码示例**：

```javascript
import { HumanMessage, AIMessage, SystemMessage } from "@langchain/core/messages";

const systemMsg = new SystemMessage("你是一个烹饪助手");
const humanMsg = new HumanMessage("红烧肉怎么做？");
const aiMsg = new AIMessage("首先需要准备五花肉...");

// 判断消息类型
console.log(msg.type);         // "human" | "ai" | "system"
console.log(msg.constructor.name); // "HumanMessage" | "AIMessage" | "SystemMessage"
```

- **注意事项**：
  - `model.invoke()` 的返回值本身就是一个 `AIMessage`，可以直接 `history.addMessage(response)`
  - `SystemMessage` 通常不存入 history，每次手动放在消息数组的最前面

---

### 知识点 7：OpenAIEmbeddings 文本向量化

- **概念解释**：将文本转换为高维浮点向量，语义相似的文本在向量空间中距离更近，是语义检索的基础。

- **配置方式**：

```javascript
const embeddings = new OpenAIEmbeddings({
  apiKey: process.env.OPENAI_API_KEY,
  model: 'text-embedding-v3',  // 使用 v3 嵌入模型
  configuration: {
    baseURL: process.env.OPENAI_BASE_URL  // 支持代理/自定义端点
  },
  dimensions: 1024  // 指定输出维度（v3 支持压缩维度）
});

// 单条查询
const vector = await embeddings.embedQuery("我叫赵六，是数据科学家");

// 批量文档
const vectors = await embeddings.embedDocuments(["文本1", "文本2"]);
```

- **注意事项**：
  - `dimensions` 参数是 text-embedding-v3 新特性，可以压缩向量维度（1024 比完整的 3072 维更省空间）
  - Milvus 的 `dim` 必须与 `dimensions` 保持一致，否则插入/查询会报错

---

## 代码解析

### `history-test.mjs`
最基础的内存历史记录演示。每轮对话通过 `addMessage` 写入，`getMessages` 读出，再拼接 `SystemMessage` 后调用模型。展示了多轮对话中上下文是如何被自动携带的。

### `history-test2.mjs` + `history-test3.mjs`
两个文件配合演示文件持久化的完整生命周期：
- `history-test2.mjs` 模拟"第一次对话"，将历史写入 `chat_history.json`
- `history-test3.mjs` 模拟"程序重启后续聊"，从文件恢复历史，继续进行第三轮对话

### `truncation-memory.mjs`
同一文件演示两种截断策略，依次运行。
- `messageCountTruncation()`：用 `Array.slice(-N)` 实现，最简单
- `tokenCountTruncation()`：集成 `js-tiktoken` + LangChain `trimMessages` API，精确按 token 截断

### `summarization-memory.mjs`
按消息数量（`maxMessages = 6`）触发总结。将旧消息用 `getBufferString` 格式化后发给 LLM 做摘要，然后清空 history，只保留最近 2 条真实消息。

### `summarization-memory2.mjs`
按 token 数（`maxTokens = 200`）触发总结，增加精度。从数组末尾向前累积 token，确定保留范围，剩余的旧消息送去总结。

### `insert-conversations.mjs`
Milvus 数据准备脚本。依次完成：创建集合 → 创建 IVF_FLAT 索引 → 加载集合 → 批量插入对话数据（含向量）。是 `retrieval-memory.mjs` 的前置步骤。

### `retrieval-memory.mjs`
RAG 记忆的完整实现。每轮对话：① 将用户输入向量化 → ② Milvus 语义检索 Top-2 历史 → ③ 拼装含历史上下文的 Prompt → ④ 调用模型 → ⑤ 将新对话向量化后写回 Milvus（动态积累记忆）。

---

## 知识点总结表

| 知识点 | 说明 | 所在文件 |
|--------|------|----------|
| `InMemoryChatMessageHistory` | 内存对话历史，进程重启后丢失 | `history-test.mjs` |
| `FileSystemChatMessageHistory` | 文件持久化历史，支持多 session | `history-test2.mjs`, `history-test3.mjs` |
| 消息数量截断 | `slice(-N)` 保留最近 N 条消息 | `truncation-memory.mjs` |
| Token 数量截断 | `trimMessages` + `js-tiktoken` 精确截断 | `truncation-memory.mjs` |
| `getEncoding("cl100k_base")` | GPT-4 系列 tokenizer，精确计数 | `truncation-memory.mjs`, `summarization-memory2.mjs` |
| 总结压缩（按条数触发） | 旧消息 → LLM 摘要，节省 context | `summarization-memory.mjs` |
| 总结压缩（按 token 触发） | 更精准的触发条件 | `summarization-memory2.mjs` |
| `getBufferString` | 消息数组 → 可读文本，供摘要 prompt 使用 | `summarization-memory.mjs`, `summarization-memory2.mjs` |
| Milvus 集合创建与索引 | 向量数据库建表，IVF_FLAT + COSINE | `insert-conversations.mjs` |
| `OpenAIEmbeddings.embedQuery` | 文本向量化，支持自定义维度 | `retrieval-memory.mjs`, `insert-conversations.mjs` |
| RAG 记忆（向量语义检索） | 检索最相关历史片段注入 Prompt | `retrieval-memory.mjs` |
| `HumanMessage` / `AIMessage` / `SystemMessage` | LangChain 消息类型体系 | 所有文件 |
| `sessionId` 隔离 | 同文件存多 session 的历史记录 | `history-test2.mjs`, `history-test3.mjs` |

---

## 五种记忆策略对比

| 策略 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **InMemory** | 简单，零依赖 | 重启即丢失 | 开发调试、单次会话 |
| **FileSystem** | 持久化，零服务依赖 | 不适合高并发，文件可能很大 | 单用户桌面应用 |
| **截断（Truncation）** | 成本可控，实现简单 | 直接丢弃旧信息，遗忘早期对话 | 对历史不敏感的场景 |
| **总结（Summarization）** | 保留语义，压缩上下文 | 需要额外 LLM 调用，有摘要失真风险 | 长对话、需保留关键信息 |
| **检索（Retrieval/RAG）** | 按需检索相关历史，最精准 | 需要向量数据库基础设施，复杂度高 | 生产级智能助手、个人知识库 |

---

## 扩展学习

1. **LangGraph 中的记忆管理**：LangGraph 提供了更结构化的状态管理，内置 checkpointer 可以持久化整个 agent 状态
2. **Mem0**：专门为 AI Agent 设计的记忆层，项目中 `mem0-test` 目录有相关探索
3. **pgvector + PostgreSQL**：生产环境中可用 pgvector 替代 Milvus，减少基础设施复杂度
4. **RunnableWithMessageHistory**：LangChain 提供的高层封装，自动管理 sessionId 和历史注入，避免手动拼装消息
5. **Token 预算管理**：生产应用中需要同时考虑 prompt tokens + completion tokens + history tokens 的总预算控制
6. **混合记忆策略**：实际产品中往往将截断 + 总结 + 检索三者结合使用（短期内存 + 压缩中期 + 长期向量检索）
