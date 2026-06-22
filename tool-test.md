# tool-test 学习笔记

## 概述

`tool-test` 是一个 **AI Agent 工具调用（Tool Use）** 的综合实验项目，核心目标是演示如何让 LLM（大语言模型）使用"工具"来完成复杂任务。

项目围绕两个主要方向展开：
1. **LangChain 原生工具**：用 `@langchain/core/tools` 手动定义工具，供 Agent 调用文件系统、执行命令等操作，最终实现一个能自主创建 React 项目的"Mini Cursor"。
2. **MCP（Model Context Protocol）**：使用标准化协议构建 MCP Server，并通过 `@langchain/mcp-adapters` 将 MCP 工具注入 LangChain Agent，实现工具的标准化复用。

---

## 技术栈

| 库/框架 | 版本 | 用途 |
|---------|------|------|
| `@langchain/core` | ^1.1.7 | LangChain 核心包，提供 tool、消息类型等 |
| `@langchain/openai` | ^1.2.0 | OpenAI 兼容接口（支持通义千问等） |
| `@langchain/mcp-adapters` | ^1.1.1 | 将 MCP Server 工具适配为 LangChain 工具 |
| `@modelcontextprotocol/sdk` | ^1.25.1 | MCP 官方 SDK，用于构建 MCP Server |
| `zod` | ^4.2.1 | TypeScript-first 数据校验，用于定义工具参数 Schema |
| `chalk` | ^5.6.2 | 终端彩色输出，用于调试日志美化 |
| `dotenv` | ^17.2.3 | 从 `.env` 文件加载环境变量 |

---

## 文件结构

```
tool-test/
├── package.json               # 项目配置 & 依赖声明
└── src/
    ├── hello-langchain.mjs    # 最简 LangChain 入门示例
    ├── tool-file-read.mjs     # 单工具（文件读取）+ 手动工具循环
    ├── all-tools.mjs          # 四个通用工具的定义与导出（read/write/exec/list）
    ├── mini-cursor.mjs        # 完整 Agent：调用工具自动创建 React 项目
    ├── my-mcp-server.mjs      # 自定义 MCP Server（工具 + 资源）
    ├── langchain-mcp-test.mjs # LangChain + MCP + Resources 集成示例
    ├── mcp-test.mjs           # 多 MCP Server（高德地图/文件系统/Chrome DevTools）
    └── node-exec.mjs          # Node.js child_process.spawn 原生命令执行示例
```

---

## 核心知识点

### 知识点 1：LangChain 工具的定义（`tool()` 函数）

- **概念解释**：LangChain 使用 `tool()` 工厂函数将普通 JS 函数封装成 AI 可调用的工具。每个工具需要提供 `name`、`description`（帮助 LLM 理解何时调用）、`schema`（用 Zod 定义参数类型）三要素。
- **代码示例**（来自 `all-tools.mjs`）：

```javascript
import { tool } from '@langchain/core/tools';
import { z } from 'zod';

const readFileTool = tool(
  async ({ filePath }) => {
    const content = await fs.readFile(filePath, 'utf-8');
    return `文件内容:\n${content}`;
  },
  {
    name: 'read_file',
    description: '读取指定路径的文件内容',
    schema: z.object({
      filePath: z.string().describe('文件路径'),
    }),
  }
);
```

- **注意事项**：
  - `description` 极其重要，LLM 依赖它来决定是否调用该工具，要写得清晰准确。
  - `schema` 中每个字段也要加 `.describe()`，LLM 靠字段描述来填充参数值。
  - 工具函数应返回字符串，LLM 只能理解文本格式的工具结果。

---

### 知识点 2：将工具绑定到模型（`bindTools`）

- **概念解释**：调用 `model.bindTools(tools)` 将工具列表"注入"模型。绑定后，模型的每次响应可能包含 `tool_calls` 字段，告诉外部"我要调用哪个工具、传什么参数"。
- **代码示例**（来自 `mini-cursor.mjs`）：

```javascript
const tools = [readFileTool, writeFileTool, executeCommandTool, listDirectoryTool];
const modelWithTools = model.bindTools(tools);

const response = await modelWithTools.invoke(messages);

// 检查是否有工具调用
if (response.tool_calls && response.tool_calls.length > 0) {
  console.log('需要调用工具:', response.tool_calls.map(t => t.name));
}
```

- **注意事项**：
  - `bindTools` 返回新的模型实例，不修改原始 `model`。
  - 模型本身**不执行**工具，只"声明"调用意图，**实际执行由我们的代码负责**。
  - `temperature: 0` 在工具调用场景中推荐设置，使模型更确定性地选择工具。

---

### 知识点 3：Agent 工具调用循环（ReAct 模式）

- **概念解释**：真正的 Agent 是一个**循环**：LLM 思考 → 决定调用工具 → 执行工具 → 将结果反馈给 LLM → LLM 再次思考……直到 LLM 不再需要工具为止。这就是 **ReAct（Reasoning + Acting）** 模式。
- **代码示例**（来自 `tool-file-read.mjs`，手动实现循环）：

```javascript
let response = await modelWithTools.invoke(messages);
messages.push(response); // 将 AI 回复加入历史

while (response.tool_calls && response.tool_calls.length > 0) {
  // 1. 并发执行所有工具调用
  const toolResults = await Promise.all(
    response.tool_calls.map(async (toolCall) => {
      const tool = tools.find(t => t.name === toolCall.name);
      return await tool.invoke(toolCall.args);
    })
  );

  // 2. 将工具结果以 ToolMessage 存入消息历史
  response.tool_calls.forEach((toolCall, index) => {
    messages.push(new ToolMessage({
      content: toolResults[index],
      tool_call_id: toolCall.id, // 必须与 tool_calls 中的 id 严格对应
    }));
  });

  // 3. 再次调用模型（携带工具结果）
  response = await modelWithTools.invoke(messages);
}

console.log('最终回复:', response.content);
```

- **注意事项**：
  - `ToolMessage` 的 `tool_call_id` 必须与对应 `tool_calls[i].id` 严格一致，否则模型会报错。
  - 建议设置 `maxIterations`（如 30）防止死循环。
  - 使用 `Promise.all` 可以并发执行多个工具调用，提高效率（见 `tool-file-read.mjs`）。

---

### 知识点 4：LangChain 消息类型

- **概念解释**：LangChain 对话历史由多种消息类型组成，每种类型代表不同角色，共同构成 LLM 的完整上下文。

| 消息类型 | 导入来源 | 用途 |
|----------|----------|------|
| `SystemMessage` | `@langchain/core/messages` | 系统提示词，设定 AI 角色和规则 |
| `HumanMessage` | `@langchain/core/messages` | 用户输入的消息 |
| `AIMessage`（即模型响应） | 模型自动返回 | 模型的回复，可能包含 `tool_calls` |
| `ToolMessage` | `@langchain/core/messages` | 工具执行结果，需提供 `tool_call_id` |

- **代码示例**（来自 `langchain-mcp-test.mjs`）：

```javascript
import { HumanMessage, SystemMessage, ToolMessage } from '@langchain/core/messages';

const messages = [
  new SystemMessage(resourceContent),              // 系统指令
  new HumanMessage("查一下用户 002 的信息"),        // 用户问题
];

// 工具执行后追加结果
messages.push(new ToolMessage({
  content: toolResult,
  tool_call_id: toolCall.id,
}));
```

- **注意事项**：
  - 完整消息历史格式：`[System, Human, AI(含tool_calls), Tool, AI, ...]`
  - 每次循环都要把 AI 回复和工具结果 `push` 进 `messages`，LLM 才能看到上下文。

---

### 知识点 5：MCP Server 的构建

- **概念解释**：MCP（Model Context Protocol）是 Anthropic 主导的标准化 AI 工具协议。通过官方 SDK 构建 MCP Server，可以让任何支持 MCP 的客户端（Cursor、Claude Desktop 等）调用你的工具，实现跨客户端复用。MCP Server 可以提供两种内容：**工具（Tools）** 和 **资源（Resources）**。
- **代码示例**（来自 `my-mcp-server.mjs`）：

```javascript
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { z } from 'zod';

const server = new McpServer({ name: 'my-mcp-server', version: '1.0.0' });

// 注册工具
server.registerTool('query_user', {
  description: '查询数据库中的用户信息',
  inputSchema: {
    userId: z.string().describe('用户 ID，例如: 001, 002, 003'),
  },
}, async ({ userId }) => {
  const user = database.users[userId];
  return {
    content: [{ type: 'text', text: `用户信息：\n- 姓名: ${user.name}` }],
  };
});

// 注册资源（静态文档/配置等）
server.registerResource('使用指南', 'docs://guide', {
  description: 'MCP Server 使用文档',
  mimeType: 'text/plain',
}, async () => {
  return {
    contents: [{ uri: 'docs://guide', mimeType: 'text/plain', text: '...' }],
  };
});

// 使用 stdio 传输（本地进程通信）
const transport = new StdioServerTransport();
await server.connect(transport);
```

- **注意事项**：
  - MCP 工具返回格式为 `{ content: [{ type: 'text', text: '...' }] }`，这是固定格式。
  - `StdioServerTransport` 适合本地子进程调用；远程服务可使用 `StreamableHTTP`。
  - **资源（Resources）** 是只读的静态数据（如文档、配置），**工具（Tools）** 是动态可执行的函数。

---

### 知识点 6：MultiServerMCPClient 连接多个 MCP Server

- **概念解释**：`@langchain/mcp-adapters` 提供的 `MultiServerMCPClient` 可以同时连接多个 MCP Server（本地进程或远程 HTTP），并将所有 Server 的工具统一聚合为 LangChain 工具列表。
- **代码示例**（来自 `mcp-test.mjs`）：

```javascript
import { MultiServerMCPClient } from '@langchain/mcp-adapters';

const mcpClient = new MultiServerMCPClient({
  mcpServers: {
    // 本地子进程方式（stdio）
    'my-mcp-server': {
      command: "node",
      args: ["/path/to/my-mcp-server.mjs"]
    },
    // 远程 HTTP 方式（StreamableHTTP）
    "amap-maps-streamableHTTP": {
      url: "https://mcp.amap.com/mcp?key=" + process.env.AMAP_MAPS_API_KEY
    },
    // 使用 npx 启动第三方 MCP Server
    "filesystem": {
      command: "npx",
      args: ["-y", "@modelcontextprotocol/server-filesystem", "/allowed/path"]
    },
  }
});

const tools = await mcpClient.getTools();  // 获取所有 Server 的工具
const modelWithTools = model.bindTools(tools);

// 使用完毕后关闭连接
await mcpClient.close();
```

- **注意事项**：
  - `getTools()` 会聚合所有已配置 Server 的工具，工具名可能有命名冲突需注意。
  - 使用完毕必须调用 `mcpClient.close()` 释放资源（子进程/连接）。
  - 远程 MCP Server 的工具结果可能返回对象（如 `{ text: '...' }`），需要做类型判断再转为字符串。

---

### 知识点 7：MCP Resources 的使用

- **概念解释**：MCP 资源（Resources）类似于"只读的知识库"，可以在 Agent 启动时读取并注入到 System Prompt 中，为 AI 提供背景知识（如使用指南、配置文件、数据库 schema 等）。
- **代码示例**（来自 `langchain-mcp-test.mjs`）：

```javascript
// 列出所有资源
const res = await mcpClient.listResources();

// 读取所有资源内容，拼成字符串
let resourceContent = '';
for (const [serverName, resources] of Object.entries(res)) {
  for (const resource of resources) {
    const content = await mcpClient.readResource(serverName, resource.uri);
    resourceContent += content[0].text;
  }
}

// 将资源内容注入 SystemMessage
const messages = [
  new SystemMessage(resourceContent),  // AI 以此为背景知识
  new HumanMessage(query)
];
```

- **注意事项**：
  - Resources 与 Tools 的区别：Resources 是静态数据（AI 读取），Tools 是动态函数（AI 调用执行）。
  - 资源内容注入 SystemMessage，相当于给 AI "背景文档"，特别适合放使用说明、数据字典等。

---

### 知识点 8：Node.js `child_process.spawn` 执行命令

- **概念解释**：`spawn` 是 Node.js 原生的子进程 API，相比 `exec` 的优势在于**实时流式输出**（通过 `stdio: 'inherit'` 直接将子进程的输出打印到父进程终端）。
- **代码示例**（来自 `all-tools.mjs` 和 `node-exec.mjs`）：

```javascript
import { spawn } from 'node:child_process';

const [cmd, ...args] = command.split(' ');

const child = spawn(cmd, args, {
  cwd,             // 工作目录
  stdio: 'inherit', // 实时输出到控制台（不缓存）
  shell: true,     // 使用 shell 解析命令（支持管道等）
});

child.on('close', (code) => {
  if (code === 0) {
    resolve('命令执行成功');
  } else {
    resolve(`命令执行失败，退出码: ${code}`);
  }
});
```

- **注意事项**：
  - `stdio: 'inherit'` 意味着子进程输出实时显示，但**无法在代码中捕获输出内容**；若需要捕获输出请用 `stdio: 'pipe'`。
  - `shell: true` 启用 shell 解析，支持管道 `|`、重定向 `>` 等语法，但 Windows 上 `echo` 行为不同，需特殊处理（可改为 `shell: 'powershell.exe'`）。
  - 在 Agent 的 `execute_command` 工具中，推荐使用 `workingDirectory` 参数代替在命令中写 `cd`，避免路径问题。

---

### 知识点 9：Zod Schema 校验

- **概念解释**：Zod 是 TypeScript-first 的 Schema 校验库，在这里用于定义工具的参数结构。LangChain 会自动将 Zod Schema 转换为 JSON Schema，供 LLM 理解参数格式。
- **代码示例**（来自 `all-tools.mjs`）：

```javascript
import { z } from 'zod';

// 必填字符串
z.object({ filePath: z.string().describe('文件路径') })

// 可选字符串
z.object({
  command: z.string().describe('要执行的命令'),
  workingDirectory: z.string().optional().describe('工作目录（推荐指定）'),
})

// 多个必填参数
z.object({
  filePath: z.string().describe('文件路径'),
  content: z.string().describe('要写入的文件内容'),
})
```

- **注意事项**：
  - `.optional()` 标记可选参数，LLM 可以不传该参数。
  - `.describe()` 是关键，LLM 根据描述理解参数语义来填值。
  - Zod v4（本项目使用）的 API 与 v3 略有差异，注意版本兼容性。

---

## 代码解析

### `hello-langchain.mjs`

最简单的入门文件，直接调用 ChatOpenAI 模型：

```javascript
import dotenv from 'dotenv';
import { ChatOpenAI } from '@langchain/openai';

dotenv.config();

const model = new ChatOpenAI({ 
    modelName: process.env.MODEL_NAME || "qwen-coder-turbo",
    apiKey: process.env.OPENAI_API_KEY,
    configuration: {
        baseURL: process.env.OPENAI_BASE_URL, // 使用自定义 baseURL 兼容国产模型
    },
});

const response = await model.invoke("介绍下自己");
console.log(response.content);
```

关键点：`configuration.baseURL` 让 LangChain 的 OpenAI 客户端指向第三方兼容接口（如通义千问的 API 地址）。

---

### `tool-file-read.mjs`

演示单工具 + 手动工具循环的完整流程，是理解工具调用机制的最佳入门文件：

1. 定义 `readFileTool`
2. `model.bindTools([readFileTool])`
3. 发起初次请求
4. 进入 `while` 循环检测 `tool_calls`
5. 并发执行工具（`Promise.all`）
6. 把结果包装成 `ToolMessage` 追加到历史
7. 再次调用模型，直到没有 `tool_calls`

---

### `all-tools.mjs`

定义并导出四个通用工具，供 `mini-cursor.mjs` 使用：

| 工具名 | 功能 |
|--------|------|
| `read_file` | 读取文件内容（错误时返回错误信息而非抛异常） |
| `write_file` | 写文件（自动 `mkdir -p` 创建目录） |
| `execute_command` | 执行系统命令（`spawn` + `stdio: inherit`，支持 `workingDirectory`） |
| `list_directory` | 列出目录下的文件列表 |

---

### `mini-cursor.mjs`

项目最核心的文件，实现了一个能自主创建 React 项目的 Agent：

- System Prompt 中详细规定了工具使用规则（特别是 `workingDirectory` 的正确用法）
- 任务描述（`case1`）非常详细，包含创建项目、修改代码、安装依赖、启动服务等完整步骤
- 这是一个完整的 **Coding Agent** 原型，展示了 AI 如何自主完成多步骤的编程任务

---

### `my-mcp-server.mjs`

一个最简的自定义 MCP Server，包含：
- 一个**工具**：`query_user`，根据 userId 查询内存中的用户数据
- 一个**资源**：`docs://guide`，返回服务器使用说明文档
- 使用 `StdioServerTransport` 通过标准输入/输出与客户端通信

---

### `langchain-mcp-test.mjs`

演示 LangChain 通过 MCP 连接自定义服务器，并读取 Resources 注入 SystemMessage：

1. 创建 `MultiServerMCPClient`，配置本地 MCP Server
2. `listResources()` + `readResource()` 读取服务器文档
3. 将文档内容作为 SystemMessage 注入 Agent 上下文
4. 运行 Agent 循环完成用户查询任务

---

### `mcp-test.mjs`

演示同时连接多个不同类型的 MCP Server：
- 本地进程（`my-mcp-server`）
- 远程 HTTP（高德地图 MCP）
- npx 启动（文件系统 MCP、Chrome DevTools MCP）

通过一个自然语言任务（查询酒店 + 打开浏览器展示图片）展示多工具协同的强大能力。

---

### `node-exec.mjs`

独立演示 `child_process.spawn` 的原生用法，用于理解 `execute_command` 工具背后的底层机制。

---

## 知识点总结表

| 知识点 | 说明 | 所在文件 |
|--------|------|----------|
| `tool()` 函数定义工具 | 封装函数为 AI 可调用工具，需 name/description/schema | `all-tools.mjs`, `tool-file-read.mjs` |
| `bindTools()` 绑定工具 | 将工具注入模型，模型响应携带 tool_calls | `mini-cursor.mjs`, `tool-file-read.mjs` |
| ReAct 工具循环 | 循环：调用模型→执行工具→返回结果→再次调用 | `tool-file-read.mjs`, `mini-cursor.mjs` |
| `ToolMessage` | 工具结果消息，需携带 tool_call_id | 全部 Agent 文件 |
| MCP Server 构建 | `McpServer` + `registerTool` + `registerResource` | `my-mcp-server.mjs` |
| `MultiServerMCPClient` | 连接多个 MCP Server 并聚合工具 | `mcp-test.mjs`, `langchain-mcp-test.mjs` |
| MCP Resources | 静态只读数据，注入 SystemMessage 作背景知识 | `langchain-mcp-test.mjs` |
| `StdioServerTransport` | 通过 stdio 实现本地进程间 MCP 通信 | `my-mcp-server.mjs` |
| `spawn` 命令执行 | 实时输出子进程，支持 workingDirectory | `all-tools.mjs`, `node-exec.mjs` |
| Zod Schema | 工具参数校验与描述，LLM 据此填充参数 | `all-tools.mjs`, `my-mcp-server.mjs` |
| `baseURL` 配置 | 让 LangChain 适配第三方 OpenAI 兼容 API | `hello-langchain.mjs` 等全部文件 |
| `maxIterations` 防死循环 | Agent 循环设置最大迭代次数 | `mini-cursor.mjs`, `mcp-test.mjs` |

---

## 扩展学习

1. **LangGraph**：在 ReAct 循环基础上，用状态机（图）管理更复杂的 Agent 工作流，支持分支、循环、人工介入。
2. **LangChain AgentExecutor**：LangChain 内置的 Agent 执行器，封装了手动实现的工具循环逻辑，可减少样板代码。
3. **MCP 协议深入**：了解 MCP 的 Prompts、Sampling 等高级特性，以及如何构建生产级 MCP Server。
4. **工具调用的错误处理**：在生产环境中，工具可能失败，需要设计重试机制、错误恢复策略和降级方案。
5. **流式输出（Streaming）**：使用 `model.stream()` 实现打字机效果，提升用户体验。
6. **结构化输出（Structured Output）**：使用 `model.withStructuredOutput()` 让模型直接返回 JSON 对象，而非自由文本。
7. **Prompt Engineering for Tools**：如何写好 System Prompt 来引导 Agent 正确使用工具，避免错误调用（参考 `mini-cursor.mjs` 中的规则设计）。
