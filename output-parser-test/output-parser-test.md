# output-parser-test 学习笔记

## 概述

`output-parser-test` 是一个围绕 **LangChain 输出解析（Output Parsing）与结构化输出（Structured Output）** 的实验项目。它系统演示了如何把大模型的自然语言结果转换成可编程处理的结构化数据，包括：

- 让模型直接返回 JSON，再手动 `JSON.parse`
- 使用 `JsonOutputParser` / `StructuredOutputParser` 自动解析
- 使用 `zod` 定义强类型 schema 并做校验
- 使用 `withStructuredOutput()` 直接拿到结构化对象
- 使用工具调用（tool calling）承载结构化输出
- 在流式场景中解析普通文本、结构化 JSON 和 tool call 参数
- 使用原生 `response_format: json_schema` 约束模型输出
- 使用 XML 作为替代结构化格式

除了基础示例，这个目录还包含几个更贴近 Agent 场景的案例：

- `mini-cursor.mjs`：一个最小版“会调用工具的智能体”
- `smart-import.mjs`：从自然语言中提取结构化人物信息并写入 MySQL
- `create-table.mjs`：准备数据库表结构

整体上，这个项目的核心目标是回答一个实际问题：**如何让 LLM 的输出从“给人看”变成“给程序用”。**

---

## 技术栈

| 库 / 工具 | 版本 | 用途 |
|-----------|------|------|
| `@langchain/core` | `^1.1.17` | 输出解析器、消息对象、工具定义 |
| `@langchain/openai` | `^1.2.3` | `ChatOpenAI` 模型接入 |
| `zod` | `^4.3.6` | 定义结构化 schema、运行时校验 |
| `zod-to-json-schema` | `^3.25.1` | 将 Zod schema 转为原生 JSON Schema |
| `mysql2` | `^3.16.2` | MySQL 连接与批量插入 |
| `chalk` | `^5.6.2` | 终端彩色日志输出 |
| `dotenv` | `^17.2.3` | 环境变量加载 |

补充说明：

- `package.json` 的 `type` 是 `commonjs`，但示例文件都使用 `.mjs`，因此这些文件依然会按 ESM 方式执行。
- 多数脚本通过 `dotenv/config` 读取 `MODEL_NAME`、`OPENAI_API_KEY`、`OPENAI_BASE_URL` 等环境变量。

---

## 文件结构

```text
output-parser-test/
├── package.json
├── src/
│   ├── normal.mjs                         # 让模型直接输出 JSON，再手动 JSON.parse
│   ├── json-output-parser.mjs             # JsonOutputParser 基础示例
│   ├── structured-output-parser.mjs       # StructuredOutputParser（简单字段定义）
│   ├── structured-output-parser2.mjs      # StructuredOutputParser + Zod 复杂嵌套结构
│   ├── with-structured-output.mjs         # withStructuredOutput 直接返回结构化对象
│   ├── stream-normal.mjs                  # 普通文本流式输出
│   ├── stream-with-structured-output.mjs  # withStructuredOutput 的流式结构化输出
│   ├── stream-structured-partial.mjs      # 先流文本，再整体解析为结构化对象
│   ├── tool-calls-args.mjs                # 通过 bindTools 让模型以 tool call 返回结构化参数
│   ├── stream-tool-calls-raw.mjs          # 直接查看 tool_call_chunks 的原始流
│   ├── stream-tool-calls-parser.mjs       # 用 JsonOutputToolsParser 解析流式工具参数
│   ├── structured-json-schema.mjs         # 使用原生 response_format + json_schema
│   ├── xml-output-parser.mjs              # XML 输出与 XMLOutputParser 解析
│   └── test/
│       ├── all-tools.mjs                  # 一组可复用的本地工具
│       ├── create-table.mjs               # 创建 MySQL 数据库与 demo 表
│       ├── mini-cursor.mjs                # 一个最小可运行的工具型 Agent 示例
│       └── smart-import.mjs               # 从自然语言提取好友信息并写入 MySQL
```

---

## 核心知识点

### 知识点 1：手动 `JSON.parse` 是最朴素、也最脆弱的做法

- **概念解释**：最直接的方式是让模型“按 JSON 返回”，然后代码里手动调用 `JSON.parse()`。这种方案实现简单，但非常依赖模型是否老老实实输出纯 JSON。
- **代码示例**（来自 `src/normal.mjs`）：

```javascript
const question = "请介绍一下爱因斯坦的信息。请以 JSON 格式返回，包含以下字段：name（姓名）、birth_year（出生年份）、nationality（国籍）、major_achievements（主要成就，数组）、famous_theory（著名理论）。";

const response = await model.invoke(question);
console.log(response.content);

const jsonResult = JSON.parse(response.content);
console.log(jsonResult);
```

- **注意事项**：
  - 如果模型在 JSON 前后多说一句解释文字，`JSON.parse` 就会直接报错。
  - 这种方式只做“语法解析”，**不做字段校验**。
  - 适合最基础的实验，不适合对稳定性要求高的生产场景。

---

### 知识点 2：`JsonOutputParser` 会主动告诉模型“该怎么输出”

- **概念解释**：`JsonOutputParser` 的思路是：
  1. 通过 `getFormatInstructions()` 生成格式要求
  2. 把格式要求拼进 prompt
  3. 再用 `parser.parse()` 对模型返回结果做解析
- **代码示例**（来自 `src/json-output-parser.mjs`）：

```javascript
const parser = new JsonOutputParser();

const question = `请介绍一下爱因斯坦的信息。请以 JSON 格式返回，包含以下字段：name（姓名）、birth_year（出生年份）、nationality（国籍）、major_achievements（主要成就，数组）、famous_theory（著名理论）。

${parser.getFormatInstructions()}`;

const response = await model.invoke(question);
const result = await parser.parse(response.content);
```

- **注意事项**：
  - `getFormatInstructions()` 是关键，它不是解析器本身，而是给模型的“输出说明书”。
  - 仍然依赖模型遵守 prompt，只是成功率比“纯手写提示”更高。
  - 相比手动 `JSON.parse()`，这里的代码更可复用、可维护。

---

### 知识点 3：`StructuredOutputParser.fromNamesAndDescriptions()` 适合简单结构

- **概念解释**：当结构不复杂时，可以用字段名 + 描述快速定义一个结构化输出模板，不必一开始就上 Zod。
- **代码示例**（来自 `src/structured-output-parser.mjs`）：

```javascript
const parser = StructuredOutputParser.fromNamesAndDescriptions({
    name: "姓名",
    birth_year: "出生年份",
    nationality: "国籍",
    major_achievements: "主要成就，用逗号分隔的字符串",
    famous_theory: "著名理论"
});

const question = `请介绍一下爱因斯坦的信息。

${parser.getFormatInstructions()}`;

const response = await model.invoke(question);
const result = await parser.parse(response.content);
```

- **注意事项**：
  - 这种方式更轻量，适合 demo、字段少、嵌套少的场景。
  - 字段描述会直接影响模型输出质量。
  - 复杂数组、嵌套对象、可选字段一多，就更适合升级到 Zod schema。

---

### 知识点 4：`StructuredOutputParser.fromZodSchema()` 能把“解析”和“校验”结合起来

- **概念解释**：Zod 可以把结构定义、字段说明、可选性和类型约束放在一个 schema 里，再由 `StructuredOutputParser` 负责把模型结果解析成对象并验证。
- **代码示例**（来自 `src/structured-output-parser2.mjs`）：

```javascript
const scientistSchema = z.object({
    name: z.string().describe("科学家的全名"),
    birth_year: z.number().describe("出生年份"),
    death_year: z.number().optional().describe("去世年份，如果还在世则不填"),
    nationality: z.string().describe("国籍"),
    fields: z.array(z.string()).describe("研究领域列表"),
    awards: z.array(
        z.object({
            name: z.string().describe("奖项名称"),
            year: z.number().describe("获奖年份"),
            reason: z.string().optional().describe("获奖原因")
        })
    ).describe("获得的重要奖项列表"),
    major_achievements: z.array(z.string()).describe("主要成就列表"),
    biography: z.string().describe("简短传记，100字以内")
});

const parser = StructuredOutputParser.fromZodSchema(scientistSchema);
const response = await model.invoke(question);
const result = await parser.parse(response.content);
```

- **注意事项**：
  - `.describe()` 很重要，它既是文档，也是给模型看的字段语义提示。
  - `optional()` 可以降低模型在“不确定字段”上的失败率。
  - 解析通过后拿到的对象更可靠，出错时也能知道是“JSON 语法错”还是“schema 不匹配”。

---

### 知识点 5：`withStructuredOutput()` 是最省心的高层封装

- **概念解释**：`withStructuredOutput(schema)` 会返回一个“已经绑定结构化约束”的模型实例。调用方不用再手动拼格式说明，也不用自己调用 parser 解析。
- **代码示例**（来自 `src/with-structured-output.mjs`）：

```javascript
const scientistSchema = z.object({
    name: z.string().describe("科学家的全名"),
    birth_year: z.number().describe("出生年份"),
    nationality: z.string().describe("国籍"),
    fields: z.array(z.string()).describe("研究领域列表"),
});

const structuredModel = model.withStructuredOutput(scientistSchema);
const result = await structuredModel.invoke("介绍一下爱因斯坦");
```

- **注意事项**：
  - 这是实际项目里最常用、最顺手的一种写法。
  - 它屏蔽了很多底层细节，开发体验最好。
  - 但你也要知道：它并不是“魔法”，底层仍然是通过模型能力或工具调用机制来达成结构化输出。

---

### 知识点 6：流式输出分两类——“文本先流出”与“结构逐步成型”

- **概念解释**：这个目录对比了三种典型流式场景：
  1. 普通文本流式输出
  2. 结构化文本先完整流出，再最后整体解析
  3. 结构化对象本身在流里逐步成型

#### 6.1 普通文本流式输出

- **代码示例**（来自 `src/stream-normal.mjs`）：

```javascript
const stream = await model.stream(prompt);

let fullContent = '';
for await (const chunk of stream) {
    const content = chunk.content;
    fullContent += content;
    process.stdout.write(content);
}
```

- **特点**：
  - 最接近聊天界面的打字机效果
  - 只适合“人看”的场景
  - 程序侧想拿结构化数据时，还得事后再解析

#### 6.2 文本先流，再整体解析为结构化对象

- **代码示例**（来自 `src/stream-structured-partial.mjs`）：

```javascript
const parser = StructuredOutputParser.fromZodSchema(schema);
const prompt = `详细介绍莫扎特的信息。\n\n${parser.getFormatInstructions()}`;
const stream = await model.stream(prompt);

let fullContent = '';
for await (const chunk of stream) {
    fullContent += chunk.content;
    process.stdout.write(chunk.content);
}

const result = await parser.parse(fullContent);
```

- **特点**：
  - 用户侧能看到实时输出
  - 代码侧在结束后依然能拿到结构化对象
  - 中途不能稳定获得“半成品结构”

#### 6.3 结构化对象本身随流更新

- **代码示例**（来自 `src/stream-with-structured-output.mjs`）：

```javascript
const structuredModel = model.withStructuredOutput(schema);
const stream = await structuredModel.stream(prompt);

let result = null;
for await (const chunk of stream) {
    result = chunk;
    console.log(JSON.stringify(chunk, null, 2));
}
```

- **特点**：
  - 每个 chunk 都可能是当前阶段的“部分对象”
  - 更适合 UI 逐步显示结构化字段
  - 需要考虑字段暂时为空、数组未完整等中间态问题

---

### 知识点 7：Tool Calling 可以把结构化结果放到 `tool_calls[].args` 里

- **概念解释**：除了让模型“返回 JSON 文本”，还可以把结构化输出设计成一次工具调用。模型不再输出一段 JSON 字符串，而是输出“我要调用哪个工具 + 参数是什么”。
- **代码示例**（来自 `src/tool-calls-args.mjs`）：

```javascript
const modelWithTool = model.bindTools([
    {
        name: "extract_scientist_info",
        description: "提取和结构化科学家的详细信息",
        schema: scientistSchema
    }
]);

const response = await modelWithTool.invoke("介绍一下爱因斯坦");
const result = response.tool_calls[0].args;
```

- **注意事项**：
  - 这里结构化数据不在 `response.content`，而在 `response.tool_calls[0].args`。
  - 对 Agent 来说，这种形式更自然，因为它和“调用工具”是一套统一协议。
  - 当你要串工具、做函数调用、执行多步推理时，这种方式通常比“纯 JSON 文本”更稳。

---

### 知识点 8：流式 Tool Call 最难的地方，是参数 JSON 会被拆碎

- **概念解释**：一旦开启流式 tool calling，模型返回的 `args` 往往不是一整个完整 JSON，而是被分裂成很多片段。这也是实际开发里最容易懵的地方。

#### 8.1 直接观察原始碎片

- **代码示例**（来自 `src/stream-tool-calls-raw.mjs`）：

```javascript
const stream = await modelWithTool.stream("详细介绍牛顿的生平和成就");

for await (const chunk of stream) {
    if (chunk.tool_call_chunks && chunk.tool_call_chunks.length > 0) {
        process.stdout.write(chunk.tool_call_chunks[0].args || '');
    }
}
```

- **现象**：
  - 你看到的可能是 JSON 的残片，而不是完整对象。
  - 如果直接 `JSON.parse`，大概率会失败。

#### 8.2 用 `JsonOutputToolsParser` 重组成可用对象

- **代码示例**（来自 `src/stream-tool-calls-parser.mjs`）：

```javascript
const parser = new JsonOutputToolsParser();
const chain = modelWithTool.pipe(parser);
const stream = await chain.stream("详细介绍牛顿的生平和成就");

for await (const chunk of stream) {
    if (chunk.length > 0) {
        const toolCall = chunk[0];
        const currentContent = JSON.stringify(toolCall.args || {}, null, 2);
        console.log(toolCall.args);
    }
}
```

- **注意事项**：
  - `JsonOutputToolsParser` 的核心价值就是：把工具调用的 JSON 参数安全地拼起来。
  - 这个能力对于“实时展示结构化进度”特别重要。
  - 你需要意识到“原始流”和“可消费的流”是两层东西。

---

### 知识点 9：`response_format: json_schema` 是更“原生”的结构化约束

- **概念解释**：部分模型支持原生 JSON Schema 模式，不再依赖 prompt 里讲规则，而是直接通过底层参数要求模型输出满足某个 schema 的 JSON。
- **代码示例**（来自 `src/structured-json-schema.mjs`）：

```javascript
const scientistSchema = z.object({
    name: z.string().describe("科学家的全名"),
    birth_year: z.number().describe("出生年份"),
    field: z.string().describe("主要研究领域"),
    achievements: z.array(z.string()).describe("主要成就列表")
}).strict();

const nativeJsonSchema = zodToJsonSchema(scientistSchema);

const model = new ChatOpenAI({
    modelName: "qwen-max",
    modelKwargs: {
        response_format: {
            type: "json_schema",
            json_schema: {
                name: "scientist_info",
                strict: true,
                schema: nativeJsonSchema
            }
        }
    }
});
```

- **注意事项**：
  - `.strict()` 的意思是：不允许多余字段。
  - `zodToJsonSchema()` 把应用层 schema 转成模型接口能理解的原生 JSON Schema。
  - 这种方式通常比“靠 prompt 提示”更硬约束，但前提是模型/供应商真的支持这个能力。

---

### 知识点 10：XML 也是一种可解析输出格式

- **概念解释**：结构化输出不一定只能用 JSON。LangChain 也支持 XML 解析，适合某些老系统集成或需要标签化结构的场景。
- **代码示例**（来自 `src/xml-output-parser.mjs`）：

```javascript
const parser = new XMLOutputParser();

const question = `请提取以下文本中的人物信息：阿尔伯特·爱因斯坦出生于 1879 年，是一位伟大的物理学家。

${parser.getFormatInstructions()}`;

const response = await model.invoke(question);
const result = await parser.parse(response.content);
```

- **注意事项**：
  - XML 可读性不错，但在 JavaScript 生态里通常不如 JSON 顺手。
  - 如果下游系统已经是 XML 协议，这种方式就很自然。
  - 一旦标签层级复杂，调试成本会高于 JSON。

---

### 知识点 11：`tool()` 可以把本地能力包装成 LangChain 工具

- **概念解释**：`src/test/all-tools.mjs` 演示了如何把文件读写、命令执行、目录遍历等能力封装成标准工具，供模型调用。
- **代码示例**（来自 `src/test/all-tools.mjs`）：

```javascript
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
  - `schema` 决定了模型能传什么参数。
  - 工具函数最好返回清晰、可继续推理的文本，而不是只打印日志。
  - 工具越规范，模型越容易稳定调用。

---

### 知识点 12：最小 Agent 的核心是“消息历史 + 工具调用 + 工具结果回填”

- **概念解释**：`src/test/mini-cursor.mjs` 是本目录最有教学价值的案例之一。它不只是解析输出，而是把结构化输出用于一个完整 Agent 回路中。
- **关键流程**：
  1. `SystemMessage` 规定工具使用规则
  2. `HumanMessage` 提交任务
  3. 模型流式生成回复/工具调用
  4. 把完整的 `AIMessage` 存回历史
  5. 执行真实工具
  6. 用 `ToolMessage` 把结果回填给模型
  7. 继续下一轮，直到模型不再调用工具

- **代码示例**（来自 `src/test/mini-cursor.mjs`）：

```javascript
const modelWithTools = model.bindTools(tools);
const history = new InMemoryChatMessageHistory();

await history.addMessage(new SystemMessage(`你是一个项目管理助手，使用工具完成任务。`));
await history.addMessage(new HumanMessage(query));

const rawStream = await modelWithTools.stream(messages);
let fullAIMessage = null;

for await (const chunk of rawStream) {
    fullAIMessage = fullAIMessage ? fullAIMessage.concat(chunk) : chunk;
}

await history.addMessage(fullAIMessage);

for (const toolCall of fullAIMessage.tool_calls) {
    const toolResult = await foundTool.invoke(toolCall.args);
    await history.addMessage(
        new ToolMessage({
            content: toolResult,
            tool_call_id: toolCall.id,
        })
    );
}
```

- **注意事项**：
  - 流式阶段要先把 `AIMessageChunk` 拼回完整消息，再统一入历史。
  - `ToolMessage` 的 `tool_call_id` 必须对应之前的工具调用。
  - 这是理解 LangChain Agent 内部机制的一个非常好的入门样例。

---

### 知识点 13：`JsonOutputToolsParser.parseResult()` 适合解析“还在增长中的工具调用”

- **概念解释**：`mini-cursor.mjs` 里还有一个非常实战的点：一边接收流，一边尝试把“半成品 AI 消息”解析成工具调用，从而实现写文件内容的实时预览。
- **代码示例**（来自 `src/test/mini-cursor.mjs`）：

```javascript
const toolParser = new JsonOutputToolsParser();

for await (const chunk of rawStream) {
    fullAIMessage = fullAIMessage ? fullAIMessage.concat(chunk) : chunk;

    let parsedTools = null;
    try {
        parsedTools = await toolParser.parseResult([{ message: fullAIMessage }]);
    } catch (e) {
        // JSON 还不完整，先忽略
    }
}
```

- **注意事项**：
  - 这是一种“增量解析”思路。
  - 对于 `write_file` 这种会生成长文本参数的工具，体验会明显更好。
  - 真正难的不是“最终解析”，而是“流中间怎么稳定地解析半成品”。

---

### 知识点 14：结构化输出可以直接驱动数据库写入

- **概念解释**：`src/test/smart-import.mjs` 演示了一个典型业务场景：让模型从一段自然语言中提取多个人的结构化信息，再批量写入数据库。
- **代码示例**（来自 `src/test/smart-import.mjs`）：

```javascript
const friendSchema = z.object({
  name: z.string().describe('姓名'),
  gender: z.string().describe('性别（男/女）'),
  birth_date: z.string().describe('出生日期，格式：YYYY-MM-DD，如果无法确定具体日期，根据年龄估算'),
  company: z.string().nullable().describe('公司名称，如果没有则返回 null'),
  title: z.string().nullable().describe('职位/头衔，如果没有则返回 null'),
  phone: z.string().nullable().describe('手机号，如果没有则返回 null'),
  wechat: z.string().nullable().describe('微信号，如果没有则返回 null'),
});

const friendsArraySchema = z.array(friendSchema).describe('好友信息数组');
const structuredModel = model.withStructuredOutput(friendsArraySchema);
const results = await structuredModel.invoke(prompt);
```

接着把结果批量插入 MySQL：

```javascript
const values = results.map((result) => [
  result.name,
  result.gender,
  result.birth_date || null,
  result.company,
  result.title,
  result.phone,
  result.wechat,
]);

const [insertResult] = await connection.query(insertSql, [values]);
```

- **注意事项**：
  - 这类场景比“打印到控制台”更接近真实业务价值。
  - 如果 schema 设计得足够细，LLM 就可以承担一部分“文本结构化抽取”的工作。
  - 但生产中仍然建议加二次校验、去重、异常补录等保护逻辑。

---

### 知识点 15：模型输出解析的本质，是“约束 + 验证 + 消费”三件事

- **概念解释**：这个目录把结构化输出的三层本质拆得很清楚：
  - **约束**：通过 prompt、schema、tool call、response_format 告诉模型应该怎么输出
  - **验证**：通过 parser / zod 判断它是否真的按要求输出
  - **消费**：把解析后的数据用于日志、UI、数据库、工具执行、Agent 状态推进

- **实践启发**：
  - 输出解析不是孤立功能，而是 Agent / 自动化工作流的基础设施。
  - 结构化输出质量，直接决定后续程序逻辑是否稳定。

---

## 代码解析

### `src/normal.mjs`
最基础的“提示模型返回 JSON + 手动 `JSON.parse`”示例。优点是简单，缺点是最容易被模型输出格式波动击穿。

### `src/json-output-parser.mjs`
引入 `JsonOutputParser` 后，prompt 更规范，解析更自动化。适合学习 LangChain parser 的第一步。

### `src/structured-output-parser.mjs`
展示 `StructuredOutputParser.fromNamesAndDescriptions()` 的用法。适合简单字段场景，不需要定义完整的 Zod schema。

### `src/structured-output-parser2.mjs`
升级为复杂嵌套 Zod schema，包含数组、对象、可选字段，展示了结构化输出在复杂数据模型上的表达能力。

### `src/with-structured-output.mjs`
用最少代码完成“定义 schema → 调用模型 → 得到强类型对象”的闭环，是日常开发中最推荐的写法之一。

### `src/stream-normal.mjs`
演示非结构化文本流式输出。重点在 `for await...of` 消费模型流，以及累积完整文本。

### `src/stream-with-structured-output.mjs`
演示 `withStructuredOutput()` 在流式模式下的表现。每个 chunk 都可能是一份“逐步成型的对象”。

### `src/stream-structured-partial.mjs`
折中方案：先流式输出 JSON 文本给用户看，再在结束后整体解析。适合兼顾用户体验与程序消费。

### `src/tool-calls-args.mjs`
展示结构化输出与工具调用的结合。模型不是“回答一段 JSON”，而是“产生一个工具调用参数对象”。

### `src/stream-tool-calls-raw.mjs`
直接观察原始 `tool_call_chunks`，帮助理解：为什么流式 tool call 不能直接拿来 `JSON.parse`。

### `src/stream-tool-calls-parser.mjs`
在原始流上加一层 `JsonOutputToolsParser`，把工具参数逐步还原成程序可消费的对象，是流式 tool call 的关键示例。

### `src/structured-json-schema.mjs`
不走 prompt 约束，而是用底层 `response_format.json_schema` 做更硬的输出限制，体现“原生结构化输出”能力。

### `src/xml-output-parser.mjs`
展示除了 JSON 之外的另一种结构化格式。适合理解 parser 抽象并不只服务于 JSON。

### `src/test/all-tools.mjs`
封装 4 个本地工具：读文件、写文件、执行命令、列目录。是 Agent 示例的基础设施层。

### `src/test/create-table.mjs`
创建 `hello` 数据库和 `friends` 表，并插入 demo 数据。为 `smart-import.mjs` 提供下游存储环境。

### `src/test/mini-cursor.mjs`
本项目最完整的 Agent 案例：模型绑定工具、流式生成、增量解析、工具执行、结果回填、继续推理，形成闭环。

### `src/test/smart-import.mjs`
用 `withStructuredOutput()` 从自然语言中抽取多个好友的结构化信息，再批量插入 MySQL。展示了结构化输出在数据导入类场景的实际价值。

---

## 数据流梳理

### 场景一：普通 JSON 解析

```text
用户问题
  -> model.invoke(prompt)
  -> response.content（字符串）
  -> JSON.parse / parser.parse
  -> JavaScript 对象
```

### 场景二：Tool Call 结构化输出

```text
用户问题
  -> model.bindTools(tools)
  -> AIMessage.tool_calls
  -> tool_calls[0].args
  -> 结构化参数对象
```

### 场景三：Agent + 工具回路

```text
用户任务
  -> 模型生成 tool call
  -> 程序执行本地工具
  -> ToolMessage 回填
  -> 模型继续思考
  -> 最终文本答复 / 更多工具调用
```

### 场景四：自然语言抽取入库

```text
原始描述文本
  -> withStructuredOutput(arraySchema)
  -> 结构化好友数组
  -> MySQL batch insert
  -> 数据库存档
```

---

## 难点与坑

### 1. “模型说自己返回 JSON” 不等于真的能被解析
- 手写 prompt 很容易遇到前后夹杂说明文字、Markdown 代码块、字段缺失等情况。
- 只要你后面要自动消费数据，就应优先用 parser / schema / tool call。

### 2. 流式 Tool Call 的参数不是一次性给全
- 原始 `tool_call_chunks[].args` 往往是碎片。
- 想做实时展示或中途解析，必须有增量拼装逻辑。

### 3. `response.content` 和 `response.tool_calls` 是两个世界
- 纯文本/JSON 解析主要看 `content`
- 工具调用型结构化输出主要看 `tool_calls[].args`
- 很多人第一次接触时会取错地方

### 4. schema 描述质量会直接影响模型输出质量
- `z.string().describe("...")` 不是装饰品，而是 prompt 的一部分。
- 描述不清楚，模型就容易乱填字段。

### 5. 结构化输出成功，不代表业务数据一定可信
- 比如年龄估算生日、职位推断、缺失字段补 null，都带有模型推断成分。
- 真正落库前，仍然建议做业务层校验。

### 6. 流式结构化对象要处理“半成品状态”
- 某一时刻对象里可能只有 `name`，数组可能只吐出一半。
- UI 和后端代码都要接受这种“尚未完成”的状态。

---

## 知识点总结表

| 知识点 | 说明 | 所在文件 |
|--------|------|----------|
| 手动 `JSON.parse` | 最简单但最脆弱的结构化输出方式 | `src/normal.mjs` |
| `JsonOutputParser` | 提供格式说明并自动解析 JSON | `src/json-output-parser.mjs` |
| `StructuredOutputParser` | 适合字段化、半结构化输出 | `src/structured-output-parser.mjs` |
| Zod + `StructuredOutputParser` | 复杂嵌套结构的解析与校验 | `src/structured-output-parser2.mjs` |
| `withStructuredOutput()` | 高层封装，直接返回结构化对象 | `src/with-structured-output.mjs` |
| 普通流式输出 | 消费 `model.stream()` 的文本 chunk | `src/stream-normal.mjs` |
| 流式后整体解析 | 先累积文本，再解析为结构化结果 | `src/stream-structured-partial.mjs` |
| 流式结构化对象 | 流里直接得到逐步成型的对象 | `src/stream-with-structured-output.mjs` |
| `bindTools()` | 将 schema 作为工具参数协议 | `src/tool-calls-args.mjs` |
| 原始 `tool_call_chunks` | 观察工具参数流式碎片 | `src/stream-tool-calls-raw.mjs` |
| `JsonOutputToolsParser` | 解析 tool call 参数流 | `src/stream-tool-calls-parser.mjs`, `src/test/mini-cursor.mjs` |
| 原生 JSON Schema 输出 | 使用 `response_format: json_schema` 强约束输出 | `src/structured-json-schema.mjs` |
| `XMLOutputParser` | XML 格式的结构化解析 | `src/xml-output-parser.mjs` |
| `tool()` 封装工具 | 将本地能力包装成可调用工具 | `src/test/all-tools.mjs` |
| `ToolMessage` 回填 | Agent 把工具结果送回模型继续推理 | `src/test/mini-cursor.mjs` |
| 结构化抽取入库 | LLM 提取数组数据并批量写入 MySQL | `src/test/smart-import.mjs` |
| MySQL 初始化脚本 | 创建数据库、表、插入 demo 数据 | `src/test/create-table.mjs` |

---

## 推荐学习顺序

1. 先看 `src/normal.mjs`，理解“为什么光靠 prompt 不稳”
2. 再看 `src/json-output-parser.mjs` 和 `src/structured-output-parser.mjs`
3. 然后看 `src/structured-output-parser2.mjs`，理解 Zod schema 的价值
4. 接着看 `src/with-structured-output.mjs`，体验更现代的写法
5. 再看三个 `stream-*` 文件，理解流式与结构化之间的关系
6. 然后看 `tool-calls-args.mjs` 和两个 `stream-tool-calls-*` 文件
7. 最后看 `src/test/mini-cursor.mjs` 和 `src/test/smart-import.mjs`，把输出解析放到真实 Agent/业务场景里理解

---

## 扩展学习

1. **LangChain Runnable 与 LCEL**：把 parser、model、prompt 进一步串成更清晰的链式工作流。
2. **OpenAI / Qwen 原生结构化输出能力差异**：不同模型对 schema、tool call、json_schema 的支持程度不同。
3. **Zod 的高级能力**：如 `union`、`enum`、`refine`、`transform`，可进一步提升结构化结果的业务约束能力。
4. **生产级容错策略**：比如解析失败重试、自动修复 JSON、字段回填、兜底策略。
5. **前端流式结构化 UI**：把“半成品对象”实时渲染到界面，是 AI 应用体验优化的重要方向。
6. **Agent 工具协议设计**：工具 schema 如何写得既清晰又稳定，是 Agent 成功率的关键因素之一。
7. **结构化抽取评测**：为字段抽取任务建立准确率、召回率、错误类型统计，比只看“能跑通”更重要。
