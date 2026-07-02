# openai_adk

Google ADK 原生仅支持 Gemini 模型，无法直接对接 OpenAI 兼容 API（DeepSeek、硅基流动、智谱、Ollama 等）。本包作为适配层，将 OpenAI Chat Completions 协议（含 SSE 流式、function calling）转换为 ADK 标准的 `adkmodel.LLM` 接口，使 Agent 可无缝切换供应商，无需修改上层业务逻辑。

## 功能特性

- **多供应商支持**：支持 Gemini、OpenAI 兼容、硅基流动、DeepSeek、智谱、小米 MiMo、Ollama、vLLM 等
- **SSE 流式响应**：完整支持 Server-Sent Events 流式输出
- **Function Calling**：支持工具调用，自动转换 ADK 与 OpenAI 格式
- **自动重试**：网络错误、429 限流、5xx 错误自动重试（指数退避）
- **调试日志**：可选的请求/响应日志记录
- **思考内容**：支持 DeepSeek、Qwen3 等模型的推理内容输出

## 支持的供应商

| 供应商 | 标识符 | 默认 BaseURL | 说明 |
|--------|--------|--------------|------|
| Google Gemini | `gemini` | - | 原生支持，需 API Key |
| OpenAI 兼容 | `openai_compatible` | - | 通用 OpenAI 兼容接口 |
| 硅基流动 | `siliconflow` | `https://api.siliconflow.cn/v1` | 国内 AI 平台 |
| DeepSeek | `deepseek` | `https://api.deepseek.com` | 深度求索模型 |
| 智谱 | `zhipu` | `https://open.bigmodel.cn/api/paas/v4` | 智谱 AI 平台 |
| 小米 MiMo | `mimo` | `https://api.xiaomimimo.com` | 小米 AI 模型 |
| Ollama | `ollama` | `http://localhost:11434/v1` | 本地部署，无需 API Key |
| vLLM | `vllm` | `http://localhost:8000/v1` | 本地部署，无需 API Key |

## 安装

```bash
go get github.com/mr-dadong/openai_adk
```

## 快速开始

### 基本用法

```go
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/mr-dadong/openai_adk"
)

func main() {
	ctx := context.Background()

	// 创建 DeepSeek 模型
	model, err := openai_adk.NewModel(ctx, openai_adk.ClientConfig{
		Name:      "deepseek-chat",
		Provider:  openai_adk.ProviderDeepseek,
		ModelName: "deepseek-chat",
		APIKey:    os.Getenv("DEEPSEEK_API_KEY"),
	})
	if err != nil {
		log.Fatalf("创建模型失败: %v", err)
	}

	fmt.Printf("模型名称: %s\n", model.Name())
}
```

### 使用硅基流动

```go
model, err := openai_adk.NewModel(ctx, openai_adk.ClientConfig{
	Name:      "siliconflow-deepseek",
	Provider:  openai_adk.ProviderSiliconFlow,
	ModelName: "deepseek-ai/DeepSeek-V3.2",
	APIKey:    os.Getenv("SILICONFLOW_API_KEY"),
})
```

### 使用本地 Ollama

```go
// Ollama 本地部署无需 API Key
model, err := openai_adk.NewModel(ctx, openai_adk.ClientConfig{
	Name:      "ollama-llama3",
	Provider:  openai_adk.ProviderOllama,
	ModelName: "llama3",
	BaseURL:   "http://localhost:11434/v1", // 可选，有默认值
})
```

### 使用自定义 OpenAI 兼容接口

```go
model, err := openai_adk.NewModel(ctx, openai_adk.ClientConfig{
	Name:      "custom-model",
	Provider:  openai_adk.ProviderOpenAICompatible,
	ModelName: "gpt-4",
	BaseURL:   "https://your-api-endpoint.com/v1",
	APIKey:    os.Getenv("CUSTOM_API_KEY"),
})
```

## 与 ADK-Go 集成使用

本包实现了 `adkmodel.LLM` 接口，可直接与 Google ADK-Go 框架集成。以下是一个完整的示例：

### 安装 ADK-Go

```bash
go get google.golang.org/adk
```

### 创建使用 OpenAI 兼容模型的 Agent

```go
package main

import (
	"context"
	"log"
	"os"

	"github.com/mr-dadong/openai_adk"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
)

// 定义工具输入输出结构
type ReadFileInput struct {
	Path string `json:"path" description:"文件路径"`
}

type ReadFileOutput struct {
	Content string `json:"content"`
}

func main() {
	ctx := context.Background()

	// 1. 创建 OpenAI 兼容模型（以 DeepSeek 为例）
	model, err := openai_adk.NewModel(ctx, openai_adk.ClientConfig{
		Name:      "deepseek-chat",
		Provider:  openai_adk.ProviderDeepseek,
		ModelName: "deepseek-chat",
		APIKey:    os.Getenv("DEEPSEEK_API_KEY"),
	})
	if err != nil {
		log.Fatalf("创建模型失败: %v", err)
	}

	// 2. 创建工具
	readFileTool, err := functiontool.New(functiontool.Config{
		Name:        "read_file",
		Description: "读取指定路径的文件内容",
		Handler: func(ctx tool.Context, input ReadFileInput) (ReadFileOutput, error) {
			content, err := os.ReadFile(input.Path)
			if err != nil {
				return ReadFileOutput{}, err
			}
			return ReadFileOutput{Content: string(content)}, nil
		},
	})
	if err != nil {
		log.Fatalf("创建工具失败: %v", err)
	}

	// 3. 创建 Agent
	agent, err := llmagent.New(llmagent.Config{
		Name:        "file_reader_agent",
		Model:       model,
		Description: "一个可以读取文件的助手",
		Instruction: `你是一个文件助手。当用户要求读取文件时，使用 read_file 工具获取文件内容，
然后将内容展示给用户。如果文件不存在，如实告知用户。`,
		Tools: []tool.Tool{readFileTool},
	})
	if err != nil {
		log.Fatalf("创建 Agent 失败: %v", err)
	}

	// 4. 使用 Agent（这里只是示例，实际使用需要配合 launcher）
	log.Printf("Agent 创建成功: %s\n", agent.Name())
}
```

### 使用硅基流动的 Agent 示例

```go
// 创建硅基流动模型
model, err := openai_adk.NewModel(ctx, openai_adk.ClientConfig{
	Name:      "siliconflow-deepseek",
	Provider:  openai_adk.ProviderSiliconFlow,
	ModelName: "deepseek-ai/DeepSeek-V3.2",
	APIKey:    os.Getenv("SILICONFLOW_API_KEY"),
})

// 创建 Agent 时使用该模型
agent, err := llmagent.New(llmagent.Config{
	Name:  "my_agent",
	Model: model,
	// ... 其他配置
})
```

### 使用本地 Ollama 的 Agent 示例

```go
// 创建本地 Ollama 模型（无需 API Key）
model, err := openai_adk.NewModel(ctx, openai_adk.ClientConfig{
	Name:      "ollama-agent",
	Provider:  openai_adk.ProviderOllama,
	ModelName: "llama3",
})

// 创建 Agent
agent, err := llmagent.New(llmagent.Config{
	Name:  "local_agent",
	Model: model,
	// ... 其他配置
})
```

## 配置选项

### ClientConfig 结构体

```go
type ClientConfig struct {
	Name         string // 模型显示名称，用于错误消息
	Provider     string // 供应商标识：gemini, openai_compatible, siliconflow 等
	ModelName    string // 模型标识符，如 gpt-4, deepseek-chat
	BaseURL      string // API 端点地址
	APIKey       string // API 密钥
	APIKeyEnv    string // API 密钥环境变量名，用于错误提示
	ToolsEnabled *bool  // 是否支持工具调用，默认 true
	Debug        bool   // 是否启用请求/响应调试日志
}
```

### 调试模式

启用调试模式可以查看请求和响应的详细信息：

```go
model, err := openai_adk.NewModel(ctx, openai_adk.ClientConfig{
	Name:      "debug-model",
	Provider:  openai_adk.ProviderDeepseek,
	ModelName: "deepseek-chat",
	APIKey:    os.Getenv("DEEPSEEK_API_KEY"),
	Debug:     true, // 启用调试日志
})
```

调试日志会输出到标准错误输出（stderr），包括：
- 完整的请求 JSON
- 完整的响应 JSON
- Token 使用统计

## 错误处理

包会返回明确的错误信息：

- **缺少 API Key**：`api_key is required by model "xxx"`
- **不支持的供应商**：`unsupported model provider "xxx"`
- **缺少 BaseURL**：`base_url is required by openai_compatible model "xxx"`
- **HTTP 错误**：包含状态码和错误详情
- **API 错误**：包含错误类型和消息

## 重试机制

包内置了自动重试机制：

- **最大重试次数**：3 次
- **重试延迟**：指数退避（1s, 2s, 4s），最大 30s
- **可重试错误**：网络错误、429 限流、5xx 服务器错误
- **Retry-After 支持**：自动解析响应头中的 Retry-After 值

## 示例代码

更多示例请参考 `openai_compatible_test.go` 文件，其中包含：

- 基本文本生成
- 流式响应处理
- 工具调用
- 多模态内容（图片）
- 历史消息处理

## 许可证

MIT License
