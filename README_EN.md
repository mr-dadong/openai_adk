# openai_adk

Google ADK natively only supports Gemini models and cannot directly interface with OpenAI-compatible APIs (DeepSeek, SiliconFlow, Zhipu, Ollama, etc.). This package serves as an adapter layer, converting the OpenAI Chat Completions protocol (including SSE streaming and function calling) to the ADK standard `adkmodel.LLM` interface, enabling agents to seamlessly switch providers without modifying upper-level business logic.

## Features

- **Multi-provider Support**: Supports Gemini, OpenAI-compatible, SiliconFlow, DeepSeek, Zhipu, Xiaomi MiMo, Ollama, vLLM, etc.
- **SSE Streaming Response**: Complete support for Server-Sent Events streaming output
- **Function Calling**: Supports tool calling, automatically converting between ADK and OpenAI formats
- **Automatic Retry**: Automatic retry for network errors, 429 rate limiting, and 5xx errors (exponential backoff)
- **Debug Logging**: Optional request/response logging
- **Thinking Content**: Supports reasoning content output for models like DeepSeek and Qwen3

## Supported Providers

| Provider | Identifier | Default BaseURL | Description |
|----------|------------|-----------------|-------------|
| Google Gemini | `gemini` | - | Native support, requires API Key |
| OpenAI Compatible | `openai_compatible` | - | Generic OpenAI-compatible interface |
| SiliconFlow | `siliconflow` | `https://api.siliconflow.cn/v1` | Domestic AI platform |
| DeepSeek | `deepseek` | `https://api.deepseek.com` | DeepSeek models |
| Zhipu | `zhipu` | `https://open.bigmodel.cn/api/paas/v4` | Zhipu AI platform |
| Xiaomi MiMo | `mimo` | `https://api.xiaomimimo.com` | Xiaomi AI model |
| Ollama | `ollama` | `http://localhost:11434/v1` | Local deployment, no API Key required |
| vLLM | `vllm` | `http://localhost:8000/v1` | Local deployment, no API Key required |

## Installation

```bash
go get github.com/mr-dadong/openai_adk
```

## Quick Start

### Basic Usage

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

	// Create DeepSeek model
	model, err := openai_adk.NewModel(ctx, openai_adk.ClientConfig{
		Name:      "deepseek-chat",
		Provider:  openai_adk.ProviderDeepseek,
		ModelName: "deepseek-chat",
		APIKey:    os.Getenv("DEEPSEEK_API_KEY"),
	})
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	fmt.Printf("Model name: %s\n", model.Name())
}
```

### Using SiliconFlow

```go
model, err := openai_adk.NewModel(ctx, openai_adk.ClientConfig{
	Name:      "siliconflow-deepseek",
	Provider:  openai_adk.ProviderSiliconFlow,
	ModelName: "deepseek-ai/DeepSeek-V3.2",
	APIKey:    os.Getenv("SILICONFLOW_API_KEY"),
})
```

### Using Local Ollama

```go
// Local Ollama deployment requires no API Key
model, err := openai_adk.NewModel(ctx, openai_adk.ClientConfig{
	Name:      "ollama-llama3",
	Provider:  openai_adk.ProviderOllama,
	ModelName: "llama3",
	BaseURL:   "http://localhost:11434/v1", // Optional, has default value
})
```

### Using Custom OpenAI-Compatible Interface

```go
model, err := openai_adk.NewModel(ctx, openai_adk.ClientConfig{
	Name:      "custom-model",
	Provider:  openai_adk.ProviderOpenAICompatible,
	ModelName: "gpt-4",
	BaseURL:   "https://your-api-endpoint.com/v1",
	APIKey:    os.Getenv("CUSTOM_API_KEY"),
})
```

## Integration with ADK-Go

This package implements the `adkmodel.LLM` interface and can be directly integrated with the Google ADK-Go framework. Here's a complete example:

### Install ADK-Go

```bash
go get google.golang.org/adk
```

### Create Agent with OpenAI-Compatible Model

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

// Define tool input/output structures
type ReadFileInput struct {
	Path string `json:"path" description:"File path"`
}

type ReadFileOutput struct {
	Content string `json:"content"`
}

func main() {
	ctx := context.Background()

	// 1. Create OpenAI-compatible model (using DeepSeek as example)
	model, err := openai_adk.NewModel(ctx, openai_adk.ClientConfig{
		Name:      "deepseek-chat",
		Provider:  openai_adk.ProviderDeepseek,
		ModelName: "deepseek-chat",
		APIKey:    os.Getenv("DEEPSEEK_API_KEY"),
	})
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	// 2. Create tool
	readFileTool, err := functiontool.New(functiontool.Config{
		Name:        "read_file",
		Description: "Read file content at specified path",
		Handler: func(ctx tool.Context, input ReadFileInput) (ReadFileOutput, error) {
			content, err := os.ReadFile(input.Path)
			if err != nil {
				return ReadFileOutput{}, err
			}
			return ReadFileOutput{Content: string(content)}, nil
		},
	})
	if err != nil {
		log.Fatalf("Failed to create tool: %v", err)
	}

	// 3. Create Agent
	agent, err := llmagent.New(llmagent.Config{
		Name:        "file_reader_agent",
		Model:       model,
		Description: "An assistant that can read files",
		Instruction: `You are a file assistant. When users request to read a file, use the read_file tool to get the file content,
then display the content to the user. If the file doesn't exist, inform the user honestly.`,
		Tools: []tool.Tool{readFileTool},
	})
	if err != nil {
		log.Fatalf("Failed to create Agent: %v", err)
	}

	// 4. Use Agent (this is just an example, actual usage requires a launcher)
	log.Printf("Agent created successfully: %s\n", agent.Name())
}
```

### Agent Example with SiliconFlow

```go
// Create SiliconFlow model
model, err := openai_adk.NewModel(ctx, openai_adk.ClientConfig{
	Name:      "siliconflow-deepseek",
	Provider:  openai_adk.ProviderSiliconFlow,
	ModelName: "deepseek-ai/DeepSeek-V3.2",
	APIKey:    os.Getenv("SILICONFLOW_API_KEY"),
})

// Use this model when creating Agent
agent, err := llmagent.New(llmagent.Config{
	Name:  "my_agent",
	Model: model,
	// ... other configurations
})
```

### Agent Example with Local Ollama

```go
// Create local Ollama model (no API Key required)
model, err := openai_adk.NewModel(ctx, openai_adk.ClientConfig{
	Name:      "ollama-agent",
	Provider:  openai_adk.ProviderOllama,
	ModelName: "llama3",
})

// Create Agent
agent, err := llmagent.New(llmagent.Config{
	Name:  "local_agent",
	Model: model,
	// ... other configurations
})
```

## Configuration Options

### ClientConfig Structure

```go
type ClientConfig struct {
	Name         string // Model display name, used in error messages
	Provider     string // Provider identifier: gemini, openai_compatible, siliconflow, etc.
	ModelName    string // Model identifier, e.g., gpt-4, deepseek-chat
	BaseURL      string // API endpoint address
	APIKey       string // API key
	APIKeyEnv    string // API key environment variable name, used for error prompts
	ToolsEnabled *bool  // Whether to support tool calling, default true
	Debug        bool   // Whether to enable request/response debug logging
}
```

### Debug Mode

Enable debug mode to view detailed request and response information:

```go
model, err := openai_adk.NewModel(ctx, openai_adk.ClientConfig{
	Name:      "debug-model",
	Provider:  openai_adk.ProviderDeepseek,
	ModelName: "deepseek-chat",
	APIKey:    os.Getenv("DEEPSEEK_API_KEY"),
	Debug:     true, // Enable debug logging
})
```

Debug logs are output to standard error (stderr) and include:
- Complete request JSON
- Complete response JSON
- Token usage statistics

## Error Handling

The package returns clear error messages:

- **Missing API Key**: `api_key is required by model "xxx"`
- **Unsupported Provider**: `unsupported model provider "xxx"`
- **Missing BaseURL**: `base_url is required by openai_compatible model "xxx"`
- **HTTP Errors**: Include status code and error details
- **API Errors**: Include error type and message

## Retry Mechanism

The package has a built-in automatic retry mechanism:

- **Maximum Retry Attempts**: 3 times
- **Retry Delay**: Exponential backoff (1s, 2s, 4s), maximum 30s
- **Retryable Errors**: Network errors, 429 rate limiting, 5xx server errors
- **Retry-After Support**: Automatically parses Retry-After value from response headers

## Example Code

For more examples, please refer to the `openai_compatible_test.go` file, which includes:

- Basic text generation
- Streaming response handling
- Tool calling
- Multimodal content (images)
- Message history handling

## License

MIT License