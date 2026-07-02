package openai_adk

import (
	"fmt"
	"net/http"
)

// openAICompatibleModel 实现了 adkmodel.LLM 接口，用于对接 OpenAI 兼容的 API。
// 它通过 HTTP 请求将 ADK 标准请求转换为 OpenAI Chat Completions 格式，
// 并将响应转换回 ADK 标准格式，从而屏蔽底层 API 差异。
type openAICompatibleModel struct {
	name         string       // 默认模型名称（如 gpt-4o、deepseek-chat）
	baseURL      string       // API 端点地址（含 /chat/completions 后缀）
	apiKey       string       // API 认证密钥
	toolsEnabled bool         // 是否启用工具调用
	debug        bool         // 是否启用请求/响应日志
	client       *http.Client // HTTP 客户端（复用连接，控制超时）
}

// chatRequest 是 OpenAI Chat Completions API 的请求体结构。
// JSON 标签映射 OpenAI 官方字段名。
type chatRequest struct {
	Model         string         `json:"model"`                    // 模型名称
	Messages      []chatMessage  `json:"messages"`                 // 对话消息列表
	Tools         []chatTool     `json:"tools,omitempty"`          // 可供模型调用的工具声明
	ToolChoice    string         `json:"tool_choice,omitempty"`    // 工具选择策略；有工具时按 OpenAI 兼容协议显式使用 auto。
	Stream        bool           `json:"stream,omitempty"`         // 是否启用流式响应
	StreamOptions *streamOptions `json:"stream_options,omitempty"` // 流式选项（包含 usage 统计）
	Temperature   *float32       `json:"temperature,omitempty"`    // 采样温度（0-2）
	TopP          *float32       `json:"top_p,omitempty"`          // 核采样参数
	MaxTokens     int32          `json:"max_tokens,omitempty"`     // 最大输出 token 数
	Stop          []string       `json:"stop,omitempty"`           // 停止词列表
}

// streamOptions 控制流式响应的额外选项。
type streamOptions struct {
	IncludeUsage bool `json:"include_usage"` // 是否在最后一个 chunk 中返回 usage 统计
}

// chatMessage 是单条对话消息，包含角色和内容。
type chatMessage struct {
	Role       string     `json:"role"`                   // 角色：system、user、assistant、tool
	Content    any        `json:"content,omitempty"`      // 消息内容；视觉模型可为多模态数组
	ToolCalls  []toolCall `json:"tool_calls,omitempty"`   // assistant 发起的工具调用
	ToolCallID string     `json:"tool_call_id,omitempty"` // tool 响应对应的调用 ID
	Name       string     `json:"name,omitempty"`         // tool 响应对应的函数名
}

// chatContentPart 是 OpenAI 兼容接口的多模态消息片段。
type chatContentPart struct {
	Type     string        `json:"type"`                // text 或 image_url。
	Text     string        `json:"text,omitempty"`      // 文本片段。
	ImageURL *chatImageURL `json:"image_url,omitempty"` // 图片片段。
}

type chatImageURL struct {
	URL    string `json:"url"`              // data:image/...;base64,... 格式。
	Detail string `json:"detail,omitempty"` // 交给模型自动选择解析细节。
}

// chatTool 是 OpenAI Chat Completions 的工具声明结构。
type chatTool struct {
	Type     string       `json:"type"`     // 固定为 function
	Function chatFunction `json:"function"` // 函数声明
}

// chatFunction 描述一个可调用函数。
type chatFunction struct {
	Name        string `json:"name"`                  // 函数名
	Description string `json:"description,omitempty"` // 函数说明
	Parameters  any    `json:"parameters,omitempty"`  // JSON Schema 参数
}

// toolCall 表示模型返回的一次工具调用。
type toolCall struct {
	ID       string           `json:"id,omitempty"`    // 工具调用 ID
	Type     string           `json:"type,omitempty"`  // 固定为 function
	Function toolCallFunction `json:"function"`        // 函数名与参数
	Index    int              `json:"index,omitempty"` // 流式响应中的工具调用序号
}

// toolCallFunction 是工具调用的函数载荷。
type toolCallFunction struct {
	Name      string `json:"name,omitempty"`      // 函数名
	Arguments string `json:"arguments,omitempty"` // JSON 字符串形式的参数
}

// usage 表示 OpenAI API 返回的 token 使用统计。
// OpenAI 标准格式：prompt_tokens + completion_tokens = total_tokens。
type usage struct {
	PromptTokens     int32 `json:"prompt_tokens"`     // 输入 token 数
	CompletionTokens int32 `json:"completion_tokens"` // 输出 token 数
	TotalTokens      int32 `json:"total_tokens"`      // 总 token 数
	ReasoningTokens  int32 `json:"reasoning_tokens"`  // 思考/推理 token 数（DeepSeek、Qwen3 等）
}

// chatResponse 是 OpenAI Chat Completions 非流式 API 的响应体结构。
type chatResponse struct {
	Choices []struct {
		Message struct {
			Content          string     `json:"content"`              // 模型回复的文本
			ReasoningContent string     `json:"reasoning_content"`    // 思考/推理内容（DeepSeek、Qwen3、mimo 等）
			ToolCalls        []toolCall `json:"tool_calls,omitempty"` // 模型请求执行的工具
		} `json:"message"`
		FinishReason string `json:"finish_reason"` // 停止原因：stop、length、tool_calls、content_filter
	} `json:"choices"` // 候选回复列表
	Model string    `json:"model"`           // 实际使用的模型名称（可能与请求不同）
	Usage *usage    `json:"usage,omitempty"` // token 使用统计
	Error *apiError `json:"error,omitempty"` // API 错误信息
}

// streamResponse 是 OpenAI Chat Completions 流式（SSE）API 的响应块结构。
// 与 chatResponse 不同，流式响应使用 delta 而非 message，内容为增量片段。
type streamResponse struct {
	Choices []struct {
		Delta struct {
			Content          string     `json:"content"`              // 增量文本片段
			ReasoningContent string     `json:"reasoning_content"`    // 增量思考/推理片段（DeepSeek、Qwen3、mimo 等）
			ToolCalls        []toolCall `json:"tool_calls,omitempty"` // 增量工具调用片段
		} `json:"delta"`
		FinishReason string `json:"finish_reason,omitempty"` // 停止原因（通常在最后一个 chunk 中）
	} `json:"choices"` // 候选回复列表
	Model string    `json:"model,omitempty"` // 实际使用的模型名称（通常在第一个 chunk 中）
	Usage *usage    `json:"usage,omitempty"` // token 使用统计（通常在最后一个 chunk 中）
	Error *apiError `json:"error,omitempty"` // API 错误信息
}

// apiError 表示 OpenAI 兼容 API 返回的错误信息。
// 实现 error 接口，可直接作为 Go error 使用。
type apiError struct {
	Message string `json:"message"` // 错误描述
	Type    string `json:"type"`    // 错误类型（如 invalid_request_error）
	Code    any    `json:"code"`    // 错误码（可能为字符串或数字）
}

// Error 实现 error 接口，将 API 错误格式化为可读字符串。
func (e *apiError) Error() string {
	if e == nil {
		return ""
	}
	if e.Type == "" {
		return e.Message
	}
	return fmt.Sprintf("%s: %s", e.Type, e.Message)
}
