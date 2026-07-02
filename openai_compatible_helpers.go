package openai_adk

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"sort"
	"strings"
	"time"

	adkmodel "google.golang.org/adk/v2/model"
	"google.golang.org/genai"
)

// textResponse 构造仅含文本的 ADK 标准 LLMResponse。
func textResponse(text string, partial bool) *adkmodel.LLMResponse {
	return messageResponse(text, "", nil, partial)
}

// thinkingTextResponse 构造同时包含思考内容和正文的增量 LLMResponse。
// reasoning 非空时生成 Thought=true 的 Part，text 非空时生成普通文本 Part。
func thinkingTextResponse(reasoning string, text string, partial bool) *adkmodel.LLMResponse {
	return messageResponse(text, reasoning, nil, partial)
}

// messageResponse 构造 ADK 标准 LLMResponse，支持思考内容、文本和工具调用。
func messageResponse(text string, reasoning string, calls []toolCall, partial bool) *adkmodel.LLMResponse {
	parts := make([]*genai.Part, 0, 2+len(calls))
	// 思考内容放在最前面，与 Gemini 原生 Thought Part 一致
	if reasoning != "" {
		parts = append(parts, &genai.Part{Text: reasoning, Thought: true})
	}
	if text != "" {
		parts = append(parts, &genai.Part{Text: text})
	}
	for _, call := range calls {
		args := map[string]any{}
		if strings.TrimSpace(call.Function.Arguments) != "" {
			_ = json.Unmarshal([]byte(call.Function.Arguments), &args)
		}
		parts = append(parts, &genai.Part{
			FunctionCall: &genai.FunctionCall{
				ID:   call.ID,
				Name: call.Function.Name,
				Args: args,
			},
		})
	}
	if len(parts) == 0 {
		parts = append(parts, &genai.Part{Text: ""})
	}
	return &adkmodel.LLMResponse{
		Content:      &genai.Content{Role: genai.RoleModel, Parts: parts},
		Partial:      partial,
		TurnComplete: !partial,
	}
}

// buildUsageMetadata 将 OpenAI 格式的 usage 转换为 ADK 格式的 UsageMetadata。
// OpenAI 格式：prompt_tokens, completion_tokens, total_tokens, reasoning_tokens。
// ADK 格式：PromptTokenCount, CandidatesTokenCount, TotalTokenCount, ThoughtsTokenCount。
func buildUsageMetadata(u *usage) *genai.GenerateContentResponseUsageMetadata {
	if u == nil {
		return nil
	}
	return &genai.GenerateContentResponseUsageMetadata{
		PromptTokenCount:     u.PromptTokens,
		CandidatesTokenCount: u.CompletionTokens,
		TotalTokenCount:      u.TotalTokens,
		ThoughtsTokenCount:   u.ReasoningTokens,
	}
}

// mapFinishReason 将 OpenAI 的 finish_reason 映射为 ADK 的 FinishReason。
// OpenAI 格式：stop、length、tool_calls、content_filter。
// ADK 格式：STOP、MAX_TOKENS、SAFETY 等。
func mapFinishReason(reason string) genai.FinishReason {
	switch reason {
	case "stop":
		return genai.FinishReasonStop
	case "length":
		return genai.FinishReasonMaxTokens
	case "tool_calls":
		return genai.FinishReasonStop // 工具调用视为正常停止
	case "content_filter":
		return genai.FinishReasonSafety
	default:
		return genai.FinishReasonUnspecified
	}
}

// mergeToolCall 合并 OpenAI 流式工具调用片段。
func mergeToolCall(calls map[int]*toolCall, next toolCall) {
	cur := calls[next.Index]
	if cur == nil {
		copied := next
		calls[next.Index] = &copied
		return
	}
	if next.ID != "" {
		cur.ID = next.ID
	}
	if next.Type != "" {
		cur.Type = next.Type
	}
	if next.Function.Name != "" {
		cur.Function.Name = next.Function.Name
	}
	if next.Function.Arguments != "" {
		cur.Function.Arguments += next.Function.Arguments
	}
}

// orderedToolCalls 按流式 index 还原工具调用顺序。
func orderedToolCalls(calls map[int]*toolCall) []toolCall {
	if len(calls) == 0 {
		return nil
	}
	indexes := make([]int, 0, len(calls))
	for idx := range calls {
		indexes = append(indexes, idx)
	}
	sort.Ints(indexes)
	out := make([]toolCall, 0, len(indexes))
	for _, idx := range indexes {
		out = append(out, *calls[idx])
	}
	return out
}

// endpointURL 规范化 BaseURL，自动补全 /chat/completions 路径。
func endpointURL(baseURL string) string {
	baseURL = strings.TrimRight(strings.TrimSpace(baseURL), "/")
	if strings.HasSuffix(baseURL, "/chat/completions") {
		return baseURL // 已经是完整路径
	}
	if strings.HasSuffix(baseURL, "/v1") {
		return baseURL + "/chat/completions" // v1 路径下补全
	}
	return baseURL + "/v1/chat/completions" // 默认追加完整路径
}

// responseError 从 HTTP 错误响应中提取错误信息。
func responseError(resp *http.Response) error {
	// 限制读取 64KB，防止异常大的错误响应
	data, _ := io.ReadAll(io.LimitReader(resp.Body, 64*1024))
	var decoded chatResponse
	// 尝试 JSON 解码获取 API 层面的结构化错误
	if err := json.Unmarshal(data, &decoded); err == nil && decoded.Error != nil {
		return decoded.Error
	}
	// 回退：返回包含状态码和原始文本的错误
	return fmt.Errorf("openai_compatible request failed: %s: %s", resp.Status, strings.TrimSpace(string(data)))
}

// logRequest 记录请求体到标准错误输出，用于调试。
func logRequest(data []byte) {
	var pretty bytes.Buffer
	if err := json.Indent(&pretty, data, "", "  "); err != nil {
		pretty.Write(data)
	}
	fmt.Fprintf(os.Stderr, "\n[DEBUG] OpenAI Request:\n%s\n", pretty.String())
}

// logResponse 记录响应体到标准错误输出，用于调试。
func logResponse(data []byte) {
	var pretty bytes.Buffer
	if err := json.Indent(&pretty, data, "", "  "); err != nil {
		pretty.Write(data)
	}
	fmt.Fprintf(os.Stderr, "\n[DEBUG] OpenAI Response:\n%s\n", pretty.String())
}

// isRetryableError 判断错误是否可重试。
// 可重试的错误包括：网络错误、429 限流、5xx 服务器错误。
func isRetryableError(err error, statusCode int) bool {
	// 网络错误（非 HTTP 错误）
	if err != nil && statusCode == 0 {
		return true
	}
	// 429 限流
	if statusCode == 429 {
		return true
	}
	// 5xx 服务器错误
	if statusCode >= 500 && statusCode < 600 {
		return true
	}
	return false
}

// retryDelay 计算重试延迟时间（指数退避）。
func retryDelay(attempt int) time.Duration {
	delay := retryBaseDelay * time.Duration(1<<uint(attempt))
	if delay > retryMaxDelay {
		delay = retryMaxDelay
	}
	return delay
}

// extractRetryAfter 从响应头中提取 Retry-After 值。
func extractRetryAfter(resp *http.Response) time.Duration {
	if resp == nil {
		return 0
	}
	retryAfter := resp.Header.Get("Retry-After")
	if retryAfter == "" {
		return 0
	}
	// 尝试解析为秒数
	if seconds, err := time.ParseDuration(retryAfter + "s"); err == nil {
		return seconds
	}
	return 0
}
