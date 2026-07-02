package openai_adk

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"iter"
	"net/http"
	"os"
	"strings"
	"time"

	adkmodel "google.golang.org/adk/v2/model"
)

// defaultHTTPTimeout 是 HTTP 客户端的默认超时时间（2分钟）。
// 对于需要长时间推理的模型，这个值需要在后续迭代中考虑配置化。
const defaultHTTPTimeout = 2 * time.Minute

// 重试配置
const (
	maxRetries     = 3                // 最大重试次数
	retryBaseDelay = 1 * time.Second  // 重试基础延迟
	retryMaxDelay  = 30 * time.Second // 重试最大延迟
)

// isLocalProvider 判断供应商是否为本地部署模型，本地部署不需要 API Key。
func isLocalProvider(provider string) bool {
	switch provider {
	case ProviderOllama, ProviderVLLM:
		return true
	}
	return false
}

// newOpenAICompatibleModel 创建 OpenAI 兼容协议的模型客户端。
func newOpenAICompatibleModel(cfg ClientConfig) (adkmodel.LLM, error) {
	// Ollama 和 vLLM 本地部署无需 API Key，其他供应商必须提供
	if strings.TrimSpace(cfg.APIKey) == "" && !isLocalProvider(cfg.Provider) {
		return nil, missingAPIKeyError(cfg)
	}
	// BaseURL 为必填，无法像 Gemini 那样默认
	if strings.TrimSpace(cfg.BaseURL) == "" {
		return nil, fmt.Errorf("base_url is required by openai_compatible model %q", cfg.Name)
	}
	toolsEnabled := true
	if cfg.ToolsEnabled != nil {
		toolsEnabled = *cfg.ToolsEnabled
	}
	return &openAICompatibleModel{
		name:         cfg.ModelName,
		baseURL:      endpointURL(cfg.BaseURL), // 自动补全 /chat/completions 路径
		apiKey:       cfg.APIKey,
		toolsEnabled: toolsEnabled,
		debug:        cfg.Debug,
		client:       &http.Client{Timeout: defaultHTTPTimeout}, // 设置 2 分钟超时
	}, nil
}

// Name 返回模型的默认名称。
// 实现 adkmodel.LLM 接口。
func (m *openAICompatibleModel) Name() string {
	return m.name
}

// GenerateContent 向 OpenAI 兼容 API 发送请求并生成内容。
func (m *openAICompatibleModel) GenerateContent(ctx context.Context, req *adkmodel.LLMRequest, stream bool) iter.Seq2[*adkmodel.LLMResponse, error] {
	return func(yield func(*adkmodel.LLMResponse, error) bool) {
		// 构建请求体：优先使用请求中指定的模型名，否则用默认名
		body := chatRequest{
			Model:    m.modelName(req),
			Messages: buildMessages(req),
			Stream:   stream,
		}
		// 流式请求时启用 usage 统计（OpenAI API 要求显式声明）
		if stream {
			body.StreamOptions = &streamOptions{IncludeUsage: true}
		}
		// 仅在模型原生支持工具时发送 tools 声明；不再支持文本模拟工具调用。
		if m.toolsEnabled {
			body.Tools = buildTools(req)
			if len(body.Tools) > 0 {
				body.ToolChoice = "auto"
			}
		}
		// 填充可选的采样参数
		if req.Config != nil {
			body.Temperature = req.Config.Temperature
			body.TopP = req.Config.TopP
			body.MaxTokens = req.Config.MaxOutputTokens
			body.Stop = req.Config.StopSequences
		}

		// 序列化请求体为 JSON
		data, err := json.Marshal(body)
		if err != nil {
			yield(nil, err)
			return
		}

		// 调试模式：记录请求体
		if m.debug {
			logRequest(data)
		}

		// 重试循环
		for attempt := 0; attempt <= maxRetries; attempt++ {
			// 检查上下文是否已取消
			if ctx.Err() != nil {
				yield(nil, ctx.Err())
				return
			}

			// 构造 HTTP POST 请求（每次重试都需要新创建，因为 Body 已被读取）
			httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, m.baseURL, bytes.NewReader(data))
			if err != nil {
				yield(nil, err)
				return
			}
			httpReq.Header.Set("Content-Type", "application/json")
			httpReq.Header.Set("Authorization", "Bearer "+m.apiKey) // Bearer Token 认证

			// 执行 HTTP 请求
			resp, err := m.client.Do(httpReq)
			if err != nil {
				// 网络错误：可重试
				if isRetryableError(err, 0) && attempt < maxRetries {
					delay := retryDelay(attempt)
					if m.debug {
						fmt.Fprintf(os.Stderr, "[DEBUG] Network error, retrying in %v (attempt %d/%d): %v\n", delay, attempt+1, maxRetries, err)
					}
					time.Sleep(delay)
					continue
				}
				yield(nil, err)
				return
			}

			// 检查 HTTP 状态码
			if resp.StatusCode < 200 || resp.StatusCode >= 300 {
				// 可重试的 HTTP 错误
				if isRetryableError(nil, resp.StatusCode) && attempt < maxRetries {
					delay := extractRetryAfter(resp)
					if delay == 0 {
						delay = retryDelay(attempt)
					}
					resp.Body.Close()
					if m.debug {
						fmt.Fprintf(os.Stderr, "[DEBUG] HTTP %d, retrying in %v (attempt %d/%d)\n", resp.StatusCode, delay, attempt+1, maxRetries)
					}
					time.Sleep(delay)
					continue
				}
				// 不可重试的 HTTP 错误
				err = responseError(resp)
				resp.Body.Close()
				yield(nil, err)
				return
			}

			// 成功：处理响应
			defer resp.Body.Close()
			if stream {
				m.yieldStream(resp.Body, yield)
				return
			}
			m.yieldResponse(resp.Body, yield)
			return
		}
	}
}

// modelName 返回实际使用的模型名称。
// 优先使用 LLMRequest 中指定的模型，未指定时回退到客户端默认名称。
func (m *openAICompatibleModel) modelName(req *adkmodel.LLMRequest) string {
	if req != nil && req.Model != "" {
		return req.Model
	}
	return m.name
}

// yieldResponse 处理非流式响应：整块 JSON 解码并作为单个事件 yield。
func (m *openAICompatibleModel) yieldResponse(body io.Reader, yield func(*adkmodel.LLMResponse, error) bool) {
	// 调试模式：读取并记录响应体
	if m.debug {
		data, err := io.ReadAll(body)
		if err != nil {
			yield(nil, err)
			return
		}
		logResponse(data)
		body = bytes.NewReader(data)
	}

	var decoded chatResponse
	if err := json.NewDecoder(body).Decode(&decoded); err != nil {
		yield(nil, err)
		return
	}
	// 检查 API 层面错误
	if decoded.Error != nil {
		yield(nil, decoded.Error)
		return
	}
	// 无候选回复时返回错误
	if len(decoded.Choices) == 0 {
		yield(nil, fmt.Errorf("openai_compatible response has no choices"))
		return
	}
	// 调试模式：记录 usage 信息
	if m.debug && decoded.Usage != nil {
		fmt.Fprintf(os.Stderr, "[DEBUG] Usage: prompt=%d, completion=%d, total=%d\n",
			decoded.Usage.PromptTokens, decoded.Usage.CompletionTokens, decoded.Usage.TotalTokens)
	}
	// 产出完整响应（Partial=false），包含文本或工具调用
	resp := messageResponse(decoded.Choices[0].Message.Content, decoded.Choices[0].Message.ReasoningContent, decoded.Choices[0].Message.ToolCalls, false)
	resp.UsageMetadata = buildUsageMetadata(decoded.Usage)
	resp.FinishReason = mapFinishReason(decoded.Choices[0].FinishReason)
	resp.ModelVersion = decoded.Model
	yield(resp, nil)
}

// yieldStream 处理 SSE 流式响应：逐行解析并实时 yield 增量文本片段。
//
// SSE 协议格式说明：
//   - 服务器持续推送 data: 行，每行是一个 JSON 片段
//   - 空行或以 : 开头的行（SSE 注释）会被跳过
//   - data: [DONE] 表示流结束
//   - 每个 delta 内容作为 Partial=true 的事件产出
//   - 流结束时产出完整累积文本作为 Partial=false 的事件
//
// 缓冲区配置：初始 64KB，最大 1MB，防止单行过长导致 Scanner 报错。
//
// 参数：
//   - body: HTTP 响应体（SSE 流）
//   - yield: 迭代器回调
func (m *openAICompatibleModel) yieldStream(body io.Reader, yield func(*adkmodel.LLMResponse, error) bool) {
	// 调试模式：记录流式响应开始
	if m.debug {
		fmt.Fprintf(os.Stderr, "\n[DEBUG] OpenAI Stream Response: started\n")
	}

	// 使用缓冲扫描器逐行读取 SSE 流
	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024) // 初始 64KB，最大 1MB

	// full 累积所有增量文本，用于流结束时产出完整内容
	var full strings.Builder
	toolCalls := map[int]*toolCall{}
	var lastUsage *usage        // 累积流式响应中的 usage 信息
	var lastFinishReason string // 累积流式响应中的 finish_reason
	var modelVersion string     // 累积流式响应中的 model 信息
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		// 跳过空行和 SSE 注释行（以 : 开头）
		if line == "" || strings.HasPrefix(line, ":") {
			continue
		}
		// 仅处理 data: 前缀的行
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		// 提取 JSON 负载
		payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		// [DONE] 标记流结束，产出完整累积文本
		if payload == "[DONE]" {
			// 调试模式：记录流式响应结束和 usage 信息
			if m.debug {
				fmt.Fprintf(os.Stderr, "[DEBUG] OpenAI Stream Response: completed\n")
				if lastUsage != nil {
					fmt.Fprintf(os.Stderr, "[DEBUG] Usage: prompt=%d, completion=%d, total=%d\n",
						lastUsage.PromptTokens, lastUsage.CompletionTokens, lastUsage.TotalTokens)
				}
			}
			resp := messageResponse(full.String(), "", orderedToolCalls(toolCalls), false)
			resp.UsageMetadata = buildUsageMetadata(lastUsage)
			resp.FinishReason = mapFinishReason(lastFinishReason)
			resp.ModelVersion = modelVersion
			yield(resp, nil)
			return
		}

		// 解析流式响应块
		var decoded streamResponse
		if err := json.Unmarshal([]byte(payload), &decoded); err != nil {
			yield(nil, err)
			return
		}
		if decoded.Error != nil {
			yield(nil, decoded.Error)
			return
		}
		// 累积 model 信息（通常在第一个 chunk 中）
		if decoded.Model != "" {
			modelVersion = decoded.Model
		}
		// 累积 usage 信息（通常在最后一个 chunk 中）
		if decoded.Usage != nil {
			lastUsage = decoded.Usage
		}
		if len(decoded.Choices) == 0 {
			continue // 无候选则继续等待下一个事件
		}
		// 累积 finish_reason（通常在最后一个 chunk 中）
		if decoded.Choices[0].FinishReason != "" {
			lastFinishReason = decoded.Choices[0].FinishReason
		}
		// 流式工具调用会分多段返回，需要按 index 累积函数名和参数 JSON
		for _, call := range decoded.Choices[0].Delta.ToolCalls {
			mergeToolCall(toolCalls, call)
		}
		text := decoded.Choices[0].Delta.Content
		reasoning := decoded.Choices[0].Delta.ReasoningContent
		if text == "" && reasoning == "" {
			continue // 空增量跳过
		}
		// 累积正文文本并产出增量事件
		if text != "" {
			full.WriteString(text)
		}
		// 思考内容作为增量事件产出，用 Thought 标记区分
		if !yield(thinkingTextResponse(reasoning, text, true), nil) {
			return
		}
	}
	// Scanner 扫描过程中的错误
	if err := scanner.Err(); err != nil {
		yield(nil, err)
		return
	}
	// 正常读完所有行但未收到 [DONE]（某些实现可能不发送），补发完整内容
	if full.Len() > 0 || len(toolCalls) > 0 {
		// 调试模式：记录流式响应结束（无 [DONE]）
		if m.debug {
			fmt.Fprintf(os.Stderr, "[DEBUG] OpenAI Stream Response: completed (no [DONE])\n")
		}
		resp := messageResponse(full.String(), "", orderedToolCalls(toolCalls), false)
		resp.UsageMetadata = buildUsageMetadata(lastUsage)
		resp.FinishReason = mapFinishReason(lastFinishReason)
		resp.ModelVersion = modelVersion
		yield(resp, nil)
	}
}
