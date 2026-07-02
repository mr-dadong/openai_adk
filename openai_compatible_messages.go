package openai_adk

import (
	"encoding/base64"
	"encoding/json"
	"strings"

	adkmodel "google.golang.org/adk/model"
	"google.golang.org/genai"
)

// buildMessages 将 ADK 标准 LLMRequest 转换为 OpenAI 格式的消息列表。
func buildMessages(req *adkmodel.LLMRequest) []chatMessage {
	var messages []chatMessage
	// 系统指令作为第一条 system 消息
	if req != nil && req.Config != nil && req.Config.SystemInstruction != nil {
		if text := contentText(req.Config.SystemInstruction); text != "" {
			messages = append(messages, chatMessage{Role: "system", Content: text})
		}
	}
	if req == nil {
		return messages
	}
	// 遍历对话历史；已完成的历史工具协议会在下方过滤，避免旧 tool_calls 污染新请求。
	toolProtocolStart := activeToolProtocolStart(req.Contents)
	for i, content := range req.Contents {
		if content == nil {
			continue
		}
		// 已被后续模型文本消费过的历史工具协议不再按 tool_calls/tool 回放，避免兼容接口拒绝旧工具链路。
		if i < toolProtocolStart && hasToolParts(content) {
			continue
		}
		messages = append(messages, contentMessages(content)...)
	}
	return messages
}

// activeToolProtocolStart 返回对话历史中最后一个 Model 文本输出的索引。
func activeToolProtocolStart(contents []*genai.Content) int {
	lastModelText := -1
	for i, content := range contents {
		if content == nil || content.Role != genai.RoleModel {
			continue
		}
		for _, part := range content.Parts {
			if strings.TrimSpace(part.Text) != "" {
				lastModelText = i
				break
			}
		}
	}
	if lastModelText < 0 {
		return 0
	}
	return lastModelText + 1
}

// hasToolParts 检查内容是否包含工具调用部分。
func hasToolParts(content *genai.Content) bool {
	if content == nil {
		return false
	}
	for _, part := range content.Parts {
		if part.FunctionCall != nil || part.FunctionResponse != nil {
			return true
		}
	}
	return false
}

// contentMessages 将一条 ADK Content 转为一条或多条 OpenAI 消息。
func contentMessages(content *genai.Content) []chatMessage {
	var text strings.Builder
	var imageParts []chatContentPart
	var calls []toolCall
	var messages []chatMessage
	for _, part := range content.Parts {
		switch {
		case part.Text != "":
			text.WriteString(part.Text)
		case part.InlineData != nil:
			if strings.HasPrefix(strings.ToLower(part.InlineData.MIMEType), "image/") {
				// OpenAI 兼容视觉接口使用 image_url，内联图片转为 data URL 传递。
				imageParts = append(imageParts, chatContentPart{
					Type: "image_url",
					ImageURL: &chatImageURL{
						URL:    "data:" + part.InlineData.MIMEType + ";base64," + base64.StdEncoding.EncodeToString(part.InlineData.Data),
						Detail: "auto",
					},
				})
			}
		case part.FunctionCall != nil:
			calls = append(calls, functionCallMessage(part.FunctionCall))
		case part.FunctionResponse != nil:
			// OpenAI 要求工具返回使用独立的 tool 角色消息
			messages = append(messages, functionResponseMessage(part.FunctionResponse))
		}
	}
	if text.Len() > 0 || len(imageParts) > 0 || len(calls) > 0 {
		contentValue := any(text.String())
		if len(imageParts) > 0 {
			parts := make([]chatContentPart, 0, 1+len(imageParts))
			if text.Len() > 0 {
				parts = append(parts, chatContentPart{Type: "text", Text: text.String()})
			}
			parts = append(parts, imageParts...)
			contentValue = parts
		}
		messages = append([]chatMessage{{
			Role:      openAIRole(content.Role),
			Content:   contentValue,
			ToolCalls: calls,
		}}, messages...)
	}
	return messages
}

// functionCallMessage 将 ADK FunctionCall 转为 OpenAI tool_call。
func functionCallMessage(call *genai.FunctionCall) toolCall {
	args, _ := json.Marshal(call.Args)
	return toolCall{
		ID:   call.ID,
		Type: "function",
		Function: toolCallFunction{
			Name:      call.Name,
			Arguments: string(args),
		},
	}
}

// functionResponseMessage 将 ADK FunctionResponse 转为 OpenAI tool 消息。
func functionResponseMessage(resp *genai.FunctionResponse) chatMessage {
	data, _ := json.Marshal(resp.Response)
	return chatMessage{
		Role:       "tool",
		Content:    string(data),
		ToolCallID: resp.ID,
		Name:       resp.Name,
	}
}

// buildTools 从 ADK 请求配置中提取函数工具声明。
func buildTools(req *adkmodel.LLMRequest) []chatTool {
	if req == nil || req.Config == nil {
		return nil
	}
	var tools []chatTool
	for _, t := range req.Config.Tools {
		for _, decl := range t.FunctionDeclarations {
			if decl == nil {
				continue
			}
			tools = append(tools, chatTool{
				Type: "function",
				Function: chatFunction{
					Name:        decl.Name,
					Description: decl.Description,
					Parameters:  decl.ParametersJsonSchema,
				},
			})
		}
	}
	return tools
}

// contentText 从 genai.Content 中提取文本内容。
func contentText(content *genai.Content) string {
	if content == nil {
		return ""
	}
	var b strings.Builder
	for _, part := range content.Parts {
		switch {
		case part.Text != "":
			b.WriteString(part.Text) // 普通文本
		case part.FunctionResponse != nil:
			// 函数调用结果序列化为 JSON 文本
			data, err := json.Marshal(part.FunctionResponse.Response)
			if err == nil {
				b.Write(data)
			}
		}
	}
	return b.String()
}

// openAIRole 将 genai 角色常量映射为 OpenAI 角色字符串。
func openAIRole(role string) string {
	switch role {
	case genai.RoleModel:
		return "assistant"
	case "system":
		return "system"
	default:
		return "user"
	}
}
