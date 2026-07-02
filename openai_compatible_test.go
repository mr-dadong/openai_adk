package openai_adk

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	adkmodel "google.golang.org/adk/model"
	"google.golang.org/genai"
)

func TestOpenAICompatibleModelGenerateContent(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			t.Fatalf("Authorization header = %q", got)
		}
		fmt.Fprint(w, `{"choices":[{"message":{"content":"hello"}}]}`)
	}))
	defer server.Close()

	llm, err := newOpenAICompatibleModel(ClientConfig{
		Name:      "test",
		ModelName: "test-model",
		BaseURL:   server.URL,
		APIKey:    "test-key",
	})
	if err != nil {
		t.Fatal(err)
	}

	var got string
	for resp, err := range llm.GenerateContent(context.Background(), textRequest("hi"), false) {
		if err != nil {
			t.Fatal(err)
		}
		got += contentText(resp.Content)
	}
	if got != "hello" {
		t.Fatalf("response text = %q", got)
	}
}

func TestOpenAICompatibleModelGenerateContentStream(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"he\"}}]}\n\n")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"llo\"}}]}\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer server.Close()

	llm, err := newOpenAICompatibleModel(ClientConfig{
		Name:      "test",
		ModelName: "test-model",
		BaseURL:   server.URL,
		APIKey:    "test-key",
	})
	if err != nil {
		t.Fatal(err)
	}

	var got string
	var partials int
	for resp, err := range llm.GenerateContent(context.Background(), textRequest("hi"), true) {
		if err != nil {
			t.Fatal(err)
		}
		if resp.Partial {
			partials++
			got += contentText(resp.Content)
		}
	}
	if got != "hello" {
		t.Fatalf("stream partial text = %q", got)
	}
	if partials != 2 {
		t.Fatalf("partials = %d", partials)
	}
}

func TestSiliconFlowProviderUsesDefaultBaseURL(t *testing.T) {
	cfg := ClientConfig{
		Name:      "deepseek",
		Provider:  ProviderSiliconFlow,
		ModelName: "deepseek-ai/DeepSeek-V3.2",
		APIKey:    "test-key",
	}

	llm, err := NewModel(context.Background(), cfg)
	if err != nil {
		t.Fatal(err)
	}

	got := llm.(*openAICompatibleModel).baseURL
	want := "https://api.siliconflow.cn/v1/chat/completions"
	if got != want {
		t.Fatalf("baseURL = %q, want %q", got, want)
	}
}

func TestOpenAICompatibleModelSendsTools(t *testing.T) {
	var got chatRequest
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Fatal(err)
		}
		fmt.Fprint(w, `{"choices":[{"message":{"content":"ok"}}]}`)
	}))
	defer server.Close()

	llm, err := newOpenAICompatibleModel(ClientConfig{
		Name:      "test",
		ModelName: "test-model",
		BaseURL:   server.URL,
		APIKey:    "test-key",
	})
	if err != nil {
		t.Fatal(err)
	}

	for _, err := range llm.GenerateContent(context.Background(), toolRequest(), false) {
		if err != nil {
			t.Fatal(err)
		}
	}

	if len(got.Tools) != 1 {
		t.Fatalf("tools = %d, want 1", len(got.Tools))
	}
	if got.ToolChoice != "auto" {
		t.Fatalf("tool_choice = %q, want auto", got.ToolChoice)
	}
	if got.Tools[0].Function.Name != "read_file" {
		t.Fatalf("tool name = %q", got.Tools[0].Function.Name)
	}
}

func TestOpenAICompatibleModelReceivesToolCall(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{"choices":[{"message":{"tool_calls":[{"id":"call_1","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"go.mod\"}"}}]}}]}`)
	}))
	defer server.Close()

	llm, err := newOpenAICompatibleModel(ClientConfig{
		Name:      "test",
		ModelName: "test-model",
		BaseURL:   server.URL,
		APIKey:    "test-key",
	})
	if err != nil {
		t.Fatal(err)
	}

	var call *genai.FunctionCall
	for resp, err := range llm.GenerateContent(context.Background(), toolRequest(), false) {
		if err != nil {
			t.Fatal(err)
		}
		for _, part := range resp.Content.Parts {
			if part.FunctionCall != nil {
				call = part.FunctionCall
			}
		}
	}
	if call == nil {
		t.Fatal("expected function call")
	}
	if call.ID != "call_1" || call.Name != "read_file" || call.Args["path"] != "go.mod" {
		t.Fatalf("function call = %#v", call)
	}
}

func TestBuildMessagesDropsCompletedHistoricalToolProtocol(t *testing.T) {
	req := &adkmodel.LLMRequest{
		Contents: []*genai.Content{
			genai.NewContentFromText("列文件", genai.RoleUser),
			toolCallContent("call_old", "list_files", map[string]any{"path": "."}),
			toolResponseContent("call_old", "list_files", map[string]any{"files": []string{"go.mod"}}),
			genai.NewContentFromText("已列出文件", genai.RoleModel),
			genai.NewContentFromText("ping", genai.RoleUser),
		},
	}

	messages := buildMessages(req)
	for _, message := range messages {
		if len(message.ToolCalls) > 0 || message.Role == "tool" {
			t.Fatalf("historical tool protocol leaked into messages: %#v", messages)
		}
	}
	if got := messages[len(messages)-1].Content; got != "ping" {
		t.Fatalf("last message = %q, want ping", got)
	}
}

func TestBuildMessagesKeepsActiveToolProtocol(t *testing.T) {
	req := &adkmodel.LLMRequest{
		Contents: []*genai.Content{
			genai.NewContentFromText("读取文件", genai.RoleUser),
			toolCallContent("call_active", "read_file", map[string]any{"path": "go.mod"}),
			toolResponseContent("call_active", "read_file", map[string]any{"content": "module xagent-cli"}),
		},
	}

	messages := buildMessages(req)
	if len(messages) != 3 {
		t.Fatalf("messages = %d, want 3: %#v", len(messages), messages)
	}
	if len(messages[1].ToolCalls) != 1 {
		t.Fatalf("active tool call not preserved: %#v", messages)
	}
	if messages[2].Role != "tool" || messages[2].ToolCallID != "call_active" {
		t.Fatalf("active tool response not preserved: %#v", messages[2])
	}
}

func TestBuildMessagesConvertsInlineImageToImageURL(t *testing.T) {
	req := &adkmodel.LLMRequest{
		Contents: []*genai.Content{{
			Role: genai.RoleUser,
			Parts: []*genai.Part{
				genai.NewPartFromText("这是什么"),
				genai.NewPartFromBytes([]byte("hello"), "image/png"),
			},
		}},
	}

	messages := buildMessages(req)
	if len(messages) != 1 {
		t.Fatalf("messages = %d, want 1", len(messages))
	}
	parts, ok := messages[0].Content.([]chatContentPart)
	if !ok {
		t.Fatalf("content type = %T, want []chatContentPart", messages[0].Content)
	}
	if len(parts) != 2 || parts[0].Type != "text" || parts[1].Type != "image_url" {
		t.Fatalf("parts = %#v", parts)
	}
	if got := parts[1].ImageURL.URL; got != "data:image/png;base64,aGVsbG8=" {
		t.Fatalf("image url = %q", got)
	}
}

func textRequest(text string) *adkmodel.LLMRequest {
	return &adkmodel.LLMRequest{
		Contents: []*genai.Content{
			genai.NewContentFromText(text, genai.RoleUser),
		},
	}
}

func toolCallContent(id, name string, args map[string]any) *genai.Content {
	// 测试直接构造 ADK 工具调用，覆盖持久化历史恢复后的协议转换行为。
	return &genai.Content{
		Role: genai.RoleModel,
		Parts: []*genai.Part{{
			FunctionCall: &genai.FunctionCall{
				ID:   id,
				Name: name,
				Args: args,
			},
		}},
	}
}

func toolResponseContent(id, name string, response map[string]any) *genai.Content {
	// 工具结果在 OpenAI 协议里必须紧跟对应 tool_call，历史清理不能影响当前轮工具结果。
	return &genai.Content{
		Role: genai.RoleUser,
		Parts: []*genai.Part{{
			FunctionResponse: &genai.FunctionResponse{
				ID:       id,
				Name:     name,
				Response: response,
			},
		}},
	}
}

func toolRequest() *adkmodel.LLMRequest {
	req := textRequest("read go.mod")
	req.Config = &genai.GenerateContentConfig{
		Tools: []*genai.Tool{{
			FunctionDeclarations: []*genai.FunctionDeclaration{{
				Name:        "read_file",
				Description: "Read a file",
				ParametersJsonSchema: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"path": map[string]any{"type": "string"},
					},
					"required": []string{"path"},
				},
			}},
		}},
	}
	return req
}
