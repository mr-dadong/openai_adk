package openai_adk

import (
	"context"
	"fmt"
	"strings"

	adkmodel "google.golang.org/adk/v2/model"
)

// ClientConfig 定义创建 LLM 客户端所需的配置参数。
// 用于解耦 provider 包与具体配置结构体的依赖，使其可被其他项目复用。
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

const (
	ProviderGemini           = "gemini"
	ProviderOpenAICompatible = "openai_compatible"
	ProviderSiliconFlow      = "siliconflow"
	ProviderDeepseek         = "deepseek"
	ProviderZhiPu            = "zhipu"
	ProviderXiaoMi           = "mimo"
	ProviderOllama           = "ollama"
	ProviderVLLM             = "vllm"
)

// defaultBaseURLs 维护各供应商的默认 API 端点地址。
// 仅当用户未显式配置 BaseURL 时使用，不在此表中的供应商需自行指定。
var defaultBaseURLs = map[string]string{
	ProviderSiliconFlow: "https://api.siliconflow.cn/v1",
	ProviderDeepseek:    "https://api.deepseek.com",
	ProviderZhiPu:       "https://open.bigmodel.cn/api/paas/v4",
	ProviderXiaoMi:      "https://api.xiaomimimo.com",
	ProviderOllama:      "http://localhost:11434/v1",
	ProviderVLLM:        "http://localhost:8000/v1",
}

func NewModel(ctx context.Context, cfg ClientConfig) (adkmodel.LLM, error) {
	provider := strings.ToLower(strings.TrimSpace(cfg.Provider))
	if provider == "" {
		provider = ProviderGemini
	}

	switch provider {
	case ProviderGemini:
		return newGeminiModel(ctx, cfg)
	case ProviderOpenAICompatible,
		ProviderSiliconFlow,
		ProviderDeepseek,
		ProviderZhiPu,
		ProviderXiaoMi,
		ProviderOllama,
		ProviderVLLM:
		if cfg.BaseURL == "" {
			if url, ok := defaultBaseURLs[provider]; ok {
				cfg.BaseURL = url
			}
		}
		return newOpenAICompatibleModel(cfg)
	default:
		return nil, fmt.Errorf("unsupported model provider %q", cfg.Provider)
	}
}
