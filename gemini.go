package openai_adk

import (
	"context"
	"strings"

	adkmodel "google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/model/gemini"
	"google.golang.org/genai"
)

func newGeminiModel(ctx context.Context, cfg ClientConfig) (adkmodel.LLM, error) {
	if strings.TrimSpace(cfg.APIKey) == "" {
		return nil, missingAPIKeyError(cfg)
	}
	return gemini.NewModel(ctx, cfg.ModelName, &genai.ClientConfig{
		APIKey: cfg.APIKey,
	})
}
