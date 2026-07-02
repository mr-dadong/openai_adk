package openai_adk

import (
	"errors"
	"fmt"
)

func missingAPIKeyError(cfg ClientConfig) error {
	if cfg.APIKeyEnv != "" {
		return fmt.Errorf("api_key or %s is required by model %q", cfg.APIKeyEnv, cfg.Name)
	}
	if cfg.Name != "" {
		return fmt.Errorf("api_key is required by model %q", cfg.Name)
	}
	return errors.New("api_key is required by the selected model")
}
