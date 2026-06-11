// Package config handles application configuration loading and validation.
package config

import "encoding/json"

// Config holds the complete application configuration.
type Config struct {
	APIKey                         string                   `json:"api_key"`
	Host                           string                   `json:"host"`
	Port                           int                      `json:"port"`
	HotReload                      bool                              `json:"hot_reload"`
	EnableStreamingScenarioRouting bool                              `json:"enable_streaming_scenario_routing"`
	RespectRequestedModel                  bool                              `json:"respect_requested_model"`
	EnableEffortScenarioRouting            bool                              `json:"enable_effort_scenario_routing"`
	RespectRequestedModelUseEffort         bool                              `json:"respect_requested_model_use_effort"`
	RespectRequestedModelUseContextThreshold bool                            `json:"respect_requested_model_use_context_threshold"`
	Models                                 map[string]ModelConfig            `json:"models"`
	Fallbacks                      map[string][]ModelConfig          `json:"fallbacks"`
	ModelOverrides                 map[string]ModelConfig            `json:"model_overrides"`
	ModelEffortOverrides           map[string]map[string]ModelConfig `json:"model_effort_overrides"`
	OpenCodeGo                     OpenCodeGoConfig                  `json:"opencode_go"`
	OpenCodeZen                    OpenCodeZenConfig                 `json:"opencode_zen"`
	AnthropicCompatible            AnthropicCompatibleConfig         `json:"anth_comp"`
	Logging                        LoggingConfig                     `json:"logging"`
}

// ModelConfig defines routing rules for a specific model.
type ModelConfig struct {
	Provider         string          `json:"provider"`
	ModelID          string          `json:"model_id"`
	Temperature      float64         `json:"temperature"`
	MaxTokens        int             `json:"max_tokens"`
	ContextThreshold int             `json:"context_threshold"`
	ReasoningEffort  string          `json:"reasoning_effort"`
	Thinking         json.RawMessage `json:"thinking,omitempty"`
}

// OpenCodeGoConfig holds the upstream OpenCode Go API settings.
type OpenCodeGoConfig struct {
	BaseURL          string `json:"base_url"`
	AnthropicBaseURL string `json:"anthropic_base_url"`
	TimeoutMs        int    `json:"timeout_ms"`
}

// OpenCodeZenConfig holds the upstream OpenCode Zen API settings.
type OpenCodeZenConfig struct {
	BaseURL          string `json:"base_url"`
	AnthropicBaseURL string `json:"anthropic_base_url"`
	ResponsesBaseURL string `json:"responses_base_url"`
	GeminiBaseURL    string `json:"gemini_base_url"`
	TimeoutMs        int    `json:"timeout_ms"`
}

// AnthropicCompatibleConfig holds the upstream Anthropic-compatible API settings.
type AnthropicCompatibleConfig struct {
	BaseURL          string `json:"base_url"`
	AnthropicBaseURL string `json:"anthropic_base_url"`
	APIKey           string `json:"api_key"`
	TimeoutMs        int    `json:"timeout_ms"`
}

// LoggingConfig controls application logging behavior.
type LoggingConfig struct {
	Level    string `json:"level"`
	Requests bool   `json:"requests"`
}
