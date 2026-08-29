package provider

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/routatic/proxy/internal/client"
	"github.com/routatic/proxy/internal/config"
	"github.com/routatic/proxy/internal/core"
	"github.com/routatic/proxy/internal/transformer"
)

const (
	defaultAnthropicVersion = "2023-06-01"
)

// AnthCompProvider implements core.Provider for transparent passthrough to
// Anthropic-compatible APIs (including direct Anthropic APIs or proxies).
type AnthCompProvider struct {
	baseProvider
}

// NewAnthCompProvider creates a new AnthCompProvider.
func NewAnthCompProvider(atomic *config.AtomicConfig) *AnthCompProvider {
	return &AnthCompProvider{baseProvider: newBaseProvider(atomic)}
}

// Name returns the provider identifier.
func (p *AnthCompProvider) Name() string { return config.ProviderAnthComp }

// ValidateRequest checks normalized content against this provider's capabilities.
func (p *AnthCompProvider) ValidateRequest(req *core.NormalizedRequest, model config.ModelConfig) error {
	return validateRequest(p, req, model)
}

// Capabilities returns provider-level capabilities.
func (p *AnthCompProvider) Capabilities() core.ProviderCapabilities {
	return core.ProviderCapabilities{
		SupportsStreaming:  true,
		SupportsTools:      true,
		SupportsThinking:   true,
		SupportsImageInput: true,
		MaxContextLength:   200_000,
		DefaultMaxTokens:   8192,
	}
}

// ModelCapabilities returns per-model capabilities.
func (p *AnthCompProvider) ModelCapabilities(modelID string) (core.ProviderCapabilities, bool) {
	return p.Capabilities(), true
}

// WireFormat returns the wire format for AnthComp models.
func (p *AnthCompProvider) WireFormat(model config.ModelConfig) core.WireFormat {
	if wf, ok := core.ParseWireFormat(model.WireFormat); ok {
		return wf
	}
	return core.WireFormatAnthropic
}

// RoundTripName returns the model ID to use in the upstream request.
func (p *AnthCompProvider) RoundTripName(model config.ModelConfig) string {
	return model.ModelID
}

// StreamIdleTimeout returns the maximum gap between bytes on an active stream.
func (p *AnthCompProvider) StreamIdleTimeout(model config.ModelConfig) time.Duration {
	const fallback = 5 * time.Minute
	cfg := p.atomic.Get()
	if cfg == nil {
		return fallback
	}
	ms := cfg.AnthropicCompatible.StreamTimeoutMs
	if ms <= 0 {
		ms = cfg.AnthropicCompatible.TimeoutMs
	}
	if ms <= 0 {
		return fallback
	}
	return time.Duration(ms) * time.Millisecond
}

// Execute sends a non-streaming request and returns the response.
func (p *AnthCompProvider) Execute(ctx context.Context, req *core.NormalizedRequest, model config.ModelConfig) (*core.ExecuteResult, error) {
	cfg := p.atomic.Get()
	endpoint := p.endpoint(cfg)
	if endpoint == "" {
		return nil, fmt.Errorf("anthropic_base_url / base_url not configured for anth_comp provider")
	}
	apiKey := p.anthCompAPIKey(cfg)

	anthropicReq := transformer.NormalizedToAnthropic(req, model)
	rawBody, err := json.Marshal(anthropicReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal anthropic request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(rawBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if apiKey != "" {
		httpReq.Header.Set("x-api-key", apiKey)
		httpReq.Header.Set("Authorization", "Bearer "+apiKey)
	}
	httpReq.Header.Set("anthropic-version", defaultAnthropicVersion)
	httpReq.Header.Set("User-Agent", routaticUserAgent)

	start := time.Now()
	resp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode >= http.StatusBadRequest {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, &client.APIError{StatusCode: resp.StatusCode, Body: string(bodyBytes)}
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	return &core.ExecuteResult{
		Body:    body,
		ModelID: model.ModelID,
		Latency: time.Since(start),
	}, nil
}

// Stream sends a streaming request and returns an io.ReadCloser for SSE events.
func (p *AnthCompProvider) Stream(ctx context.Context, req *core.NormalizedRequest, model config.ModelConfig) (io.ReadCloser, error) {
	cfg := p.atomic.Get()
	endpoint := p.endpoint(cfg)
	if endpoint == "" {
		return nil, fmt.Errorf("anthropic_base_url / base_url not configured for anth_comp provider")
	}
	apiKey := p.anthCompAPIKey(cfg)

	anthropicReq := transformer.NormalizedToAnthropic(req, model)
	rawBody, err := json.Marshal(anthropicReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal anthropic request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(rawBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if apiKey != "" {
		httpReq.Header.Set("x-api-key", apiKey)
		httpReq.Header.Set("Authorization", "Bearer "+apiKey)
	}
	httpReq.Header.Set("anthropic-version", defaultAnthropicVersion)
	httpReq.Header.Set("User-Agent", routaticUserAgent)
	httpReq.Header.Set("Accept", "text/event-stream")

	resp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}

	if resp.StatusCode >= http.StatusBadRequest {
		bodyBytes, _ := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		return nil, &client.APIError{StatusCode: resp.StatusCode, Body: string(bodyBytes)}
	}

	return resp.Body, nil
}

func (p *AnthCompProvider) endpoint(cfg *config.Config) string {
	if cfg == nil {
		return ""
	}
	if cfg.AnthropicCompatible.AnthropicBaseURL != "" {
		return cfg.AnthropicCompatible.AnthropicBaseURL
	}
	return cfg.AnthropicCompatible.BaseURL
}

func (p *AnthCompProvider) anthCompAPIKey(cfg *config.Config) string {
	if cfg == nil {
		return ""
	}
	keys := cfg.AnthropicCompatible.EffectiveAPIKeys()
	if len(keys) > 0 {
		return p.nextAPIKey(keys)
	}
	return p.nextAPIKey(cfg.EffectiveAPIKeys())
}
