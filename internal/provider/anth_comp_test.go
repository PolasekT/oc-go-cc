package provider

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/routatic/proxy/internal/config"
	"github.com/routatic/proxy/internal/core"
	"github.com/routatic/proxy/pkg/types"
)

func TestAnthCompProvider_Name(t *testing.T) {
	p := NewAnthCompProvider(nil)
	if got := p.Name(); got != "anth-comp" {
		t.Errorf("Name() = %q, want %q", got, "anth-comp")
	}
}

func TestAnthCompProvider_WireFormat(t *testing.T) {
	cfg := &config.Config{}
	atomic := config.NewAtomicConfig(cfg, "")
	p := NewAnthCompProvider(atomic)
	if got := p.WireFormat(config.ModelConfig{ModelID: "claude-3-7-sonnet"}); got != core.WireFormatAnthropic {
		t.Errorf("WireFormat() = %v, want WireFormatAnthropic", got)
	}
}

func TestAnthCompProvider_Capabilities(t *testing.T) {
	p := NewAnthCompProvider(nil)
	caps := p.Capabilities()
	if !caps.SupportsStreaming {
		t.Error("SupportsStreaming = false, want true")
	}
	if !caps.SupportsTools {
		t.Error("SupportsTools = false, want true")
	}
	if !caps.SupportsThinking {
		t.Error("SupportsThinking = false, want true")
	}
	if !caps.SupportsImageInput {
		t.Error("SupportsImageInput = false, want true")
	}
	if caps.MaxContextLength != 200_000 {
		t.Errorf("MaxContextLength = %d, want 200000", caps.MaxContextLength)
	}
}

func TestAnthCompProvider_StreamIdleTimeout(t *testing.T) {
	cfg := &config.Config{
		AnthropicCompatible: config.AnthropicCompatibleConfig{
			StreamTimeoutMs: 15000,
		},
	}
	atomic := config.NewAtomicConfig(cfg, "")
	p := NewAnthCompProvider(atomic)
	got := p.StreamIdleTimeout(config.ModelConfig{})
	if got != 15*time.Second {
		t.Errorf("StreamIdleTimeout() = %v, want 15s", got)
	}
}

func TestAnthCompProvider_Execute(t *testing.T) {
	var receivedKey string
	var receivedVer string
	var receivedBody types.MessageRequest

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedKey = r.Header.Get("x-api-key")
		receivedVer = r.Header.Get("anthropic-version")
		_ = json.NewDecoder(r.Body).Decode(&receivedBody)

		resp := types.MessageResponse{
			ID:   "msg_123",
			Type: "message",
			Role: "assistant",
			Content: []types.ContentBlock{
				{Type: "text", Text: "Hello from AnthComp!"},
			},
			Usage: types.Usage{
				InputTokens:  15,
				OutputTokens: 25,
			},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	cfg := &config.Config{
		AnthropicCompatible: config.AnthropicCompatibleConfig{
			AnthropicBaseURL: srv.URL,
			APIKey:           "test-anth-key",
		},
	}
	atomic := config.NewAtomicConfig(cfg, "")
	p := NewAnthCompProvider(atomic)

	req := &core.NormalizedRequest{
		Model: "claude-3-7-sonnet",
		Messages: []core.NormalizedMessage{
			{Role: "user", Blocks: []core.NormalizedContentBlock{{Type: "text", Text: "Hi"}}},
		},
	}
	modelCfg := config.ModelConfig{ModelID: "claude-3-7-sonnet", Provider: "anth-comp"}

	res, err := p.Execute(context.Background(), req, modelCfg)
	if err != nil {
		t.Fatalf("Execute() error: %v", err)
	}

	if receivedKey != "test-anth-key" {
		t.Errorf("receivedKey = %q, want %q", receivedKey, "test-anth-key")
	}
	if receivedVer != "2023-06-01" {
		t.Errorf("receivedVer = %q, want %q", receivedVer, "2023-06-01")
	}
	if res.ModelID != "claude-3-7-sonnet" {
		t.Errorf("res.ModelID = %q, want claude-3-7-sonnet", res.ModelID)
	}

	var resp types.MessageResponse
	if err := json.Unmarshal(res.Body, &resp); err != nil {
		t.Fatalf("unmarshal response error: %v", err)
	}
	if len(resp.Content) == 0 || resp.Content[0].Text != "Hello from AnthComp!" {
		t.Errorf("unexpected content: %+v", resp.Content)
	}
}

func TestAnthCompProvider_Stream(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("event: message_start\ndata: {}\n\n"))
	}))
	defer srv.Close()

	cfg := &config.Config{
		AnthropicCompatible: config.AnthropicCompatibleConfig{
			AnthropicBaseURL: srv.URL,
			APIKey:           "test-anth-key",
		},
	}
	atomic := config.NewAtomicConfig(cfg, "")
	p := NewAnthCompProvider(atomic)

	req := &core.NormalizedRequest{
		Model: "claude-3-7-sonnet",
		Messages: []core.NormalizedMessage{
			{Role: "user", Blocks: []core.NormalizedContentBlock{{Type: "text", Text: "Hi"}}},
		},
	}
	modelCfg := config.ModelConfig{ModelID: "claude-3-7-sonnet", Provider: "anth-comp"}

	stream, err := p.Stream(context.Background(), req, modelCfg)
	if err != nil {
		t.Fatalf("Stream() error: %v", err)
	}
	defer func() { _ = stream.Close() }()

	data, err := io.ReadAll(stream)
	if err != nil {
		t.Fatalf("ReadAll error: %v", err)
	}
	if !strings.Contains(string(data), "message_start") {
		t.Errorf("Stream data missing message_start: %s", string(data))
	}
}
