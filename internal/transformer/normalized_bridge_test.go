package transformer

import (
	"encoding/json"
	"testing"

	"github.com/routatic/proxy/internal/config"
	"github.com/routatic/proxy/internal/core"
)

func TestNormalizedToAnthropic_SystemPromptWithNewline(t *testing.T) {
	req := &core.NormalizedRequest{
		Model:        "minimax-m3",
		SystemPrompt: "Line one\nLine two\nLine three",
		MaxTokens:    100,
		Messages: []core.NormalizedMessage{
			{Role: "user", Blocks: []core.NormalizedContentBlock{{Type: "text", Text: "Hello"}}},
		},
	}

	anthropicReq := NormalizedToAnthropic(req, config.ModelConfig{ModelID: "minimax-m3"})

	// The bug: marshaling the request failed when the system prompt contained
	// unescaped newlines because we built the RawMessage by wrapping the raw
	// string in quotes instead of JSON-encoding it.
	_, err := json.Marshal(anthropicReq)
	if err != nil {
		t.Fatalf("json.Marshal failed: %v", err)
	}

	if got := anthropicReq.SystemText(); got != req.SystemPrompt {
		t.Fatalf("system text mismatch: got %q, want %q", got, req.SystemPrompt)
	}
}

func TestNormalizedToAnthropic_MessageContentWithNewline(t *testing.T) {
	req := &core.NormalizedRequest{
		Model:     "minimax-m3",
		MaxTokens: 100,
		Messages: []core.NormalizedMessage{
			{Role: "user", Blocks: []core.NormalizedContentBlock{{Type: "text", Text: "Hello\nWorld"}}},
		},
	}

	anthropicReq := NormalizedToAnthropic(req, config.ModelConfig{ModelID: "minimax-m3"})

	_, err := json.Marshal(anthropicReq)
	if err != nil {
		t.Fatalf("json.Marshal failed: %v", err)
	}

	blocks := anthropicReq.Messages[0].ContentBlocks()
	if len(blocks) != 1 || blocks[0].Text != "Hello\nWorld" {
		t.Fatalf("unexpected content blocks: %+v", blocks)
	}
}

func TestNormalizedToResponses_SystemPromptWithNewline(t *testing.T) {
	req := &core.NormalizedRequest{
		Model:        "gpt-5",
		SystemPrompt: "Line one\nLine two",
		MaxTokens:    100,
		Messages: []core.NormalizedMessage{
			{Role: "user", Blocks: []core.NormalizedContentBlock{{Type: "text", Text: "Hello\nWorld"}}},
		},
	}

	responsesReq := NormalizedToResponses(req, config.ModelConfig{ModelID: "gpt-5"})

	_, err := json.Marshal(responsesReq)
	if err != nil {
		t.Fatalf("json.Marshal failed: %v", err)
	}

	if len(responsesReq.Input) != 2 {
		t.Fatalf("input count mismatch: got %d, want 2", len(responsesReq.Input))
	}

	var systemPrompt string
	if err := json.Unmarshal(responsesReq.Input[0].Content, &systemPrompt); err != nil {
		t.Fatalf("system prompt content was not valid JSON: %v", err)
	}
	if systemPrompt != req.SystemPrompt {
		t.Fatalf("system prompt mismatch: got %q, want %q", systemPrompt, req.SystemPrompt)
	}

	var messageContent string
	if err := json.Unmarshal(responsesReq.Input[1].Content, &messageContent); err != nil {
		t.Fatalf("message content was not valid JSON: %v", err)
	}
	if messageContent != req.Messages[0].TextContent() {
		t.Fatalf("message content mismatch: got %q, want %q", messageContent, req.Messages[0].TextContent())
	}
}

func TestNormalizedToResponses_ToolResultAndEmptyContent(t *testing.T) {
	req := &core.NormalizedRequest{
		Model: "gpt-5.6-luna",
		Messages: []core.NormalizedMessage{
			{
				Role: "user",
				Blocks: []core.NormalizedContentBlock{
					{Type: "text", Text: "Run command"},
				},
			},
			{
				Role: "assistant",
				Blocks: []core.NormalizedContentBlock{
					{Type: "tool_use", Name: "Bash", Input: []byte(`{"command":"ls"}`)},
				},
			},
			{
				Role: "user",
				Blocks: []core.NormalizedContentBlock{
					{Type: "tool_result", ToolUseID: "tool_1", Content: json.RawMessage(`"file1.txt\nfile2.txt"`)},
				},
			},
			{
				Role: "assistant",
				Blocks: []core.NormalizedContentBlock{},
			},
		},
	}

	responsesReq := NormalizedToResponses(req, config.ModelConfig{
		ModelID:         "gpt-5.6-luna",
		ReasoningEffort: "high",
	})

	if responsesReq.Reasoning == nil || responsesReq.Reasoning.Effort != "high" {
		t.Fatalf("expected reasoning effort high, got %+v", responsesReq.Reasoning)
	}

	if len(responsesReq.Input) != 4 {
		t.Fatalf("expected 4 input items, got %d", len(responsesReq.Input))
	}

	for i, input := range responsesReq.Input {
		if len(input.Content) == 0 {
			t.Errorf("input[%d].content is empty", i)
		}
		var content string
		if err := json.Unmarshal(input.Content, &content); err != nil {
			t.Errorf("input[%d].content failed to unmarshal: %v", i, err)
		}
		if content == "" {
			t.Errorf("input[%d].content resolved to empty string", i)
		}
	}
}
