// Package handlers contains HTTP request handlers for API endpoints.
package handlers

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"sync/atomic"
	"time"

	"oc-go-cc/internal/client"
	"oc-go-cc/internal/config"
	"oc-go-cc/internal/metrics"
	"oc-go-cc/internal/middleware"
	"oc-go-cc/internal/router"
	"oc-go-cc/internal/token"
	"oc-go-cc/internal/transformer"
	"oc-go-cc/pkg/types"
)

// MessagesHandler handles /v1/messages requests.
type MessagesHandler struct {
	atomic              *config.AtomicConfig
	client              *client.OpenCodeClient
	modelRouter         *router.ModelRouter
	fallbackHandler     *router.FallbackHandler
	requestTransformer  *transformer.RequestTransformer
	responseTransformer *transformer.ResponseTransformer
	streamHandler       *transformer.StreamHandler
	tokenCounter        *token.Counter
	logger              *slog.Logger
	rateLimiter         *middleware.RateLimiter
	requestDedup        *middleware.RequestDeduplicator
	requestIDGen        *middleware.RequestIDGenerator
	metrics             *metrics.Metrics
}

// responseWriter wraps http.ResponseWriter to track if headers were written.
type responseWriter struct {
	http.ResponseWriter
	wroteHeader bool
}

func (w *responseWriter) WriteHeader(code int) {
	if !w.wroteHeader {
		w.wroteHeader = true
		w.ResponseWriter.WriteHeader(code)
	}
}

func (w *responseWriter) Write(b []byte) (int, error) {
	if !w.wroteHeader {
		w.WriteHeader(http.StatusOK)
	}
	return w.ResponseWriter.Write(b)
}

// Flush implements http.Flusher for SSE streaming support.
func (w *responseWriter) Flush() {
	if f, ok := w.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

// NewMessagesHandler creates a new messages handler.
func NewMessagesHandler(
	atomic *config.AtomicConfig,
	openCodeClient *client.OpenCodeClient,
	modelRouter *router.ModelRouter,
	fallbackHandler *router.FallbackHandler,
	tokenCounter *token.Counter,
	metrics *metrics.Metrics,
) *MessagesHandler {
	return &MessagesHandler{
		atomic:              atomic,
		client:              openCodeClient,
		modelRouter:         modelRouter,
		fallbackHandler:     fallbackHandler,
		requestTransformer:  transformer.NewRequestTransformer(),
		responseTransformer: transformer.NewResponseTransformer(),
		streamHandler:       transformer.NewStreamHandler(),
		tokenCounter:        tokenCounter,
		logger:              slog.Default(),
		rateLimiter:         middleware.NewRateLimiter(100, time.Minute),
		requestDedup:        middleware.NewRequestDeduplicator(500 * time.Millisecond),
		requestIDGen:        middleware.NewRequestIDGenerator(),
		metrics:             metrics,
	}
}

// HandleMessages handles POST /v1/messages.
func (h *MessagesHandler) HandleMessages(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Generate or get request ID for correlation
	requestID := r.Header.Get("X-Request-ID")
	if requestID == "" {
		requestID = h.requestIDGen.Generate()
	}
	w.Header().Set("X-Request-ID", requestID)

	reqLogger := h.logger.With("req_id", requestID)

	// Rate limiting
	clientIP := middleware.GetClientIP(r)
	if !h.rateLimiter.Allow(clientIP) {
		h.metrics.RecordRateLimited()
		reqLogger.Warn("rate limited", "client", clientIP)
		http.Error(w, "rate limited", http.StatusTooManyRequests)
		return
	}

	// Read the raw request body for debug logging
	var rawBody json.RawMessage
	if err := json.NewDecoder(r.Body).Decode(&rawBody); err != nil {
		h.sendError(w, http.StatusBadRequest, "invalid request body", err)
		return
	}

	reqLogger.Debug("incoming request", "body", lazyJSONTruncator{rawBody})

	// Deduplicate - skip duplicate requests
	if _, ok := h.requestDedup.TryAcquire(rawBody); !ok {
		h.metrics.RecordDeduplicated()
		reqLogger.Info("duplicate request skipped")
		return
	}

	// Parse into Anthropic request
	var anthropicReq types.MessageRequest
	if err := json.Unmarshal(rawBody, &anthropicReq); err != nil {
		h.sendError(w, http.StatusBadRequest, "invalid request body", err)
		return
	}

	// Validate request
	if err := anthropicReq.Validate(); err != nil {
		h.sendError(w, http.StatusBadRequest, err.Error(), nil)
		return
	}

	// Record metrics
	isStreaming := anthropicReq.Stream != nil && *anthropicReq.Stream
	h.metrics.RecordRequest(isStreaming)

	effort := ""
	if anthropicReq.OutputConfig != nil {
		effort = anthropicReq.OutputConfig.Effort
	}

	reqLogger.Info("received request",
		"model", anthropicReq.Model,
		"effort", effort,
		"streaming", isStreaming,
		"messages", len(anthropicReq.Messages),
		"tools", len(anthropicReq.Tools),
		"max_tokens", anthropicReq.MaxTokens,
	)

	// Build message content for routing.
	var routerMessages []router.MessageContent
	for _, msg := range anthropicReq.Messages {
		blocks := msg.ContentBlocks()
		content := extractTextFromBlocks(blocks)
		mc := router.MessageContent{
			Role:    msg.Role,
			Content: content,
		}
		routerMessages = append(routerMessages, mc)
	}

	// Count tokens accurately using tools and all text fields.
	systemText, err := systemAndToolsTokenText(anthropicReq.SystemText(), anthropicReq.Tools)
	if err != nil {
		reqLogger.Warn("failed to process tools for token counting", "error", err)
	}
	tokenMessages := tokenMessagesFromAnthropic(anthropicReq.Messages)
	tokenCount, err := h.tokenCounter.CountMessages(systemText, tokenMessages)
	if err != nil {
		reqLogger.Warn("failed to count tokens", "error", err)
		tokenCount = 0
	}

	// Route to appropriate model and build fallback chain.
	if effort != "" {
		reqLogger.Debug("extracted reasoning effort", "effort", effort)
	}

	modelChain, routeResult, err := h.buildModelChain(anthropicReq.Model, effort, routerMessages, tokenCount, isStreaming, reqLogger)
	if err != nil {
		h.sendError(w, http.StatusInternalServerError, "routing failed", err)
		return
	}

	reqLogger.Debug("constructed model chain", "chain", lazyModelChain{modelChain})

	reqLogger.Info("routing request",
		"scenario", routeResult.Scenario,
		"model", routeResult.Primary.ModelID,
		"provider", routeResult.Primary.Provider,
		"path", routeResult.Reason,
		"tokens", tokenCount,
	)

	if isStreaming {
		// Streaming: use ProxyStream for real-time SSE transformation
		h.handleStreaming(w, r, &anthropicReq, modelChain, rawBody, reqLogger)
	} else {
		// Non-streaming: execute with fallback and return full response
		h.handleNonStreaming(w, r, &anthropicReq, modelChain, rawBody, reqLogger)
	}
}

// buildModelChain resolves the request to a model chain (primary + fallbacks),
// honoring model_overrides (with a deduplicated scenario safety-net) and
// respecting the streaming-scenario-routing toggle.
//
// Precedence:
//  1. If requestedModel matches an entry in model_overrides, use that as the
//     primary and append the scenario chain as a deduplicated safety net.
//  2. Otherwise, fall through to scenario-based routing via routeOnce.
func (h *MessagesHandler) buildModelChain(
	requestedModel string,
	effort string,
	routerMessages []router.MessageContent,
	tokenCount int,
	isStreaming bool,
	reqLogger *slog.Logger,
) ([]config.ModelConfig, router.RouteResult, error) {
	cfg := h.atomic.Get()

	reqLogger.Debug("building model chain", "model", requestedModel, "effort", effort, "respect", cfg.RespectRequestedModel, "use_effort", cfg.RespectRequestedModelUseEffort, "enable_effort_routing", cfg.EnableEffortScenarioRouting)

	if requestedModel != "" {
		// Priority 1: Effort override
		if effort != "" && cfg.RespectRequestedModelUseEffort {
			if effortResult, ok := h.modelRouter.RouteWithEffortOverride(requestedModel, effort); ok {
				if cfg.RespectRequestedModelUseContextThreshold && effortResult.Primary.ContextThreshold > 0 && tokenCount > effortResult.Primary.ContextThreshold {
					reqLogger.Info("effort override context threshold exceeded, bypassing override", "model", requestedModel, "effort", effort, "tokens", tokenCount, "threshold", effortResult.Primary.ContextThreshold)
				} else {
					reqLogger.Debug("effort override matched", "model", requestedModel, "effort", effort, "target", effortResult.Primary.ModelID)
					scenarioResult, err := h.routeOnce(routerMessages, tokenCount, "", effort, isStreaming)
					if err != nil {
						return effortResult.GetModelChain(), effortResult, err
					}
					chain := appendUniqueModels(effortResult.GetModelChain(), scenarioResult.GetModelChain())
					return chain, effortResult, nil
				}
			}
		}

		// Priority 2: Model override
		if overrideResult, ok := h.modelRouter.RouteWithOverride(requestedModel); ok {
			if cfg.RespectRequestedModelUseContextThreshold && overrideResult.Primary.ContextThreshold > 0 && tokenCount > overrideResult.Primary.ContextThreshold {
				reqLogger.Info("model override context threshold exceeded, bypassing override", "model", requestedModel, "tokens", tokenCount, "threshold", overrideResult.Primary.ContextThreshold)
			} else {
				reqLogger.Debug("model override matched", "model", requestedModel, "target", overrideResult.Primary.ModelID)
				scenarioResult, err := h.routeOnce(routerMessages, tokenCount, "", effort, isStreaming)
				if err != nil {
					// Override is valid; surface the scenario routing error rather
					// than silently dropping the safety net.
					return overrideResult.GetModelChain(), overrideResult, err
				}
				chain := appendUniqueModels(overrideResult.GetModelChain(), scenarioResult.GetModelChain())
				return chain, overrideResult, nil
			}
		}
	}

	result, err := h.routeOnce(routerMessages, tokenCount, requestedModel, effort, isStreaming)
	if err != nil {
		return nil, result, err
	}
	return result.GetModelChain(), result, nil
}

// routeOnce performs scenario-based routing, honoring the streaming-scenario-routing
// toggle. Pass requestedModel="" to force scenario routing (used for the override
// safety-net chain), or a non-empty value to let resolveRequestedModel kick in
// (only when respect_requested_model is enabled and no override matched).
func (h *MessagesHandler) routeOnce(
	routerMessages []router.MessageContent,
	tokenCount int,
	requestedModel string,
	effort string,
	isStreaming bool,
) (router.RouteResult, error) {
	if isStreaming && !h.modelRouter.IsStreamingScenarioRoutingEnabled() {
		// Streaming: use faster models to minimize TTFT (time-to-first-token)
		return h.modelRouter.RouteForStreaming(routerMessages, tokenCount, requestedModel, effort), nil
	}
	return h.modelRouter.Route(routerMessages, tokenCount, requestedModel, effort)
}

// appendUniqueModels appends models from extra to base, skipping any model_id
// already present in base. The first occurrence of a ModelID is kept; later
// duplicates are dropped. Order of the base chain is preserved.
func appendUniqueModels(base, extra []config.ModelConfig) []config.ModelConfig {
	if len(extra) == 0 {
		return base
	}
	seen := make(map[string]struct{}, len(base))
	for _, m := range base {
		seen[m.ModelID] = struct{}{}
	}
	for _, m := range extra {
		if _, ok := seen[m.ModelID]; ok {
			continue
		}
		base = append(base, m)
		seen[m.ModelID] = struct{}{}
	}
	return base
}

// handleStreaming handles a streaming request with real-time SSE proxying.
func (h *MessagesHandler) handleStreaming(
	w http.ResponseWriter,
	r *http.Request,
	anthropicReq *types.MessageRequest,
	modelChain []config.ModelConfig,
	rawBody json.RawMessage,
	reqLogger *slog.Logger,
) {
	clientCtx := r.Context()

	rw := &responseWriter{ResponseWriter: w}

	// Set SSE headers immediately
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")
	rw.WriteHeader(http.StatusOK)
	if f, ok := w.(http.Flusher); ok {
		f.Flush()
	}

	// Start heartbeat
	var finished int32
	heartbeatDone := make(chan struct{})
	go func() {
		ticker := time.NewTicker(3 * time.Second)
		defer ticker.Stop()

		for {
			select {
			case <-ticker.C:
				if atomic.LoadInt32(&finished) == 1 {
					return
				}
				_, _ = fmt.Fprintf(rw, ":keepalive\n\n")
				if f, ok := w.(http.Flusher); ok {
					f.Flush()
				}
			case <-heartbeatDone:
				return
			case <-clientCtx.Done():
				return
			}
		}
	}()
	defer func() {
		atomic.StoreInt32(&finished, 1)
		close(heartbeatDone)
	}()

	streamStart := time.Now()

	for _, model := range modelChain {
		select {
		case <-clientCtx.Done():
			reqLogger.Info("client disconnected, stopping streaming fallbacks")
			return
		default:
		}

		reqLogger.Info("attempting streaming model", "model", model.ModelID, "provider", model.Provider)

		ctx, cancel := context.WithTimeout(context.Background(), h.client.GetTimeout(model))

		// Check if this is an Anthropic-native model or uses anth-comp provider
		if client.IsAnthropicModel(model.ModelID) || client.IsAnthropicCompatible(model) {
			modelBody, err := h.normalizeAnthropicBody(rawBody, model.ModelID, reqLogger)
			if err != nil {
				reqLogger.Warn("failed to normalize Anthropic body, using original with model replacement", "error", err)
				modelBody = h.replaceModelInRawBody(rawBody, model.ModelID, reqLogger)
			}

			if err := h.handleAnthropicStreaming(ctx, rw, modelBody, model.ModelID, model, reqLogger); err != nil {
				cancel()
				if clientCtx.Err() == context.Canceled {
					reqLogger.Info("client disconnected during anthropic stream")
					return
				}
				reqLogger.Warn("anthropic streaming failed", "model", model.ModelID, "error", err)
				continue
			}
			cancel()
			latency := time.Since(streamStart)
			h.metrics.RecordSuccess(model.ModelID, latency)
			// Wait, the user wants token statistics in "model succeeded". 
			// Streaming doesn't return the full token stats easily without a response transformer hooked into the stream.
			reqLogger.Info("streaming completed", "model", model.ModelID, "latency", latency)
			return
		} else if client.IsZen(model) {
			// Zen-specific endpoint handling
			endpointType := client.ClassifyEndpoint(model.ModelID)
			switch endpointType {
			case client.EndpointResponses:
				if err := h.handleResponsesStreaming(ctx, rw, anthropicReq, model, clientCtx, reqLogger); err != nil {
					cancel()
					if clientCtx.Err() == context.Canceled {
						reqLogger.Info("client disconnected during responses stream")
						return
					}
					reqLogger.Warn("responses streaming failed", "model", model.ModelID, "error", err)
					continue
				}
				cancel()
				latency := time.Since(streamStart)
				h.metrics.RecordSuccess(model.ModelID, latency)
				reqLogger.Info("streaming completed", "model", model.ModelID, "latency", latency)
				return

			case client.EndpointGemini:
				if err := h.handleGeminiStreaming(ctx, rw, anthropicReq, model, clientCtx, reqLogger); err != nil {
					cancel()
					if clientCtx.Err() == context.Canceled {
						reqLogger.Info("client disconnected during gemini stream")
						return
					}
					reqLogger.Warn("gemini streaming failed", "model", model.ModelID, "error", err)
					continue
				}
				cancel()
				latency := time.Since(streamStart)
				h.metrics.RecordSuccess(model.ModelID, latency)
				reqLogger.Info("streaming completed", "model", model.ModelID, "latency", latency)
				return

			default:
				// Fall through to OpenAI-compatible handling
			}
		} else {
			// OpenAI-compatible models (both Go and Zen)
			openaiReq, err := h.requestTransformer.TransformRequest(anthropicReq, model)
			if err != nil {
				cancel()
				reqLogger.Warn("request transform failed", "model", model.ModelID, "error", err)
				continue
			}

			streamBody, err := h.client.GetStreamingBody(ctx, model.ModelID, openaiReq, model)
			if err != nil {
				cancel()
				if clientCtx.Err() == context.Canceled {
					reqLogger.Info("client disconnected during upstream request")
					return
				}
				reqLogger.Warn("streaming request failed", "model", model.ModelID, "error", err)
				continue
			}

			if err := h.streamHandler.ProxyStream(rw, streamBody, model.ModelID, clientCtx); err != nil {
				_ = streamBody.Close()
				cancel()
				if err == transformer.ErrClientDisconnected {
					reqLogger.Info("client disconnected during stream")
					return
				}
				if clientCtx.Err() == context.Canceled {
					reqLogger.Info("client disconnected during stream (context canceled)")
					return
				}
				reqLogger.Warn("stream proxy failed", "model", model.ModelID, "error", err)
				continue
			}

			_ = streamBody.Close()
			cancel()
			latency := time.Since(streamStart)
			h.metrics.RecordSuccess(model.ModelID, latency)
			reqLogger.Info("streaming completed", "model", model.ModelID, "latency", latency)
			return
		}
	}

	h.metrics.RecordFailure()
	if !rw.wroteHeader {
		h.sendError(w, http.StatusBadGateway, "all streaming models failed", nil)
	} else {
		h.sendStreamError(rw, "all upstream models failed", reqLogger)
	}
}

// handleResponsesStreaming handles streaming for OpenAI Responses endpoint.
func (h *MessagesHandler) handleResponsesStreaming(
	ctx context.Context,
	w http.ResponseWriter,
	anthropicReq *types.MessageRequest,
	model config.ModelConfig,
	clientCtx context.Context,
	reqLogger *slog.Logger,
) error {
	req, err := h.requestTransformer.TransformToResponses(anthropicReq, model)
	if err != nil {
		return fmt.Errorf("responses transform failed: %w", err)
	}

	streamBody, err := h.client.GetResponsesStreamingBody(ctx, model.ModelID, req, model)
	if err != nil {
		return err
	}

	if err := h.streamHandler.ProxyResponsesStream(w, streamBody, model.ModelID, clientCtx); err != nil {
		_ = streamBody.Close()
		return err
	}

	_ = streamBody.Close()
	return nil
}

// handleGeminiStreaming handles streaming for Gemini endpoint.
func (h *MessagesHandler) handleGeminiStreaming(
	ctx context.Context,
	w http.ResponseWriter,
	anthropicReq *types.MessageRequest,
	model config.ModelConfig,
	clientCtx context.Context,
	reqLogger *slog.Logger,
) error {
	req, err := h.requestTransformer.TransformToGemini(anthropicReq, model)
	if err != nil {
		return fmt.Errorf("gemini transform failed: %w", err)
	}

	streamBody, err := h.client.GetGeminiStreamingBody(ctx, model.ModelID, req, model)
	if err != nil {
		return err
	}

	if err := h.streamHandler.ProxyGeminiStream(w, streamBody, model.ModelID, clientCtx); err != nil {
		_ = streamBody.Close()
		return err
	}

	_ = streamBody.Close()
	return nil
}

// normalizeAnthropicBody ensures the request follows strict Anthropic API rules by moving
// any "system" role messages from the messages array to the top-level system field.
// It also replaces the "model" field with the provided modelID.
func (h *MessagesHandler) normalizeAnthropicBody(raw json.RawMessage, modelID string, reqLogger *slog.Logger) (json.RawMessage, error) {
	var m map[string]interface{}
	if err := json.Unmarshal(raw, &m); err != nil {
		return nil, err
	}
	m["model"] = modelID

	messages, ok := m["messages"].([]interface{})
	if !ok {
		data, err := json.Marshal(m)
		return json.RawMessage(data), err
	}

	var newMessages []interface{}
	var extraSystem []interface{}

	for _, msg := range messages {
		mobj, ok := msg.(map[string]interface{})
		if !ok {
			newMessages = append(newMessages, msg)
			continue
		}
		role, _ := mobj["role"].(string)
		if role == "system" {
			content := mobj["content"]
			if s, ok := content.(string); ok {
				extraSystem = append(extraSystem, map[string]interface{}{
					"type": "text",
					"text": s,
				})
			} else if blocks, ok := content.([]interface{}); ok {
				extraSystem = append(extraSystem, blocks...)
			}
		} else {
			newMessages = append(newMessages, msg)
		}
	}

	if len(extraSystem) > 0 {
		reqLogger.Debug("normalizing Anthropic body: moving system messages from messages array to system field", "count", len(extraSystem))
		m["messages"] = newMessages

		currentSystem := m["system"]
		var finalSystem []interface{}

		if currentSystem == nil {
			finalSystem = extraSystem
		} else if s, ok := currentSystem.(string); ok {
			finalSystem = append([]interface{}{map[string]interface{}{"type": "text", "text": s}}, extraSystem...)
		} else if blocks, ok := currentSystem.([]interface{}); ok {
			finalSystem = append(blocks, extraSystem...)
		} else {
			// Fallback: if current system is unknown format, prioritize extras
			finalSystem = extraSystem
		}
		m["system"] = finalSystem
	}

	data, err := json.Marshal(m)
	if err != nil {
		return nil, err
	}
	reqLogger.Debug("normalized Anthropic body", "body", lazyJSONTruncator{json.RawMessage(data)})
	return json.RawMessage(data), nil
}

// replaceModelInRawBody replaces the model field in raw JSON body with the actual model ID.
func (h *MessagesHandler) replaceModelInRawBody(rawBody json.RawMessage, modelID string, reqLogger *slog.Logger) json.RawMessage {
	bodyStr := string(rawBody)

	if idx := strings.Index(bodyStr, `"model":"`); idx != -1 {
		start := idx + len(`"model":"`)
		if end := strings.Index(bodyStr[start:], `"`); end != -1 {
			oldModel := bodyStr[start : start+end]
			newBody := bodyStr[:start] + modelID + bodyStr[start+end:]
			reqLogger.Debug("replaced model in request body",
				"old_model", oldModel,
				"new_model", modelID,
				"success", true)
			return json.RawMessage(newBody)
		}
	}

	reqLogger.Warn("could not find model field in request body, using original",
		"body_preview", bodyStr[:min(len(bodyStr), 200)])
	return rawBody
}

// handleAnthropicStreaming sends a raw Anthropic request to the Anthropic endpoint.
func (h *MessagesHandler) handleAnthropicStreaming(
	ctx context.Context,
	w http.ResponseWriter,
	rawBody json.RawMessage,
	modelID string,
	model config.ModelConfig,
	reqLogger *slog.Logger,
) error {
	reqLogger.Debug("sending anthropic streaming request",
		"model_id", modelID,
		"body_preview", string(rawBody)[:min(len(rawBody), 200)])

	resp, err := h.client.SendAnthropicRequest(ctx, rawBody, true, model)
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()

	_, err = io.Copy(w, resp.Body)
	if err != nil {
		if ctx.Err() == context.Canceled {
			return transformer.ErrClientDisconnected
		}
		return fmt.Errorf("failed to copy response: %w", err)
	}

	return nil
}

// sendStreamError sends an error event in the SSE stream.
func (h *MessagesHandler) sendStreamError(w http.ResponseWriter, message string, reqLogger *slog.Logger) {
	reqLogger.Error("sending stream error", "message", message)

	errorEvent := map[string]interface{}{
		"type": "error",
		"error": map[string]interface{}{
			"type":    "api_error",
			"message": message,
		},
	}

	data, _ := json.Marshal(errorEvent)
	_, _ = fmt.Fprintf(w, "event: error\ndata: %s\n\n", string(data))

	if f, ok := w.(http.Flusher); ok {
		f.Flush()
	}
}

// handleNonStreaming handles a non-streaming request with fallback.
func (h *MessagesHandler) handleNonStreaming(
	w http.ResponseWriter,
	r *http.Request,
	anthropicReq *types.MessageRequest,
	modelChain []config.ModelConfig,
	rawBody json.RawMessage,
	reqLogger *slog.Logger,
) {
	ctx := r.Context()
	startTime := time.Now()

	result, responseBody, err := h.fallbackHandler.ExecuteWithFallback(
		ctx,
		modelChain,
		reqLogger,
		func(ctx context.Context, model config.ModelConfig) ([]byte, error) {
			timeoutCtx, cancel := context.WithTimeout(ctx, h.client.GetTimeout(model))
			defer cancel()

			// Check if this is an Anthropic-native model or uses anth-comp provider
			if client.IsAnthropicModel(model.ModelID) || client.IsAnthropicCompatible(model) {
				modelBody, err := h.normalizeAnthropicBody(rawBody, model.ModelID, reqLogger)
				if err != nil {
					reqLogger.Warn("failed to normalize Anthropic body, using original with model replacement", "error", err)
					modelBody = h.replaceModelInRawBody(rawBody, model.ModelID, reqLogger)
				}
				return h.executeAnthropicRequest(timeoutCtx, modelBody, model, reqLogger)
			}

			// Zen-specific endpoint handling
			if client.IsZen(model) {
				endpointType := client.ClassifyEndpoint(model.ModelID)
				switch endpointType {
				case client.EndpointResponses:
					return h.executeResponsesRequest(timeoutCtx, anthropicReq, model, reqLogger)
				case client.EndpointGemini:
					return h.executeGeminiRequest(timeoutCtx, anthropicReq, model, reqLogger)
				default:
					// Fall through to OpenAI-compatible handling
				}
			}

			// OpenAI-compatible models (both Go and Zen)
			return h.executeOpenAIRequest(timeoutCtx, anthropicReq, model, reqLogger)
		},
	)

	if err != nil {
		h.metrics.RecordFailure()
		h.sendError(w, http.StatusBadGateway, "all models failed", err)
		return
	}

	latency := time.Since(startTime)
	h.metrics.RecordSuccess(result.ModelID, latency)

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(responseBody)
}

// executeAnthropicRequest executes a request to the Anthropic endpoint (for MiniMax models).
func (h *MessagesHandler) executeAnthropicRequest(
	ctx context.Context,
	rawBody json.RawMessage,
	model config.ModelConfig,
	reqLogger *slog.Logger,
) ([]byte, error) {
	resp, err := h.client.SendAnthropicRequest(ctx, rawBody, false, model)
	if err != nil {
		return nil, fmt.Errorf("anthropic request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	reqLogger.Debug("anthropic response", "body", string(body))

	return body, nil
}

// executeOpenAIRequest executes a request to the OpenAI endpoint with transformation.
func (h *MessagesHandler) executeOpenAIRequest(
	ctx context.Context,
	anthropicReq *types.MessageRequest,
	model config.ModelConfig,
	reqLogger *slog.Logger,
) ([]byte, error) {
	openaiReq, err := h.requestTransformer.TransformRequest(anthropicReq, model)
	if err != nil {
		return nil, fmt.Errorf("request transform failed: %w", err)
	}

	resp, err := h.client.ChatCompletionNonStreaming(ctx, model.ModelID, openaiReq, model)
	if err != nil {
		return nil, fmt.Errorf("chat completion failed: %w", err)
	}

	anthropicResp, err := h.responseTransformer.TransformResponse(resp, model.ModelID)
	if err != nil {
		return nil, fmt.Errorf("response transform failed: %w", err)
	}

	return json.Marshal(anthropicResp)
}

// executeResponsesRequest executes a request to the OpenAI Responses endpoint.
func (h *MessagesHandler) executeResponsesRequest(
	ctx context.Context,
	anthropicReq *types.MessageRequest,
	model config.ModelConfig,
	reqLogger *slog.Logger,
) ([]byte, error) {
	req, err := h.requestTransformer.TransformToResponses(anthropicReq, model)
	if err != nil {
		return nil, fmt.Errorf("responses transform failed: %w", err)
	}

	resp, err := h.client.ResponsesCompletionNonStreaming(ctx, model.ModelID, req, model)
	if err != nil {
		return nil, fmt.Errorf("responses completion failed: %w", err)
	}

	anthropicResp, err := h.responseTransformer.TransformResponsesResponse(resp, model.ModelID)
	if err != nil {
		return nil, fmt.Errorf("response transform failed: %w", err)
	}

	return json.Marshal(anthropicResp)
}

// executeGeminiRequest executes a request to the Gemini endpoint.
func (h *MessagesHandler) executeGeminiRequest(
	ctx context.Context,
	anthropicReq *types.MessageRequest,
	model config.ModelConfig,
	reqLogger *slog.Logger,
) ([]byte, error) {
	req, err := h.requestTransformer.TransformToGemini(anthropicReq, model)
	if err != nil {
		return nil, fmt.Errorf("gemini transform failed: %w", err)
	}

	resp, err := h.client.GeminiCompletionNonStreaming(ctx, model.ModelID, req, model)
	if err != nil {
		return nil, fmt.Errorf("gemini completion failed: %w", err)
	}

	anthropicResp, err := h.responseTransformer.TransformGeminiResponse(resp, model.ModelID)
	if err != nil {
		return nil, fmt.Errorf("response transform failed: %w", err)
	}

	return json.Marshal(anthropicResp)
}

// extractTextFromBlocks extracts plain text from Anthropic content blocks.
func extractTextFromBlocks(blocks []types.ContentBlock) string {
	var content string
	for _, block := range blocks {
		switch block.Type {
		case "text":
			content += block.Text
		case "tool_use":
			content += fmt.Sprintf("[Tool Use: %s]", block.Name)
		case "tool_result":
			content += block.TextContent()
		case "thinking":
			// Skip thinking blocks for text extraction
		case "image":
			content += "[Image]"
		}
	}
	return content
}

// sendError sends an error response in Anthropic format.
func (h *MessagesHandler) sendError(w http.ResponseWriter, statusCode int, message string, err error) {
	h.logger.Error("request error",
		"status", statusCode,
		"message", message,
		"error", err,
	)

	if rw, ok := w.(*responseWriter); ok && rw.wroteHeader {
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)

	errorResp := transformer.TransformErrorResponse(statusCode, message)
	_ = json.NewEncoder(w).Encode(errorResp)
}

// lazyModelChain formats the model chain only when logged.
type lazyModelChain struct {
	chain []config.ModelConfig
}

func (l lazyModelChain) String() string {
	var names []string
	for _, m := range l.chain {
		names = append(names, fmt.Sprintf("%s(%s)", m.ModelID, m.Provider))
	}
	return fmt.Sprintf("[%s]", strings.Join(names, " "))
}
// lazyJSONTruncator delays JSON parsing and truncation until the log is actually formatted.
type lazyJSONTruncator struct {
	raw json.RawMessage
}

func (l lazyJSONTruncator) String() string {
	var m interface{}
	if err := json.Unmarshal(l.raw, &m); err != nil {
		return string(l.raw)
	}

	truncateRecursive(m)

	data, _ := json.Marshal(m)
	return string(data)
}

func truncateRecursive(v interface{}) {
	switch val := v.(type) {
	case map[string]interface{}:
		for k, item := range val {
			if s, ok := item.(string); ok {
				if len(s) > 50 {
					val[k] = s[:50] + "..."
				}
			} else {
				truncateRecursive(item)
			}
		}
	case []interface{}:
		for _, item := range val {
			truncateRecursive(item)
		}
	}
}
