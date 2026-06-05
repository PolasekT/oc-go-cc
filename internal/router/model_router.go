// Package router defines HTTP route registration and middleware chaining,
// as well as model selection based on request scenarios.
package router

import (
	"fmt"
	"log/slog"

	"oc-go-cc/internal/config"
)

// ModelRouter handles model selection based on scenarios.
type ModelRouter struct {
	atomic *config.AtomicConfig
	logger *slog.Logger
}

// NewModelRouter creates a new model router.
func NewModelRouter(atomic *config.AtomicConfig, logger *slog.Logger) *ModelRouter {
	return &ModelRouter{
		atomic: atomic,
		logger: logger,
	}
}

// RouteResult contains the selected model and fallback chain.
type RouteResult struct {
	Primary   config.ModelConfig
	Fallbacks []config.ModelConfig
	Scenario  Scenario
	Reason    string
}

// resolveRequestedModel checks if the user-specified model should override
// scenario-based routing. Returns the route result and true if it matched,
// or zero value and false if scenario routing should proceed normally.
func (r *ModelRouter) resolveRequestedModel(cfg *config.Config, requestedModel string, effort string) (RouteResult, bool) {
	r.logger.Debug("resolving requested model", "model", requestedModel, "effort", effort, "respect", cfg.RespectRequestedModel, "use_effort", cfg.RespectRequestedModelUseEffort)
	if !cfg.RespectRequestedModel || requestedModel == "" {
		return RouteResult{}, false
	}

	// If respect_requested_model_use_effort is enabled, try effort-based override first
	if cfg.RespectRequestedModelUseEffort && effort != "" {
		if result, ok := r.RouteWithEffortOverride(requestedModel, effort); ok {
			r.logger.Debug("resolved requested model via effort override", "model", requestedModel, "effort", effort, "target", result.Primary.ModelID)
			result.Reason = fmt.Sprintf("req_model_effort_override(%s.%s)", requestedModel, effort)
			return result, true
		}
	}


	// Look up the requested model in config to inherit its settings
	primary, ok := cfg.Models[requestedModel]
	reason := fmt.Sprintf("req_model(%s)", requestedModel)
	if !ok {
		r.logger.Debug("requested model not in models config, using as-is", "model", requestedModel)
		// Unknown model — create a bare config and inherit defaults
		primary = config.ModelConfig{
			Provider: "opencode-go",
			ModelID:  requestedModel,
		}
		if def, ok := cfg.Models["default"]; ok {
			primary.Temperature = def.Temperature
			primary.MaxTokens = def.MaxTokens
		}
		reason = fmt.Sprintf("req_model_unknown(%s)", requestedModel)
	} else {
		r.logger.Debug("requested model found in models config", "model", requestedModel, "target", primary.ModelID)
	}

	fallbacks := cfg.Fallbacks["default"]

	r.logger.Debug("resolved requested model successfully", "modelID", primary.ModelID, "provider", primary.Provider)
	return RouteResult{
		Primary:   primary,
		Fallbacks: fallbacks,
		Scenario:  ScenarioDefault,
		Reason:    reason,
	}, true
}

// RouteWithEffortOverride checks if the requested model and effort match a model_effort_overrides entry.
//
// Returns the override RouteResult and true if matched, or a zero value and
// false if no match is found.
func (r *ModelRouter) RouteWithEffortOverride(requestedModel string, effort string) (RouteResult, bool) {
	if effort == "" {
		return RouteResult{}, false
	}
	cfg := r.atomic.Get()
	if cfg.ModelEffortOverrides == nil {
		r.logger.Debug("no model_effort_overrides configured")
		return RouteResult{}, false
	}
	
	modelMap, ok := cfg.ModelEffortOverrides[requestedModel]
	if !ok {
		r.logger.Debug("no effort overrides for requested model", "model", requestedModel)
		return RouteResult{}, false
	}
	
	override, ok := modelMap[effort]
	if !ok {
		r.logger.Debug("no matching effort for model", "model", requestedModel, "effort", effort)
		return RouteResult{}, false
	}
	
	fallbacks := cfg.Fallbacks[requestedModel]
	if len(fallbacks) == 0 {
		fallbacks = cfg.Fallbacks["default"]
	}
	
	return RouteResult{
		Primary:   override,
		Fallbacks: fallbacks,
		Scenario:  ScenarioOverride,
		Reason:    fmt.Sprintf("effort_override(%s.%s)", requestedModel, effort),
	}, true
}

// Route determines which model to use for a request.
// If respect_requested_model is enabled and requestedModel is provided, it overrides scenario-based routing.
func (r *ModelRouter) Route(messages []MessageContent, tokenCount int, requestedModel string, effort string) (RouteResult, error) {
	cfg := r.atomic.Get()

	if result, ok := r.resolveRequestedModel(cfg, requestedModel, effort); ok {
		return result, nil
	}

	// Otherwise, use scenario-based routing
	result := DetectScenario(messages, tokenCount, cfg)

	scenarioKey := string(result.Scenario)
	r.logger.Debug("detected scenario", "scenario", scenarioKey, "effort", effort, "enable_effort_routing", cfg.EnableEffortScenarioRouting)

	// If enable_effort_scenario_routing is enabled, try effort-based override for the scenario
	if cfg.EnableEffortScenarioRouting && effort != "" {
		if effortResult, ok := r.RouteWithEffortOverride(scenarioKey, effort); ok {
			r.logger.Debug("resolved scenario via effort override", "scenario", scenarioKey, "effort", effort, "target", effortResult.Primary.ModelID)
			return effortResult, nil
		}
	}

	// Get primary model for scenario
	primary, ok := cfg.Models[scenarioKey]
	if !ok {
		// Fall back to default if scenario model not configured
		primary, ok = cfg.Models["default"]
		if !ok {
			return RouteResult{}, fmt.Errorf("no default model configured")
		}
	}

	// Get fallbacks for scenario
	fallbacks := cfg.Fallbacks[scenarioKey]
	if len(fallbacks) == 0 {
		// Fall back to default fallbacks
		fallbacks = cfg.Fallbacks["default"]
	}

	return RouteResult{
		Primary:   primary,
		Fallbacks: fallbacks,
		Scenario:  result.Scenario,
		Reason:    fmt.Sprintf("scenario(%s: %s)", scenarioKey, result.Reason),
	}, nil
}

// IsStreamingScenarioRoutingEnabled returns whether streaming requests should use
// scenario-based routing instead of always routing to the fast model.
func (r *ModelRouter) IsStreamingScenarioRoutingEnabled() bool {
	return r.atomic.Get().EnableStreamingScenarioRouting
}

// RouteWithOverride checks if the requested model matches a model_overrides entry.
//
// When matched, the returned RouteResult uses the override ModelConfig as the
// primary. The fallback chain is fallbacks[<requestedModel>], falling back to
// fallbacks["default"] when the override key has no entry (matching the
// behavior of Route and RouteForStreaming). The caller (MessagesHandler) is
// expected to merge a scenario-derived safety-net chain on top.
//
// Returns the override RouteResult and true if matched, or a zero value and
// false if the requested model has no entry in model_overrides.
func (r *ModelRouter) RouteWithOverride(requestedModel string) (RouteResult, bool) {
	cfg := r.atomic.Get()
	if cfg.ModelOverrides == nil {
		return RouteResult{}, false
	}
	override, ok := cfg.ModelOverrides[requestedModel]
	if !ok {
		return RouteResult{}, false
	}
	fallbacks := cfg.Fallbacks[requestedModel]
	if len(fallbacks) == 0 {
		fallbacks = cfg.Fallbacks["default"]
	}
	return RouteResult{
		Primary:   override,
		Fallbacks: fallbacks,
		Scenario:  ScenarioOverride,
		Reason:    fmt.Sprintf("model_override(%s)", requestedModel),
	}, true
}

// GetModelChain returns the full chain of models to try (primary + fallbacks).
func (rr *RouteResult) GetModelChain() []config.ModelConfig {
	chain := []config.ModelConfig{rr.Primary}
	chain = append(chain, rr.Fallbacks...)
	return chain
}

// RouteForStreaming determines which model to use for streaming requests.
// Prioritizes fast TTFT (time-to-first-token) over capability.
// If respect_requested_model is enabled and requestedModel is provided, it overrides scenario-based routing.
func (r *ModelRouter) RouteForStreaming(messages []MessageContent, tokenCount int, requestedModel string, effort string) RouteResult {
	cfg := r.atomic.Get()

	if result, ok := r.resolveRequestedModel(cfg, requestedModel, effort); ok {
		return result
	}

	// Otherwise, use scenario-based routing for streaming
	result := RouteForStreaming(messages, tokenCount, cfg)

	scenarioKey := string(result.Scenario)
	r.logger.Debug("detected scenario for streaming", "scenario", scenarioKey, "effort", effort, "enable_effort_routing", cfg.EnableEffortScenarioRouting)

	// If enable_effort_scenario_routing is enabled, try effort-based override for the scenario
	if cfg.EnableEffortScenarioRouting && effort != "" {
		if effortResult, ok := r.RouteWithEffortOverride(scenarioKey, effort); ok {
			r.logger.Debug("resolved streaming scenario via effort override", "scenario", scenarioKey, "effort", effort, "target", effortResult.Primary.ModelID)
			return effortResult
		}
	}

	// Get primary model for scenario
	primary, ok := cfg.Models[scenarioKey]
	if !ok {
		// Fall back to fast scenario if not configured
		primary, ok = cfg.Models["fast"]
		if !ok {
			// Fall back to default
			primary = cfg.Models["default"]
		}
	}

	// Get fallbacks for scenario
	fallbacks := cfg.Fallbacks[scenarioKey]
	if len(fallbacks) == 0 {
		// Fall back to fast fallbacks
		fallbacks = cfg.Fallbacks["fast"]
	}

	return RouteResult{
		Primary:   primary,
		Fallbacks: fallbacks,
		Scenario:  result.Scenario,
		Reason:    fmt.Sprintf("scenario(%s: %s)", scenarioKey, result.Reason),
	}
}
