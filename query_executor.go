package gothought

import (
	"context"
	"fmt"

	"github.com/gobenpark/gothought/errors"
	"github.com/gobenpark/gothought/memory"
	"github.com/gobenpark/gothought/messages"
	"github.com/gobenpark/gothought/providers"
	"github.com/gobenpark/gothought/tools"
	"go.uber.org/zap"
)

// QueryExecutor handles query execution and tool calling
type QueryExecutor interface {
	Execute(ctx context.Context) (*messages.Message, error)
	ExecuteStreaming(ctx context.Context, callback func(messages.Message) error) error
	ExecuteWithStructuredOutput(ctx context.Context, obj interface{}) error
}

// queryExecutor implements QueryExecutor interface
type queryExecutor struct {
	provider       Provider
	tools          map[string]tools.Tool
	maxIterations  int
	memoryManager  memory.MemoryManager
	messageBuilder MessageBuilder
	logger         *zap.Logger
}

// NewQueryExecutor creates a new QueryExecutor instance
func NewQueryExecutor(provider Provider, tools map[string]tools.Tool, maxIterations int, memoryManager memory.MemoryManager, messageBuilder MessageBuilder, logger *zap.Logger) QueryExecutor {
	return &queryExecutor{
		provider:       provider,
		tools:          tools,
		maxIterations:  maxIterations,
		memoryManager:  memoryManager,
		messageBuilder: messageBuilder,
		logger:         logger,
	}
}

// Execute runs a query to the language model and returns the response
func (qe *queryExecutor) Execute(ctx context.Context) (*messages.Message, error) {
	if ctx == nil {
		return nil, errors.NewValidationError("context", "context cannot be nil")
	}

	msgs := qe.memoryManager.GetMessages()
	if err := providers.ValidateMessages(msgs); err != nil {
		return nil, err
	}

	for i := 0; i < qe.maxIterations; i++ {
		qe.logger.Debug("Starting provider generation", zap.Int("iteration", i+1), zap.Int("max_iterations", qe.maxIterations))
		response, finishReason, err := qe.provider.Generate(ctx, qe.tools, msgs)
		if err != nil {
			qe.logger.Error("Provider generation failed", zap.Error(err), zap.Int("iteration", i+1))
			return nil, errors.NewProviderError("failed to generate response", err).WithContext("iteration", i)
		}

		qe.logger.Debug("Provider response received", zap.String("finish_reason", finishReason), zap.Int("iteration", i+1))

		switch finishReason {
		case messages.FinishReasonStop:
			qe.logger.Debug("Query completed with stop reason")
			return response, nil
		case messages.FinishReasonToolCalls:
			qe.logger.Debug("Processing tool calls", zap.Int("tool_calls_count", len(response.ToolCalls)))
			msgs = append(msgs, *response)
			qe.memoryManager.AddMessage(*response)

			if err := qe.handleToolCalls(ctx, response, &msgs); err != nil {
				return nil, err
			}
		}
	}
	return nil, errors.NewMaxIterationsError(qe.maxIterations)
}

// ExecuteStreaming executes a streaming query to the language model
func (qe *queryExecutor) ExecuteStreaming(ctx context.Context, callback func(messages.Message) error) error {
	if ctx == nil {
		return errors.NewValidationError("context", "context cannot be nil")
	}

	if callback == nil {
		return errors.NewValidationError("callback", "callback function cannot be nil")
	}

	msgs := qe.memoryManager.GetMessages()
	if err := providers.ValidateMessages(msgs); err != nil {
		return err
	}

	// Check if provider supports streaming
	streamingProvider, ok := any(qe.provider).(StreamingCapable)
	if !ok {
		qe.logger.Error("Provider does not support streaming")
		return errors.NewProviderError("streaming not supported for this provider", nil)
	}

	// If tools are available, we need to handle potential tool calls
	if len(qe.tools) > 0 {
		qe.logger.Debug("Tools detected, using hybrid approach for streaming")
		return qe.executeStreamingWithTools(ctx, streamingProvider, msgs, callback)
	}

	// No tools, direct streaming
	qe.logger.Debug("No tools, using direct streaming")
	if err := streamingProvider.GenerateStreaming(ctx, qe.tools, msgs, callback); err != nil {
		qe.logger.Error("Streaming generation failed", zap.Error(err))
		return errors.NewProviderError("streaming generation failed", err)
	}
	qe.logger.Debug("Streaming execution completed successfully")
	return nil
}

// executeStreamingWithTools handles streaming when tools are available
func (qe *queryExecutor) executeStreamingWithTools(ctx context.Context, streamingProvider StreamingCapable, msgs []messages.Message, callback func(messages.Message) error) error {
	// Handle tool calls in a loop similar to Execute
	for i := 0; i < qe.maxIterations; i++ {
		qe.logger.Debug("Starting iteration", zap.Int("iteration", i+1))

		// First, do a non-streaming call to check if it's a tool call
		response, finishReason, err := qe.provider.Generate(ctx, qe.tools, msgs)
		if err != nil {
			qe.logger.Error("Provider generation failed", zap.Error(err), zap.Int("iteration", i+1))
			return errors.NewProviderError("failed to generate response", err).WithContext("iteration", i)
		}

		switch finishReason {
		case messages.FinishReasonStop:
			// This is the final response - stream it
			qe.logger.Debug("Final response detected, streaming to user")

			// Store the response first
			qe.memoryManager.AddMessage(*response)

			// Re-run with streaming for the final response only
			if err := streamingProvider.GenerateStreaming(ctx, qe.tools, msgs, callback); err != nil {
				qe.logger.Error("Streaming generation failed", zap.Error(err))
				return errors.NewProviderError("streaming generation failed", err)
			}

			qe.logger.Debug("Streaming execution completed successfully")
			return nil

		case messages.FinishReasonToolCalls:
			// Process tool calls (don't stream these)
			qe.logger.Debug("Processing tool calls", zap.Int("tool_calls_count", len(response.ToolCalls)))
			msgs = append(msgs, *response)
			qe.memoryManager.AddMessage(*response)

			if err := qe.handleToolCalls(ctx, response, &msgs); err != nil {
				return err
			}
			// Continue to next iteration
		}
	}

	return errors.NewMaxIterationsError(qe.maxIterations)
}

// ExecuteWithStructuredOutput executes a query and parses the result into the provided object
func (qe *queryExecutor) ExecuteWithStructuredOutput(ctx context.Context, obj interface{}) error {
	if ctx == nil {
		return errors.NewValidationError("context", "context cannot be nil")
	}

	if obj == nil {
		return errors.NewValidationError("object", "output object cannot be nil")
	}

	msgs := qe.memoryManager.GetMessages()
	if err := providers.ValidateMessages(msgs); err != nil {
		return err
	}

	if len(msgs) == 0 {
		return errors.NewValidationError("messages", "no messages available for structured output")
	}

	// Create a copy of messages for this specific request
	msgsCopy := make([]messages.Message, len(msgs))
	copy(msgsCopy, msgs)

	// Modify the last message to include schema prompt
	msgLen := len(msgsCopy)
	msgsCopy[msgLen-1].Message += "\n\n" + GenerateSchemaPrompt(obj)

	res, _, err := qe.provider.Generate(ctx, qe.tools, msgsCopy)
	if err != nil {
		return errors.NewProviderError("failed to generate structured response", err)
	}

	if err := ParsePrompt(obj, res.Message); err != nil {
		return errors.NewParsingError("failed to parse structured output", err)
	}
	return nil
}

// handleToolCalls processes tool calls from the language model response
func (qe *queryExecutor) handleToolCalls(ctx context.Context, response *messages.Message, msgs *[]messages.Message) error {
	for _, tl := range response.ToolCalls {
		qe.logger.Debug("Executing tool call",
			zap.String("tool_name", tl.Function.Name),
			zap.String("tool_id", tl.ID),
			zap.String("arguments", tl.Function.Arguments),
		)

		tool, exists := qe.tools[tl.Function.Name]
		if !exists {
			qe.logger.Error("Tool not found", zap.String("tool_name", tl.Function.Name))
			return errors.NewToolError(tl.Function.Name, "tool not found", nil)
		}

		tres, err := tool.Call(ctx, tl.Function.Arguments)
		if err != nil {
			qe.logger.Error("Tool execution failed",
				zap.String("tool_name", tl.Function.Name),
				zap.Error(err),
			)
			return errors.NewToolError(tl.Function.Name, "tool execution failed", err)
		}

		qe.logger.Debug("Tool execution completed",
			zap.String("tool_name", tl.Function.Name),
			zap.String("result_length", fmt.Sprintf("%d chars", len(tres))),
		)

		toolMsg := messages.Message{
			Role:       "tool",
			ToolCallID: tl.ID,
			Message:    tres,
		}
		*msgs = append(*msgs, toolMsg)
		qe.memoryManager.AddMessage(toolMsg)
	}
	return nil
}
