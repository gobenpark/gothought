package gothought

import (
	"context"

	"github.com/gobenpark/gothought/errors"
	"github.com/gobenpark/gothought/memory"
	"github.com/gobenpark/gothought/messages"
	"github.com/gobenpark/gothought/tools"
	"go.uber.org/zap"
)

type LanguageModel struct {
	tools          map[string]tools.Tool
	provider       Provider
	maxIterations  int // maxIterations default int values 10
	memoryManager  memory.MemoryManager
	messageBuilder MessageBuilder
	queryExecutor  QueryExecutor
	logger         *zap.Logger // logger for debug and operational logging
}

func NewLanguageModel(p Provider, options ...Option) *LanguageModel {
	cli := &LanguageModel{
		provider:      p,
		maxIterations: 10,
		tools:         map[string]tools.Tool{},
	}

	for _, option := range options {
		option(cli)
	}

	// Initialize with default memory manager if none provided
	if cli.memoryManager == nil {
		cli.memoryManager = memory.NewMemoryManager(memory.DefaultMemoryConfig())
	}

	// Initialize with noop logger if none provided
	if cli.logger == nil {
		cli.logger = zap.NewNop()
	}

	// Initialize components
	cli.messageBuilder = NewMessageBuilder(cli.memoryManager, cli.logger)
	cli.queryExecutor = NewQueryExecutor(cli.provider, cli.tools, cli.maxIterations, cli.memoryManager, cli.messageBuilder, cli.logger)

	return cli
}

// SetPrompts replaces the entire conversation history with a new set of messages.
// This allows for completely resetting or initializing the conversation context
// with predefined messages of various roles (system, user, AI, etc.).
func (l *LanguageModel) SetPrompts(prompts []messages.Message) {
	l.messageBuilder.SetPrompts(prompts)
}

// AddTool registers a new tool with the language model.
// Tools allow the language model to perform actions or access external functionality
// during the conversation through function calling.
func (l *LanguageModel) AddTool(t tools.Tool) *LanguageModel {
	l.tools[t.Name()] = t
	// Update query executor with new tools
	l.queryExecutor = NewQueryExecutor(l.provider, l.tools, l.maxIterations, l.memoryManager, l.messageBuilder, l.logger)
	return l
}

// SystemPrompt adds a system instruction message to the conversation.
// It appends a new message with the "system" role to the client's message list.
// System messages are typically used to set the behavior of the language model.
func (l *LanguageModel) SystemPrompt(prompt string) *LanguageModel {
	l.logger.Debug("Adding system prompt", zap.String("prompt", prompt))
	l.messageBuilder.SystemPrompt(prompt)
	return l
}

// SystemPromptTemplate adds a system message using a template with variable substitution.
// The template uses Go's text/template syntax with {{.Variable}} placeholders.
//
// Example:
//
//	template, _ := NewPromptTemplate("system", "You are a {{.Role}} assistant. Your expertise is in {{.Domain}}.")
//	model.SystemPromptTemplate(template, map[string]string{"Role": "helpful", "Domain": "programming"})
func (l *LanguageModel) SystemPromptTemplate(template *PromptTemplate, data interface{}) *LanguageModel {
	l.messageBuilder.SystemPromptTemplate(template, data)
	return l
}

// SystemPromptf adds a system message using a template string with variable substitution.
// This is a convenience method for one-time template usage without creating a PromptTemplate object.
func (l *LanguageModel) SystemPromptf(templateStr string, data interface{}) *LanguageModel {
	l.messageBuilder.SystemPromptf(templateStr, data)
	return l
}

// AIPrompt adds an AI-generated message to the conversation.
// It appends a new message with the "assistant" role to the client's message list.
func (l *LanguageModel) AIPrompt(prompt string) *LanguageModel {
	l.logger.Debug("Adding AI prompt", zap.String("prompt", prompt))
	l.messageBuilder.AIPrompt(prompt)
	return l
}

// Prompt adds a custom message to the conversation.
// It appends the provided message with its specified role to the client's message list.
func (l *LanguageModel) Prompt(message messages.Message) *LanguageModel {
	l.messageBuilder.AddMessage(message)
	return l
}

// HumanPrompt adds a user message to the conversation.
// It appends a new message with the "user" role to the client's message list.
func (l *LanguageModel) HumanPrompt(prompt string) *LanguageModel {
	l.logger.Debug("Adding human prompt", zap.String("prompt", prompt))
	l.messageBuilder.HumanPrompt(prompt)
	return l
}

// HumanPromptTemplate adds a user message using a template with variable substitution.
// The template uses Go's text/template syntax with {{.Variable}} placeholders.
//
// Example:
//
//	template, _ := NewPromptTemplate("greeting", "Hello {{.Name}}, you are a {{.Role}}")
//	model.HumanPromptTemplate(template, User{Name: "Alice", Role: "developer"})
func (l *LanguageModel) HumanPromptTemplate(template *PromptTemplate, data interface{}) *LanguageModel {
	l.messageBuilder.HumanPromptTemplate(template, data)
	return l
}

// HumanPromptf adds a user message using a template string with variable substitution.
// This is a convenience method for one-time template usage without creating a PromptTemplate object.
//
// Example:
//
//	model.HumanPromptf("Hello {{.Name}}, you are {{.Age}} years old", map[string]interface{}{
//	    "Name": "Bob",
//	    "Age": 30,
//	})
func (l *LanguageModel) HumanPromptf(templateStr string, data interface{}) *LanguageModel {
	l.messageBuilder.HumanPromptf(templateStr, data)
	return l
}

// Q executes a query to the language model and returns the response.
// It manages tool calls through multiple iterations if necessary,
// up to the configured maximum number of iterations.
func (l *LanguageModel) Q(ctx context.Context) (*messages.Message, error) {
	l.logger.Debug("Starting query execution")
	response, err := l.queryExecutor.Execute(ctx)
	if err != nil {
		l.logger.Error("Query execution failed", zap.Error(err))
	} else {
		l.logger.Debug("Query execution completed successfully")
	}
	return response, err
}

// QStream executes a streaming query to the language model.
// It checks if the provider supports streaming capabilities and
// processes the response through the provided callback function.
func (l *LanguageModel) QStream(ctx context.Context, callback func(messages.Message) error) error {
	err := l.queryExecutor.ExecuteStreaming(ctx, callback)
	if err != nil {
		l.logger.Error("Query execution failed", zap.Error(err))
	} else {
		l.logger.Debug("Query execution completed successfully")
	}

	return err
}

// QWith takes a context and an interface object that defines the structure
// of the expected output. The function appends a schema prompt to the last message,
// processes the response from the provider, and parses the result into the provided object.
// This is particularly useful for getting structured, type-safe responses from the language model.
func (l *LanguageModel) QWith(ctx context.Context, obj interface{}) error {
	return l.queryExecutor.ExecuteWithStructuredOutput(ctx, obj)
}

// Context Management Methods

// EnableContextManagement enables automatic context management for the language model.
// When enabled, messages are automatically managed by the context manager according
// to configured limits and compression settings.
// Note: Context management is now always enabled by default.
func (l *LanguageModel) EnableContextManagement() *LanguageModel {
	// Context management is always enabled in this architecture
	return l
}

// SetMemoryManager sets a custom context manager for the language model.
func (l *LanguageModel) SetMemoryManager(cm memory.MemoryManager) *LanguageModel {
	l.memoryManager = cm
	return l
}

// GetMemoryManager returns the current context manager.
func (l *LanguageModel) GetMemoryManager() memory.MemoryManager {
	return l.memoryManager
}

// SaveConversation saves the current conversation to persistent storage with the given session ID.
func (l *LanguageModel) SaveConversation(sessionID string) error {
	if l.memoryManager == nil {
		return errors.NewValidationError("memory_manager", "memory manager not configured")
	}
	return l.memoryManager.SaveContext(sessionID)
}

// LoadConversation loads a conversation from persistent storage with the given session ID.
func (l *LanguageModel) LoadConversation(sessionID string) error {
	if l.memoryManager == nil {
		return errors.NewValidationError("memory_manager", "memory manager not configured")
	}
	return l.memoryManager.LoadContext(sessionID)
}

// ClearConversation clears all messages from the context manager.
func (l *LanguageModel) ClearConversation() *LanguageModel {
	if l.memoryManager != nil {
		l.memoryManager.Clear()
	}
	return l
}

// CompressConversation compresses the conversation history to reduce token usage.
// This method uses the provider to summarize older messages while preserving
// system messages and recent conversation history.
//func (l *LanguageModel) CompressConversation(ctx context.Context, maxTokens int) error {
//	return l.memoryManager.CompressContext(ctx, l.provider, maxTokens)
//}

// GetConversationTokenCount estimates the total token count for the current conversation.
func (l *LanguageModel) GetConversationTokenCount(modelName string) (int, error) {
	if l.memoryManager == nil {
		return 0, errors.NewValidationError("memory_manager", "memory manager not configured")
	}
	return l.memoryManager.GetTokenCount(modelName)
}

// GetOptimizedMessages returns messages optimized for the given token constraints.
// This method filters and potentially compresses messages to fit within the specified limits.
func (l *LanguageModel) GetOptimizedMessages(maxTokens int, modelName string) ([]messages.Message, error) {
	if l.memoryManager == nil {
		return []messages.Message{}, errors.NewValidationError("memory_manager", "memory manager not configured")
	}
	return l.memoryManager.GetFilteredMessages(maxTokens, modelName)
}
