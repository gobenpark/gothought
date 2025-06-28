package gothought

import (
	"context"

	"github.com/gobenpark/gothought/tool"
)

const (
	FinishReasonStop      = "stop"
	FinishReasonToolCalls = "tool_calls"
)

type LanguageModel struct {
	tools          map[string]tool.Tool
	provider       Provider
	messages       []Message
	maxIterations  int // maxIterations default int values 10
	contextManager ContextManager
}

func NewLanguageModel(p Provider, options ...Option) *LanguageModel {
	cli := &LanguageModel{
		provider:      p,
		maxIterations: 10,
		tools:         map[string]tool.Tool{},
	}

	for _, option := range options {
		option(cli)
	}

	// Initialize with default context manager if none provided
	if cli.contextManager == nil {
		cli.contextManager = NewContextManager(DefaultContextConfig())
	}

	return cli
}

// SetPrompts replaces the entire conversation history with a new set of messages.
// This allows for completely resetting or initializing the conversation context
// with predefined messages of various roles (system, user, AI, etc.).
func (l *LanguageModel) SetPrompts(prompts []Message) {
	l.messages = prompts
}

// AddTool registers a new tool with the language model.
// Tools allow the language model to perform actions or access external functionality
// during the conversation through function calling.
func (l *LanguageModel) AddTool(t tool.Tool) *LanguageModel {
	l.tools[t.Name()] = t
	return l
}

// SystemPrompt adds a system instruction message to the conversation.
// It appends a new message with the "system" role to the client's message list.
// System messages are typically used to set the behavior of the language model.
func (l *LanguageModel) SystemPrompt(prompt string) *LanguageModel {
	msg := Message{
		Role:    "system",
		Message: prompt,
	}
	l.messages = append(l.messages, msg)

	// Also add to context manager if available
	if l.contextManager != nil {
		l.contextManager.AddMessage(msg)
		// Sync with context manager's filtered messages
		l.messages = l.contextManager.GetMessages()
	}

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
	prompt, err := template.Execute(data)
	if err != nil {
		l.messages = append(l.messages, Message{
			Role:    "system",
			Message: "[TEMPLATE_ERROR]: " + err.Error(),
		})
		return l
	}
	return l.SystemPrompt(prompt)
}

// SystemPromptf adds a system message using a template string with variable substitution.
// This is a convenience method for one-time template usage without creating a PromptTemplate object.
func (l *LanguageModel) SystemPromptf(templateStr string, data interface{}) *LanguageModel {
	template, err := NewPromptTemplate("system_inline", templateStr)
	if err != nil {
		l.messages = append(l.messages, Message{
			Role:    "system",
			Message: "[TEMPLATE_ERROR]: " + err.Error(),
		})
		return l
	}
	return l.SystemPromptTemplate(template, data)
}

// AIPrompt adds an AI-generated message to the conversation.
// It appends a new message with the "AI" role to the client's message list.
func (l *LanguageModel) AIPrompt(prompt string) *LanguageModel {
	msg := Message{
		Role:    "assistant",
		Message: prompt,
	}
	}
	l.messages = append(l.messages, msg)

	// Also add to context manager if available
	if l.contextManager != nil {
		l.contextManager.AddMessage(msg)
		// Sync with context manager's filtered messages
		l.messages = l.contextManager.GetMessages()
	}

	return l
}

// Prompt adds a custom message to the conversation.
// It appends the provided message with its specified role to the client's message list.
func (l *LanguageModel) Prompt(message Message) *LanguageModel {
	l.messages = append(l.messages, message)

	// Also add to context manager if available
	if l.contextManager != nil {
		l.contextManager.AddMessage(message)
		// Sync with context manager's filtered messages
		l.messages = l.contextManager.GetMessages()
	}

	return l
}

// HumanPrompt adds a user message to the conversation.
// It appends a new message with the "user" role to the client's message list.
func (l *LanguageModel) HumanPrompt(prompt string) *LanguageModel {
	msg := Message{
		Role:    "user",
		Message: prompt,
	}
	l.messages = append(l.messages, msg)

	// Also add to context manager if available
	if l.contextManager != nil {
		l.contextManager.AddMessage(msg)
		// Sync with context manager's filtered messages
		l.messages = l.contextManager.GetMessages()
	}

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
	prompt, err := template.Execute(data)
	if err != nil {
		// For fluent API, we store the error and continue
		// The error will be caught during validation in Q() method
		l.messages = append(l.messages, Message{
			Role:    "user",
			Message: "[TEMPLATE_ERROR]: " + err.Error(),
		})
		return l
	}
	return l.HumanPrompt(prompt)
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
	template, err := NewPromptTemplate("inline", templateStr)
	if err != nil {
		l.messages = append(l.messages, Message{
			Role:    "user",
			Message: "[TEMPLATE_ERROR]: " + err.Error(),
		})
		return l
	}
	return l.HumanPromptTemplate(template, data)
}

// Q executes a query to the language model and returns the response.
// It manages tool calls through multiple iterations if necessary,
// up to the configured maximum number of iterations.
func (l *LanguageModel) Q(ctx context.Context) (*Message, error) {
	if ctx == nil {
		return nil, NewValidationError("context", "context cannot be nil")
	}

	if err := ValidateMessages(l.messages); err != nil {
		return nil, err
	}

	messages := l.messages

	for i := 0; i < l.maxIterations; i++ {
		response, finishReason, err := l.provider.Generate(ctx, l.tools, messages)
		if err != nil {
			return nil, NewProviderError("failed to generate response", err).WithContext("iteration", i)
		}

		switch finishReason {
		case FinishReasonStop:
			return response, nil
		case FinishReasonToolCalls:
			messages = append(messages, *response)

			for _, tl := range response.ToolCalls {
				tool, exists := l.tools[tl.Function.Name]
				if !exists {
					return nil, NewToolError(tl.Function.Name, "tool not found", nil)
				}

				tres, err := tool.Call(ctx, tl.Function.Arguments)
				if err != nil {
					return nil, NewToolError(tl.Function.Name, "tool execution failed", err)
				}
				messages = append(messages, Message{
					Role:       "tool",
					ToolCallID: tl.ID,
					Message:    tres,
				})
			}
		}
	}
	return nil, NewMaxIterationsError(l.maxIterations)
}

// QStream executes a streaming query to the language model.
// It checks if the provider supports streaming capabilities and
// processes the response through the provided callback function.
func (l *LanguageModel) QStream(ctx context.Context, callback func(Message) error) error {
	if ctx == nil {
		return NewValidationError("context", "context cannot be nil")
	}

	if callback == nil {
		return NewValidationError("callback", "callback function cannot be nil")
	}

	if err := ValidateMessages(l.messages); err != nil {
		return err
	}

	if p, ok := any(l.provider).(StreamingCapable); ok {
		if err := p.GenerateStreaming(ctx, l.tools, l.messages, callback); err != nil {
			return NewProviderError("streaming generation failed", err)
		}
		return nil
	}

	return NewProviderError("streaming not supported for this provider", nil)
}

// QWith takes a context and an interface object that defines the structure
// of the expected output. The function appends a schema prompt to the last message,
// processes the response from the provider, and parses the result into the provided object.
// This is particularly useful for getting structured, type-safe responses from the language model.
func (o *LanguageModel) QWith(ctx context.Context, oj interface{}) error {
	if ctx == nil {
		return NewValidationError("context", "context cannot be nil")
	}

	if oj == nil {
		return NewValidationError("object", "output object cannot be nil")
	}

	if err := ValidateMessages(o.messages); err != nil {
		return err
	}

	if len(o.messages) == 0 {
		return NewValidationError("messages", "no messages available for structured output")
	}

	msgLen := len(o.messages)
	msg := o.messages[msgLen-1]

	msg.Message += "\n\n" + GenerateSchemaPrompt(oj)
	o.messages[msgLen-1] = msg

	res, _, err := o.provider.Generate(ctx, o.tools, o.messages)
	if err != nil {
		return NewProviderError("failed to generate structured response", err)
	}

	if err := ParsePrompt(oj, res.Message); err != nil {
		return NewParsingError("failed to parse structured output", err)
	}
	return nil
}

// Context Management Methods

// EnableContextManagement enables automatic context management for the language model.
// When enabled, messages are automatically managed by the context manager according
// to configured limits and compression settings.
func (l *LanguageModel) EnableContextManagement() *LanguageModel {
	// Sync existing messages to context manager
	for _, msg := range l.messages {
		l.contextManager.AddMessage(msg)
	}
	return l
}

// SetContextManager sets a custom context manager for the language model.
func (l *LanguageModel) SetContextManager(cm ContextManager) *LanguageModel {
	l.contextManager = cm
	return l
}

// GetContextManager returns the current context manager.
func (l *LanguageModel) GetContextManager() ContextManager {
	return l.contextManager
}

// SaveConversation saves the current conversation to persistent storage with the given session ID.
func (l *LanguageModel) SaveConversation(sessionID string) error {
	// Sync current messages to context manager
	l.contextManager.Clear()
	for _, msg := range l.messages {
		if err := l.contextManager.AddMessage(msg); err != nil {
			return err
		}
	}
	return l.contextManager.SaveContext(sessionID)
}

// LoadConversation loads a conversation from persistent storage with the given session ID.
func (l *LanguageModel) LoadConversation(sessionID string) error {
	if err := l.contextManager.LoadContext(sessionID); err != nil {
		return err
	}
	l.messages = l.contextManager.GetMessages()
	return nil
}

// ClearConversation clears all messages from both the language model and context manager.
func (l *LanguageModel) ClearConversation() *LanguageModel {
	l.messages = make([]Message, 0)
	l.contextManager.Clear()
	return l
}

// CompressConversation compresses the conversation history to reduce token usage.
// This method uses the provider to summarize older messages while preserving
// system messages and recent conversation history.
func (l *LanguageModel) CompressConversation(ctx context.Context, maxTokens int) error {
	// Sync current messages to context manager
	l.contextManager.Clear()
	for _, msg := range l.messages {
		if err := l.contextManager.AddMessage(msg); err != nil {
			return err
		}
	}

	// Perform compression
	if err := l.contextManager.CompressContext(ctx, l.provider, maxTokens); err != nil {
		return err
	}

	// Update messages from compressed context
	l.messages = l.contextManager.GetMessages()
	return nil
}

// GetConversationTokenCount estimates the total token count for the current conversation.
func (l *LanguageModel) GetConversationTokenCount(modelName string) (int, error) {
	// Sync current messages to context manager for accurate counting
	l.contextManager.Clear()
	for _, msg := range l.messages {
		if err := l.contextManager.AddMessage(msg); err != nil {
			return 0, err
		}
	}
	return l.contextManager.GetTokenCount(modelName)
}

// GetOptimizedMessages returns messages optimized for the given token constraints.
// This method filters and potentially compresses messages to fit within the specified limits.
func (l *LanguageModel) GetOptimizedMessages(maxTokens int, modelName string) ([]Message, error) {
	// Sync current messages to context manager
	l.contextManager.Clear()
	for _, msg := range l.messages {
		if err := l.contextManager.AddMessage(msg); err != nil {
			return nil, err
		}
	}
	return l.contextManager.GetFilteredMessages(maxTokens, modelName)
}
