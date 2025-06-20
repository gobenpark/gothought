package gothought

import (
	"context"
	"errors"

	"github.com/gobenpark/gothought/tool"
)

const (
	FinishReasonStop      = "stop"
	FinishReasonToolCalls = "tool_calls"
)

type LanguageModel struct {
	tools         map[string]tool.Tool
	provider      Provider
	messages      []Message
	maxIterations int // maxIterations default int values 10
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
	l.messages = append(l.messages, Message{
		Role:    "system",
		Message: prompt,
	})
	return l
}

// AIPrompt adds an AI-generated message to the conversation.
// It appends a new message with the "AI" role to the client's message list.
func (l *LanguageModel) AIPrompt(prompt string) *LanguageModel {
	l.messages = append(l.messages, Message{
		Role:    "AI",
		Message: prompt,
	})
	return l
}

// Prompt adds a custom message to the conversation.
// It appends the provided message with its specified role to the client's message list.
func (l *LanguageModel) Prompt(message Message) *LanguageModel {
	l.messages = append(l.messages, message)
	return l
}

// HumanPrompt adds a user message to the conversation.
// It appends a new message with the "user" role to the client's message list.
func (l *LanguageModel) HumanPrompt(prompt string) *LanguageModel {
	l.messages = append(l.messages, Message{
		Role:    "user",
		Message: prompt,
	})
	return l
}

// Q executes a query to the language model and returns the response.
// It manages tool calls through multiple iterations if necessary,
// up to the configured maximum number of iterations.
func (l *LanguageModel) Q(ctx context.Context) (*Message, error) {

	messages := l.messages

	for i := 0; i < l.maxIterations; i++ {
		response, finishReason, err := l.provider.Generate(ctx, l.tools, messages)
		if err != nil {
			return nil, err
		}

		switch finishReason {
		case FinishReasonStop:
			return response, nil
		case FinishReasonToolCalls:
			messages = append(messages, *response)

			for _, tl := range response.ToolCalls {
				tres, err := l.tools[tl.Function.Name].Call(ctx, tl.Function.Arguments)
				if err != nil {
					return nil, err
				}
				messages = append(messages, Message{
					Role:       "tool",
					ToolCallID: tl.ID,
					Message:    tres,
				})
			}
		}
	}
	return nil, errors.New("max iterations reached")
}

// QStream executes a streaming query to the language model.
// It checks if the provider supports streaming capabilities and
// processes the response through the provided callback function.
func (l *LanguageModel) QStream(ctx context.Context, callback func(Message) error) error {
	if p, ok := any(l.provider).(StreamingCapable); ok {
		return p.GenerateStreaming(ctx, l.tools, l.messages, callback)
	}

	return errors.New("streaming not supported for this provider")
}

// It takes a context and an interface object that defines the structure
// of the expected output. The function appends a schema prompt to the last message,
// processes the response from the provider, and parses the result into the provided object.
// This is particularly useful for getting structured, type-safe responses from the language model.
func (o *LanguageModel) QWith(ctx context.Context, oj interface{}) error {
	msgLen := len(o.messages)
	msg := o.messages[msgLen-1]

	msg.Message += "\n\n" + GenerateSchemaPrompt(oj)
	o.messages[msgLen-1] = msg

	res, _, err := o.provider.Generate(ctx, o.tools, o.messages)
	if err != nil {
		return err
	}

	if err := ParsePrompt(oj, res.Message); err != nil {
		return err
	}
	return nil
}
