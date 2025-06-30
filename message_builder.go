package gothought

import (
	"github.com/gobenpark/gothought/memory"
	"github.com/gobenpark/gothought/messages"
)

// MessageBuilder handles message construction and adds them to memory manager
type MessageBuilder interface {
	SystemPrompt(prompt string) MessageBuilder
	AIPrompt(prompt string) MessageBuilder
	HumanPrompt(prompt string) MessageBuilder
	SystemPromptTemplate(template *PromptTemplate, data interface{}) MessageBuilder
	SystemPromptf(templateStr string, data interface{}) MessageBuilder
	HumanPromptTemplate(template *PromptTemplate, data interface{}) MessageBuilder
	HumanPromptf(templateStr string, data interface{}) MessageBuilder
	AddMessage(message messages.Message) MessageBuilder
	SetPrompts(prompts []messages.Message)
}

// messageBuilder implements MessageBuilder interface
type messageBuilder struct {
	memoryManager memory.MemoryManager
}

// NewMessageBuilder creates a new MessageBuilder instance
func NewMessageBuilder(memoryManager memory.MemoryManager) MessageBuilder {
	return &messageBuilder{
		memoryManager: memoryManager,
	}
}

// SystemPrompt adds a system instruction message to the conversation
func (mb *messageBuilder) SystemPrompt(prompt string) MessageBuilder {
	msg := messages.Message{
		Role:    "system",
		Message: prompt,
	}
	mb.memoryManager.AddMessage(msg)
	return mb
}

// AIPrompt adds an AI-generated message to the conversation
func (mb *messageBuilder) AIPrompt(prompt string) MessageBuilder {
	msg := messages.Message{
		Role:    "assistant",
		Message: prompt,
	}
	mb.memoryManager.AddMessage(msg)
	return mb
}

// HumanPrompt adds a user message to the conversation
func (mb *messageBuilder) HumanPrompt(prompt string) MessageBuilder {
	msg := messages.Message{
		Role:    "user",
		Message: prompt,
	}
	mb.memoryManager.AddMessage(msg)
	return mb
}

// SystemPromptTemplate adds a system message using a template with variable substitution
func (mb *messageBuilder) SystemPromptTemplate(template *PromptTemplate, data interface{}) MessageBuilder {
	prompt, err := template.Execute(data)
	if err != nil {
		mb.memoryManager.AddMessage(messages.Message{
			Role:    "system",
			Message: "[TEMPLATE_ERROR]: " + err.Error(),
		})
		return mb
	}
	return mb.SystemPrompt(prompt)
}

// SystemPromptf adds a system message using a template string with variable substitution
func (mb *messageBuilder) SystemPromptf(templateStr string, data interface{}) MessageBuilder {
	template, err := NewPromptTemplate("system_inline", templateStr)
	if err != nil {
		mb.memoryManager.AddMessage(messages.Message{
			Role:    "system",
			Message: "[TEMPLATE_ERROR]: " + err.Error(),
		})
		return mb
	}
	return mb.SystemPromptTemplate(template, data)
}

// HumanPromptTemplate adds a user message using a template with variable substitution
func (mb *messageBuilder) HumanPromptTemplate(template *PromptTemplate, data interface{}) MessageBuilder {
	prompt, err := template.Execute(data)
	if err != nil {
		mb.memoryManager.AddMessage(messages.Message{
			Role:    "user",
			Message: "[TEMPLATE_ERROR]: " + err.Error(),
		})
		return mb
	}
	return mb.HumanPrompt(prompt)
}

// HumanPromptf adds a user message using a template string with variable substitution
func (mb *messageBuilder) HumanPromptf(templateStr string, data interface{}) MessageBuilder {
	template, err := NewPromptTemplate("inline", templateStr)
	if err != nil {
		mb.memoryManager.AddMessage(messages.Message{
			Role:    "user",
			Message: "[TEMPLATE_ERROR]: " + err.Error(),
		})
		return mb
	}
	return mb.HumanPromptTemplate(template, data)
}

// AddMessage adds a custom message to the conversation
func (mb *messageBuilder) AddMessage(message messages.Message) MessageBuilder {
	mb.memoryManager.AddMessage(message)
	return mb
}

// SetPrompts replaces the entire conversation history with a new set of messages
func (mb *messageBuilder) SetPrompts(prompts []messages.Message) {
	mb.memoryManager.Clear()
	for _, msg := range prompts {
		mb.memoryManager.AddMessage(msg)
	}
}
