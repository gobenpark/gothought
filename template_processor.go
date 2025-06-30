package gothought

import (
	"github.com/gobenpark/gothought/errors"
	"github.com/gobenpark/gothought/messages"
)

// TemplateProcessor handles template processing and error handling
type TemplateProcessor interface {
	ProcessSystemTemplate(template *PromptTemplate, data interface{}) (string, error)
	ProcessSystemTemplatef(templateStr string, data interface{}) (string, error)
	ProcessHumanTemplate(template *PromptTemplate, data interface{}) (string, error)
	ProcessHumanTemplatef(templateStr string, data interface{}) (string, error)
	CreateErrorMessage(role messages.Role, errorText string) messages.Message
}

// templateProcessor implements TemplateProcessor interface
type templateProcessor struct{}

// NewTemplateProcessor creates a new TemplateProcessor instance
func NewTemplateProcessor() TemplateProcessor {
	return &templateProcessor{}
}

// ProcessSystemTemplate processes a system template with data
func (tp *templateProcessor) ProcessSystemTemplate(template *PromptTemplate, data interface{}) (string, error) {
	if template == nil {
		return "", errors.NewValidationError("template", "template cannot be nil")
	}

	prompt, err := template.Execute(data)
	if err != nil {
		return "", errors.NewValidationError("template_execution", "system template execution failed: "+err.Error())
	}
	return prompt, nil
}

// ProcessSystemTemplatef processes a system template string with data
func (tp *templateProcessor) ProcessSystemTemplatef(templateStr string, data interface{}) (string, error) {
	if templateStr == "" {
		return "", errors.NewValidationError("template_string", "template string cannot be empty")
	}

	template, err := NewPromptTemplate("system_inline", templateStr)
	if err != nil {
		return "", errors.NewValidationError("template_creation", "failed to create system template: "+err.Error())
	}

	return tp.ProcessSystemTemplate(template, data)
}

// ProcessHumanTemplate processes a human template with data
func (tp *templateProcessor) ProcessHumanTemplate(template *PromptTemplate, data interface{}) (string, error) {
	if template == nil {
		return "", errors.NewValidationError("template", "template cannot be nil")
	}

	prompt, err := template.Execute(data)
	if err != nil {
		return "", errors.NewValidationError("template_execution", "human template execution failed: "+err.Error())
	}
	return prompt, nil
}

// ProcessHumanTemplatef processes a human template string with data
func (tp *templateProcessor) ProcessHumanTemplatef(templateStr string, data interface{}) (string, error) {
	if templateStr == "" {
		return "", errors.NewValidationError("template_string", "template string cannot be empty")
	}

	template, err := NewPromptTemplate("inline", templateStr)
	if err != nil {
		return "", errors.NewValidationError("template_creation", "failed to create human template: "+err.Error())
	}

	return tp.ProcessHumanTemplate(template, data)
}

// CreateErrorMessage creates a standardized error message for template failures
func (tp *templateProcessor) CreateErrorMessage(role messages.Role, errorText string) messages.Message {
	return messages.Message{
		Role:    role,
		Message: "[TEMPLATE_ERROR]: " + errorText,
	}
}
