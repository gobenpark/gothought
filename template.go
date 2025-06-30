package gothought

import (
	"bytes"
	"io"
	"text/template"

	"github.com/gobenpark/gothought/errors"
	"github.com/gobenpark/gothought/providers"
)

// PromptTemplate wraps Go's text/template for prompt generation with variable substitution.
// It provides a fluent interface for creating and executing templates with type safety and
// powerful template features like conditionals, loops, and custom functions.
type PromptTemplate struct {
	template *template.Template
	name     string
}

// NewPromptTemplate creates a new PromptTemplate from the given template string.
// The template uses Go's text/template syntax with {{.Variable}} placeholders.
//
// Example template string: "Hello {{.Name}}, you are a {{.Role}}."
//
// Returns an error if the template syntax is invalid.
func NewPromptTemplate(name, templateStr string) (*PromptTemplate, error) {
	if err := providers.ValidatePrompt(templateStr); err != nil {
		return nil, err
	}

	if name == "" {
		return nil, errors.NewValidationError("template_name", "template name cannot be empty")
	}

	tmpl, err := template.New(name).Parse(templateStr)
	if err != nil {
		return nil, errors.NewValidationError("template_syntax", "invalid template syntax: "+err.Error())
	}

	return &PromptTemplate{
		template: tmpl,
		name:     name,
	}, nil
}

// NewPromptTemplateWithFuncs creates a new PromptTemplate with custom functions.
// Custom functions can be used within templates for data transformation and logic.
//
// Example:
//
//	funcMap := template.FuncMap{
//	    "upper": strings.ToUpper,
//	    "add": func(a, b int) int { return a + b },
//	}
//	tmpl, err := NewPromptTemplateWithFuncs("example", "Hello {{upper .Name}}", funcMap)
func NewPromptTemplateWithFuncs(name, templateStr string, funcMap template.FuncMap) (*PromptTemplate, error) {
	if err := providers.ValidatePrompt(templateStr); err != nil {
		return nil, err
	}

	if name == "" {
		return nil, errors.NewValidationError("template_name", "template name cannot be empty")
	}

	if funcMap == nil {
		funcMap = template.FuncMap{}
	}

	tmpl, err := template.New(name).Funcs(funcMap).Parse(templateStr)
	if err != nil {
		return nil, errors.NewValidationError("template_syntax", "invalid template syntax: "+err.Error())
	}

	return &PromptTemplate{
		template: tmpl,
		name:     name,
	}, nil
}

// Execute renders the template with the provided data and returns the result as a string.
// The data can be a struct, map, or any other type that matches the template variables.
//
// Example with struct:
//
//	type User struct { Name string; Role string }
//	result, err := template.Execute(User{Name: "Alice", Role: "developer"})
//
// Example with map:
//
//	result, err := template.Execute(map[string]interface{}{
//	    "Name": "Bob",
//	    "Age": 30,
//	})
func (pt *PromptTemplate) Execute(data interface{}) (string, error) {
	var buf bytes.Buffer
	err := pt.template.Execute(&buf, data)
	if err != nil {
		return "", errors.NewValidationError("template_execution", "template execution failed: "+err.Error())
	}
	return buf.String(), nil
}

// ExecuteToWriter renders the template with the provided data and writes the result to the given writer.
// This is useful for streaming output or writing directly to files without intermediate string allocation.
func (pt *PromptTemplate) ExecuteToWriter(wr io.Writer, data interface{}) error {
	err := pt.template.Execute(wr, data)
	if err != nil {
		return errors.NewValidationError("template_execution", "template execution failed: "+err.Error())
	}
	return nil
}

// Name returns the name of the template.
func (pt *PromptTemplate) Name() string {
	return pt.name
}

// Clone creates a copy of the template that can be safely used in multiple goroutines.
// The original and cloned templates share the same parsed template tree but can be
// executed concurrently.
func (pt *PromptTemplate) Clone() (*PromptTemplate, error) {
	cloned, err := pt.template.Clone()
	if err != nil {
		return nil, errors.NewValidationError("template_clone", "failed to clone template: "+err.Error())
	}

	return &PromptTemplate{
		template: cloned,
		name:     pt.name + "_clone",
	}, nil
}
