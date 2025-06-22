package gothought

import (
	"fmt"
	"strings"

	"github.com/go-playground/validator/v10"
)

var validate *validator.Validate

func init() {
	validate = validator.New()
	validate.RegisterValidation("validrole", validateRole)
	validate.RegisterValidation("nonempty", validateNonEmpty)
}

type ValidatedMessage struct {
	Role       string      `validate:"required,validrole" json:"role"`
	Message    string      `validate:"max=50000" json:"message"`
	ToolCallID string      `json:"tool_call_id,omitempty"`
	ToolCalls  []ToolCalls `validate:"dive" json:"tool_calls,omitempty"`
}

type ValidatedToolCall struct {
	ID       string `validate:"required,nonempty" json:"id"`
	Type     string `validate:"required,eq=function" json:"type"`
	Function struct {
		Name      string `validate:"required,nonempty" json:"name"`
		Arguments string `validate:"max=10000" json:"arguments"`
	} `validate:"required" json:"function"`
}

type ProviderConfig struct {
	Model       string  `validate:"required,nonempty" json:"model"`
	APIKey      string  `validate:"required,min=10" json:"api_key"`
	Temperature float32 `validate:"min=0,max=2" json:"temperature"`
}

type LanguageModelConfig struct {
	MaxIterations int `validate:"min=1,max=100" json:"max_iterations"`
}

func validateRole(fl validator.FieldLevel) bool {
	role := fl.Field().String()
	validRoles := map[string]bool{
		"system":    true,
		"user":      true,
		"assistant": true,
		"AI":        true,
		"tool":      true,
	}
	return validRoles[role]
}

func validateNonEmpty(fl validator.FieldLevel) bool {
	return strings.TrimSpace(fl.Field().String()) != ""
}

func ValidateStruct(s interface{}) error {
	if err := validate.Struct(s); err != nil {
		var validationErrors []string
		for _, err := range err.(validator.ValidationErrors) {
			field := err.Field()
			tag := err.Tag()
			value := err.Value()

			var message string
			switch tag {
			case "required":
				message = fmt.Sprintf("field '%s' is required", field)
			case "min":
				message = fmt.Sprintf("field '%s' must be at least %s characters/value", field, err.Param())
			case "max":
				message = fmt.Sprintf("field '%s' must be at most %s characters/value", field, err.Param())
			case "eq":
				message = fmt.Sprintf("field '%s' must equal '%s'", field, err.Param())
			case "validrole":
				message = fmt.Sprintf("field '%s' has invalid role: %v", field, value)
			case "nonempty":
				message = fmt.Sprintf("field '%s' cannot be empty or whitespace", field)
			default:
				message = fmt.Sprintf("field '%s' failed validation '%s'", field, tag)
			}
			validationErrors = append(validationErrors, message)
		}
		return NewValidationError("struct", strings.Join(validationErrors, "; "))
	}
	return nil
}

func ValidatePrompt(prompt string) error {
	if strings.TrimSpace(prompt) == "" {
		return NewValidationError("prompt", "prompt cannot be empty or whitespace only")
	}

	if len(prompt) > 100000 {
		return NewValidationError("prompt", "prompt exceeds maximum length of 100,000 characters")
	}

	return nil
}

func ValidateMessages(messages []Message) error {
	if len(messages) == 0 {
		return NewValidationError("messages", "at least one message is required")
	}

	for i, msg := range messages {
		validatedMsg := ValidatedMessage{
			Role:       msg.Role,
			Message:    msg.Message,
			ToolCallID: msg.ToolCallID,
			ToolCalls:  msg.ToolCalls,
		}

		if err := ValidateStruct(validatedMsg); err != nil {
			if ge, ok := err.(*GothoughtError); ok {
				return ge.WithContext("message_index", i)
			}
			return err
		}

		if msg.Role == "tool" && strings.TrimSpace(msg.ToolCallID) == "" {
			return NewValidationError("message.tool_call_id", "tool_call_id is required for tool messages").WithContext("message_index", i)
		}

		if strings.TrimSpace(msg.Message) == "" && len(msg.ToolCalls) == 0 {
			return NewValidationError("message.content", "message must have either content or tool calls").WithContext("message_index", i)
		}
	}

	return nil
}

func ValidateProviderConfig(config ProviderConfig) error {
	return ValidateStruct(config)
}

func ValidateLanguageModelConfig(config LanguageModelConfig) error {
	return ValidateStruct(config)
}
