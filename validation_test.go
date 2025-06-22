package gothought

import (
	"testing"
)

func TestValidatePrompt(t *testing.T) {
	tests := []struct {
		name        string
		prompt      string
		shouldError bool
	}{
		{
			name:        "Valid prompt",
			prompt:      "Hello, world!",
			shouldError: false,
		},
		{
			name:        "Empty prompt",
			prompt:      "",
			shouldError: true,
		},
		{
			name:        "Whitespace only prompt",
			prompt:      "   \n\t  ",
			shouldError: true,
		},
		{
			name:        "Very long prompt",
			prompt:      string(make([]byte, 100001)),
			shouldError: true,
		},
		{
			name:        "Max length prompt",
			prompt:      string(make([]byte, 100000)),
			shouldError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidatePrompt(tt.prompt)
			hasError := err != nil

			if hasError != tt.shouldError {
				t.Errorf("Expected error: %v, got error: %v (%v)", tt.shouldError, hasError, err)
			}

			if hasError && !IsValidationError(err) {
				t.Errorf("Expected validation error, got %T", err)
			}
		})
	}
}

func TestValidateMessages(t *testing.T) {
	tests := []struct {
		name        string
		messages    []Message
		shouldError bool
	}{
		{
			name: "Valid messages",
			messages: []Message{
				{Role: "user", Message: "Hello"},
				{Role: "assistant", Message: "Hi there"},
			},
			shouldError: false,
		},
		{
			name:        "Empty messages",
			messages:    []Message{},
			shouldError: true,
		},
		{
			name: "Invalid role",
			messages: []Message{
				{Role: "invalid", Message: "Hello"},
			},
			shouldError: true,
		},
		{
			name: "Tool message without tool_call_id",
			messages: []Message{
				{Role: "tool", Message: "Result"},
			},
			shouldError: true,
		},
		{
			name: "Message without content or tool calls",
			messages: []Message{
				{Role: "user", Message: ""},
			},
			shouldError: true,
		},
		{
			name: "Message too long",
			messages: []Message{
				{Role: "user", Message: string(make([]byte, 50001))},
			},
			shouldError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateMessages(tt.messages)
			hasError := err != nil

			if hasError != tt.shouldError {
				t.Errorf("Expected error: %v, got error: %v (%v)", tt.shouldError, hasError, err)
			}

			if hasError && !IsValidationError(err) {
				t.Errorf("Expected validation error, got %T", err)
			}
		})
	}
}

func TestValidateProviderConfig(t *testing.T) {
	tests := []struct {
		name        string
		config      ProviderConfig
		shouldError bool
	}{
		{
			name: "Valid config",
			config: ProviderConfig{
				Model:       "gpt-4",
				APIKey:      "sk-1234567890",
				Temperature: 0.7,
			},
			shouldError: false,
		},
		{
			name: "Empty model",
			config: ProviderConfig{
				Model:       "",
				APIKey:      "sk-1234567890",
				Temperature: 0.7,
			},
			shouldError: true,
		},
		{
			name: "Short API key",
			config: ProviderConfig{
				Model:       "gpt-4",
				APIKey:      "short",
				Temperature: 0.7,
			},
			shouldError: true,
		},
		{
			name: "Invalid temperature (too low)",
			config: ProviderConfig{
				Model:       "gpt-4",
				APIKey:      "sk-1234567890",
				Temperature: -0.1,
			},
			shouldError: true,
		},
		{
			name: "Invalid temperature (too high)",
			config: ProviderConfig{
				Model:       "gpt-4",
				APIKey:      "sk-1234567890",
				Temperature: 2.1,
			},
			shouldError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateProviderConfig(tt.config)
			hasError := err != nil

			if hasError != tt.shouldError {
				t.Errorf("Expected error: %v, got error: %v (%v)", tt.shouldError, hasError, err)
			}

			if hasError && !IsValidationError(err) {
				t.Errorf("Expected validation error, got %T", err)
			}
		})
	}
}

func TestValidateLanguageModelConfig(t *testing.T) {
	tests := []struct {
		name        string
		config      LanguageModelConfig
		shouldError bool
	}{
		{
			name: "Valid config",
			config: LanguageModelConfig{
				MaxIterations: 10,
			},
			shouldError: false,
		},
		{
			name: "Zero iterations",
			config: LanguageModelConfig{
				MaxIterations: 0,
			},
			shouldError: true,
		},
		{
			name: "Too many iterations",
			config: LanguageModelConfig{
				MaxIterations: 101,
			},
			shouldError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateLanguageModelConfig(tt.config)
			hasError := err != nil

			if hasError != tt.shouldError {
				t.Errorf("Expected error: %v, got error: %v (%v)", tt.shouldError, hasError, err)
			}

			if hasError && !IsValidationError(err) {
				t.Errorf("Expected validation error, got %T", err)
			}
		})
	}
}
