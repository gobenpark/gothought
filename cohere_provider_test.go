package gothought

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/gobenpark/gothought/tool"
	"github.com/stretchr/testify/require"
)

func TestCohereProvider_Integration(t *testing.T) {
	apiKey := os.Getenv("COHERE_API_KEY")
	if apiKey == "" {
		t.Skip("COHERE_API_KEY not set, skipping integration test")
	}

	provider := NewCohereProvider("command", apiKey)

	messages := []Message{
		{
			Role:    "user",
			Message: "Hello! Please respond with just 'Hi there!' for this test.",
		},
	}

	ctx := context.Background()
	result, finishReason, err := provider.Generate(ctx, nil, messages)

	require.NoError(t, err)
	require.NotNil(t, result)
	require.Equal(t, "stop", finishReason)
	require.NotEmpty(t, result.Message)
	require.Equal(t, "assistant", result.Role)

	t.Logf("Cohere response: %s", result.Message)
}

func TestCohereProvider_MessageConversion(t *testing.T) {
	provider := NewCohereProvider("command", "test-key")

	tests := []struct {
		name               string
		input              []Message
		expectedMessage    string
		expectedHistoryLen int
		expectedFirstRole  string
	}{
		{
			name: "single user message",
			input: []Message{
				{Role: "user", Message: "Hello"},
			},
			expectedMessage:    "Hello",
			expectedHistoryLen: 0,
		},
		{
			name: "conversation with history",
			input: []Message{
				{Role: "user", Message: "First message"},
				{Role: "assistant", Message: "First response"},
				{Role: "user", Message: "Second message"},
			},
			expectedMessage:    "Second message",
			expectedHistoryLen: 2,
			expectedFirstRole:  "USER",
		},
		{
			name: "system message handling",
			input: []Message{
				{Role: "system", Message: "You are helpful"},
				{Role: "user", Message: "Hello"},
			},
			expectedMessage:    "You are helpful\n\nHello",
			expectedHistoryLen: 0,
		},
		{
			name: "role mapping",
			input: []Message{
				{Role: "AI", Message: "AI response"},
				{Role: "human", Message: "Human message"},
				{Role: "tool", Message: "Tool result"},
				{Role: "user", Message: "Final message"},
			},
			expectedMessage:    "Final message",
			expectedHistoryLen: 3,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			message, history := provider.convertMessagesToCohere(tt.input)
			require.Equal(t, tt.expectedMessage, message)
			require.Len(t, history, tt.expectedHistoryLen)

			if tt.expectedHistoryLen > 0 && tt.expectedFirstRole != "" {
				require.Equal(t, tt.expectedFirstRole, history[0].Role)
			}
		})
	}
}

func TestCohereProvider_Streaming(t *testing.T) {
	apiKey := os.Getenv("COHERE_API_KEY")
	if apiKey == "" {
		t.Skip("COHERE_API_KEY not set, skipping streaming test")
	}

	provider := NewCohereProvider("command", apiKey)

	messages := []Message{
		{
			Role:    "user",
			Message: "Count from 1 to 3. Respond with just the numbers.",
		},
	}

	var responses []string
	callback := func(message Message) error {
		responses = append(responses, message.Message)
		t.Logf("Streaming chunk: %s", message.Message)
		return nil
	}

	ctx := context.Background()
	err := provider.GenerateStreaming(ctx, nil, messages, callback)

	require.NoError(t, err)
	require.NotEmpty(t, responses)

	// Join all responses to get the complete message
	fullResponse := ""
	for _, chunk := range responses {
		fullResponse += chunk
	}
	require.NotEmpty(t, fullResponse)
	t.Logf("Complete streaming response: %s", fullResponse)
}

func TestCohereProvider_Validation(t *testing.T) {
	t.Run("empty API key", func(t *testing.T) {
		provider := NewCohereProvider("command", "")
		_, _, err := provider.Generate(context.Background(), nil, []Message{
			{Role: "user", Message: "test"},
		})
		require.Error(t, err)
		require.Contains(t, err.Error(), "COHERE_API_KEY cannot be empty")
	})

	t.Run("nil callback for streaming", func(t *testing.T) {
		provider := NewCohereProvider("command", "test-key")
		err := provider.GenerateStreaming(context.Background(), nil, []Message{
			{Role: "user", Message: "test"},
		}, nil)
		require.Error(t, err)
		require.Contains(t, err.Error(), "callback function cannot be nil")
	})

	t.Run("invalid messages", func(t *testing.T) {
		provider := NewCohereProvider("command", "test-key")
		_, _, err := provider.Generate(context.Background(), nil, []Message{
			{Role: "invalid", Message: "test"},
		})
		require.Error(t, err)
	})
}

func TestNewCohereProvider(t *testing.T) {
	t.Run("default model", func(t *testing.T) {
		provider := NewCohereProvider("", "test-key")
		require.Equal(t, "command", provider.model)
		require.Equal(t, "test-key", provider.apiKey)
	})

	t.Run("custom model", func(t *testing.T) {
		provider := NewCohereProvider("command-light", "test-key")
		require.Equal(t, "command-light", provider.model)
		require.Equal(t, "test-key", provider.apiKey)
	})

	t.Run("configuration methods", func(t *testing.T) {
		provider := NewCohereProvider("command", "test-key")

		retryConfig := RetryConfig{MaxAttempts: 5}
		provider = provider.WithRetryConfig(retryConfig)
		require.Equal(t, 5, provider.retryConfig.MaxAttempts)

		timeoutConfig := TimeoutConfig{RequestTimeout: 30000}
		provider = provider.WithTimeoutConfig(timeoutConfig)
		require.Equal(t, time.Duration(30000), provider.timeoutConfig.RequestTimeout)
	})
}

func TestCohereProvider_ToolConversion(t *testing.T) {
	// This would require importing the tool package for testing
	// For now, we'll test the conversion logic with a mock tool
	provider := NewCohereProvider("command", "test-key")

	// Test with empty tools
	tools := make(map[string]tool.Tool)
	cohereTools := provider.convertToolsToCohere(tools)
	require.Empty(t, cohereTools)

	// More comprehensive tool testing would require actual tool implementations
}

func TestCohereProvider_EmptyMessageHandling(t *testing.T) {
	provider := NewCohereProvider("command", "test-key")

	message, history := provider.convertMessagesToCohere([]Message{})
	require.Empty(t, message)
	require.Nil(t, history)
}

func TestCohereProvider_ModelVariants(t *testing.T) {
	testCases := []struct {
		model    string
		expected string
	}{
		{"", "command"},
		{"command", "command"},
		{"command-light", "command-light"},
		{"command-nightly", "command-nightly"},
	}

	for _, tc := range testCases {
		t.Run(tc.model, func(t *testing.T) {
			provider := NewCohereProvider(tc.model, "test-key")
			require.Equal(t, tc.expected, provider.model)
		})
	}
}
