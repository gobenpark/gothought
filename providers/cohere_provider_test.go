package providers

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/gobenpark/gothought/messages"
	"github.com/gobenpark/gothought/tools"
	"github.com/stretchr/testify/require"
)

func TestCohereProvider_Integration(t *testing.T) {
	apiKey := os.Getenv("COHERE_API_KEY")
	if apiKey == "" {
		t.Skip("COHERE_API_KEY not set, skipping integration test")
	}

	provider := NewCohereProvider("command", WithAPIKey(apiKey))

	messages := []messages.Message{
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
	provider := NewCohereProvider("command", WithAPIKey("test-key"))

	tests := []struct {
		name               string
		input              []messages.Message
		expectedMessage    string
		expectedHistoryLen int
		expectedFirstRole  string
	}{
		{
			name: "single user message",
			input: []messages.Message{
				{Role: "user", Message: "Hello"},
			},
			expectedMessage:    "Hello",
			expectedHistoryLen: 0,
		},
		{
			name: "conversation with history",
			input: []messages.Message{
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
			input: []messages.Message{
				{Role: "system", Message: "You are helpful"},
				{Role: "user", Message: "Hello"},
			},
			expectedMessage:    "You are helpful\n\nHello",
			expectedHistoryLen: 0,
		},
		{
			name: "role mapping",
			input: []messages.Message{
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

	provider := NewCohereProvider("command", WithAPIKey(apiKey))

	msgs := []messages.Message{
		{
			Role:    "user",
			Message: "Count from 1 to 3. Respond with just the numbers.",
		},
	}

	var responses []string
	callback := func(msg messages.Message) error {
		responses = append(responses, msg.Message)
		t.Logf("Streaming chunk: %s", msg.Message)
		return nil
	}

	ctx := context.Background()
	err := provider.GenerateStreaming(ctx, nil, msgs, callback)

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
		provider := NewCohereProvider("command")
		_, _, err := provider.Generate(context.Background(), nil, []messages.Message{
			{Role: "user", Message: "test"},
		})
		require.Error(t, err)
		require.Contains(t, err.Error(), "COHERE_API_KEY cannot be empty")
	})

	t.Run("nil callback for streaming", func(t *testing.T) {
		provider := NewCohereProvider("command", WithAPIKey("test-key"))
		err := provider.GenerateStreaming(context.Background(), nil, []messages.Message{
			{Role: "user", Message: "test"},
		}, nil)
		require.Error(t, err)
		require.Contains(t, err.Error(), "callback function cannot be nil")
	})

	t.Run("invalid messages", func(t *testing.T) {
		provider := NewCohereProvider("command", WithAPIKey("test-key"))
		_, _, err := provider.Generate(context.Background(), nil, []messages.Message{
			{Role: "invalid", Message: "test"},
		})
		require.Error(t, err)
	})
}

func TestNewCohereProvider(t *testing.T) {
	t.Run("default model", func(t *testing.T) {
		provider := NewCohereProvider("", WithAPIKey("test-key"))
		require.Equal(t, "command", provider.model)
		require.Equal(t, "test-key", provider.apiKey)
	})

	t.Run("custom model", func(t *testing.T) {
		provider := NewCohereProvider("command-light", WithAPIKey("test-key"))
		require.Equal(t, "command-light", provider.model)
		require.Equal(t, "test-key", provider.apiKey)
	})

	t.Run("configuration methods", func(t *testing.T) {
		provider := NewCohereProvider("command", WithAPIKey("test-key"))

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
	provider := NewCohereProvider("command", WithAPIKey("test-key"))

	// Test with empty tools
	tools := make(map[string]tools.Tool)
	cohereTools := provider.convertToolsToCohere(tools)
	require.Empty(t, cohereTools)

	// More comprehensive tool testing would require actual tool implementations
}

func TestCohereProvider_EmptyMessageHandling(t *testing.T) {
	provider := NewCohereProvider("command", WithAPIKey("test-key"))

	message, history := provider.convertMessagesToCohere([]messages.Message{})
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
			provider := NewCohereProvider(tc.model, WithAPIKey("test-key"))
			require.Equal(t, tc.expected, provider.model)
		})
	}
}
