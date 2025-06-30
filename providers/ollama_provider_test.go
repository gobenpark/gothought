package providers

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/gobenpark/gothought/messages"
	"github.com/gobenpark/gothought/providers/models"
	"github.com/stretchr/testify/require"
)

func TestOllamaProvider_Integration(t *testing.T) {
	// Check if Ollama is available for testing
	if os.Getenv("OLLAMA_TEST") == "" {
		t.Skip("OLLAMA_TEST not set, skipping integration test. Set OLLAMA_TEST=1 to run with local Ollama.")
	}

	// Test with default localhost:11434
	provider := NewOllamaProvider("gemma3", WithTemperature(0.7))

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

	t.Logf("Ollama response: %s", result.Message)
}

func TestOllamaProvider_CustomBaseURL(t *testing.T) {
	provider := NewOllamaProvider("gemma3", WithTemperature(0.7), WithOllamaURL("http://custom-ollama:11434"))
	require.Equal(t, "http://custom-ollama:11434", provider.baseURL)

	// Test with trailing slash
	provider2 := NewOllamaProvider("gemma3", WithTemperature(0.7), WithOllamaURL("http://custom-ollama:11434/"))
	require.Equal(t, "http://custom-ollama:11434", provider2.baseURL)
}

func TestOllamaProvider_MessageConversion(t *testing.T) {
	provider := NewOllamaProvider("gemma3", WithTemperature(0.7))

	tests := []struct {
		name     string
		input    []messages.Message
		expected []models.OllamaMessage
	}{
		{
			name: "basic user message",
			input: []messages.Message{
				{Role: "user", Message: "Hello"},
			},
			expected: []models.OllamaMessage{
				{Role: "user", Content: "Hello"},
			},
		},
		{
			name: "system and user messages",
			input: []messages.Message{
				{Role: "system", Message: "You are a helpful assistant"},
				{Role: "user", Message: "Hello"},
			},
			expected: []models.OllamaMessage{
				{Role: "system", Content: "You are a helpful assistant"},
				{Role: "user", Content: "Hello"},
			},
		},
		{
			name: "role mapping",
			input: []messages.Message{
				{Role: "AI", Message: "AI response"},
				{Role: "assistant", Message: "Assistant response"},
				{Role: "human", Message: "Human message"},
				{Role: "tool", Message: "Tool result"},
			},
			expected: []models.OllamaMessage{
				{Role: "assistant", Content: "AI response"},
				{Role: "assistant", Content: "Assistant response"},
				{Role: "user", Content: "Human message"},
				{Role: "user", Content: "Tool result"},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := provider.convertMessagesToOllama(tt.input)
			require.Equal(t, tt.expected, result)
		})
	}
}

func TestOllamaProvider_Streaming(t *testing.T) {
	if os.Getenv("OLLAMA_TEST") == "" {
		t.Skip("OLLAMA_TEST not set, skipping streaming test")
	}

	provider := NewOllamaProvider("gemma3", WithTemperature(0.7))

	msgs := []messages.Message{
		{
			Role:    "user",
			Message: "Count from 1 to 3. Respond with just the numbers.",
		},
	}

	var responses []string
	callback := func(message messages.Message) error {
		responses = append(responses, message.Message)
		t.Logf("Streaming chunk: %s", message.Message)
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

func TestOllamaProvider_Validation(t *testing.T) {
	t.Run("empty model", func(t *testing.T) {
		provider := NewOllamaProvider("")
		_, _, err := provider.Generate(context.Background(), nil, []messages.Message{
			{Role: "user", Message: "test"},
		})
		require.Error(t, err)
		require.Contains(t, err.Error(), "model cannot be empty")
	})

	t.Run("nil callback for streaming", func(t *testing.T) {
		provider := NewOllamaProvider("gemma3", WithTemperature(0.7))
		err := provider.GenerateStreaming(context.Background(), nil, []messages.Message{
			{Role: "user", Message: "test"},
		}, nil)
		require.Error(t, err)
		require.Contains(t, err.Error(), "callback function cannot be nil")
	})

	t.Run("invalid messages", func(t *testing.T) {
		provider := NewOllamaProvider("gemma3", WithTemperature(0.7))
		_, _, err := provider.Generate(context.Background(), nil, []messages.Message{
			{Role: "invalid", Message: "test"},
		})
		require.Error(t, err)
	})
}

func TestNewOllamaProvider(t *testing.T) {
	t.Run("default URL", func(t *testing.T) {
		provider := NewOllamaProvider("gemma3", WithTemperature(0.7))
		require.Equal(t, "http://localhost:11434", provider.baseURL)
		require.Equal(t, "gemma3", provider.model)
		require.Equal(t, float32(0.7), provider.temperature)
	})

	t.Run("custom URL", func(t *testing.T) {
		provider := NewOllamaProvider("mistral", WithTemperature(0.5), WithOllamaURL("http://remote-ollama:8080"))
		require.Equal(t, "http://remote-ollama:8080", provider.baseURL)
		require.Equal(t, "mistral", provider.model)
		require.Equal(t, float32(0.5), provider.temperature)
	})

	t.Run("configuration methods", func(t *testing.T) {
		provider := NewOllamaProvider("gemma3", WithTemperature(0.7))

		retryConfig := RetryConfig{MaxAttempts: 5}
		provider = provider.WithRetryConfig(retryConfig)
		require.Equal(t, 5, provider.retryConfig.MaxAttempts)

		timeoutConfig := TimeoutConfig{RequestTimeout: 30000}
		provider = provider.WithTimeoutConfig(timeoutConfig)
		require.Equal(t, time.Duration(30000), provider.timeoutConfig.RequestTimeout)
	})
}
