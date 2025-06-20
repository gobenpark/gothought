package gothought

import (
	"context"
	"os"
	"testing"

	"github.com/gobenpark/gothought/tool"
	"github.com/stretchr/testify/assert"
)

func TestClaudeProvider_Integration(t *testing.T) {
	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" {
		t.Skip("ANTHROPIC_API_KEY not set, skipping integration test")
	}

	provider := NewClaudeProvider("claude-3-haiku-20240307", apiKey, 0.7, 1024)

	t.Run("Simple Text Generation", func(t *testing.T) {
		messages := []Message{
			{Role: "user", Message: "Say hello in a friendly way"},
		}

		result, finishReason, err := provider.Generate(context.Background(), nil, messages)

		assert.NoError(t, err)
		assert.Equal(t, FinishReasonStop, finishReason)
		assert.NotNil(t, result)
		assert.NotEmpty(t, result.Message)
	})

	t.Run("Tool Use Integration", func(t *testing.T) {
		braveAPIKey := os.Getenv("BRAVE_API_KEY")
		if braveAPIKey == "" {
			t.Skip("BRAVE_API_KEY not set, skipping tool use test")
		}

		tools := map[string]tool.Tool{
			"brave_web_search": tool.NewBraveSearchTool(braveAPIKey),
		}

		messages := []Message{
			{Role: "user", Message: "Search for the current weather in Seoul"},
		}

		result, finishReason, err := provider.Generate(context.Background(), tools, messages)

		assert.NoError(t, err)
		assert.Equal(t, FinishReasonToolCalls, finishReason)
		assert.NotNil(t, result)
		assert.NotEmpty(t, result.ToolCalls)
		assert.Equal(t, "brave_web_search", result.ToolCalls[0].Function.Name)
	})
}

func TestClaudeProvider_MessageConversion(t *testing.T) {
	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" {
		t.Skip("ANTHROPIC_API_KEY not set, skipping integration test")
	}
	provider := NewClaudeProvider("claude-3-haiku-20240307", apiKey, 0.7, 1024)

	t.Run("Convert Simple Messages", func(t *testing.T) {
		messages := []Message{
			{Role: "system", Message: "You are a helpful assistant"},
			{Role: "user", Message: "Hello"},
			{Role: "assistant", Message: "Hi there!"},
		}

		claudeMessages, systemMessage := provider.convertMessages(messages)

		assert.Equal(t, "You are a helpful assistant", systemMessage)
		assert.Len(t, claudeMessages, 2)
		assert.Equal(t, "user", claudeMessages[0].Role)
		assert.Equal(t, "Hello", claudeMessages[0].Content[0].Text)
		assert.Equal(t, "assistant", claudeMessages[1].Role)
		assert.Equal(t, "Hi there!", claudeMessages[1].Content[0].Text)
	})

	t.Run("Convert Tool Call Messages", func(t *testing.T) {
		messages := []Message{
			{
				Role: "assistant",
				ToolCalls: []ToolCalls{
					{
						ID:   "call_123",
						Type: "function",
						Function: struct {
							Name      string `json:"name"`
							Arguments string `json:"arguments"`
						}{
							Name:      "search",
							Arguments: `{"query": "test"}`,
						},
					},
				},
			},
			{
				Role:       "tool",
				ToolCallID: "call_123",
				Message:    "Search result",
			},
		}

		claudeMessages, _ := provider.convertMessages(messages)

		assert.Len(t, claudeMessages, 2)
		assert.Equal(t, "assistant", claudeMessages[0].Role)
		assert.Equal(t, "tool_use", claudeMessages[0].Content[0].Type)
		assert.Equal(t, "call_123", claudeMessages[0].Content[0].ID)
		assert.Equal(t, "search", claudeMessages[0].Content[0].Name)

		assert.Equal(t, "tool", claudeMessages[1].Role)
		assert.Equal(t, "tool_result", claudeMessages[1].Content[0].Type)
		assert.Equal(t, "call_123", claudeMessages[1].Content[0].ToolUseID)
		assert.Equal(t, "Search result", claudeMessages[1].Content[0].Text)
	})
}

func TestClaudeProvider_Streaming(t *testing.T) {
	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" {
		t.Skip("ANTHROPIC_API_KEY not set, skipping streaming test")
	}

	provider := NewClaudeProvider("claude-3-haiku-20240307", apiKey, 0.7, 1024)

	t.Run("Streaming Response", func(t *testing.T) {
		messages := []Message{
			{Role: "user", Message: "Count from 1 to 5"},
		}

		var receivedMessages []string
		callback := func(message Message) error {
			receivedMessages = append(receivedMessages, message.Message)
			return nil
		}

		err := provider.GenerateStreaming(context.Background(), nil, messages, callback)

		assert.NoError(t, err)
		assert.NotEmpty(t, receivedMessages)
	})
}
