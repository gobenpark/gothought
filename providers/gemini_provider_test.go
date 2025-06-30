package providers

import (
	"context"
	"fmt"
	"os"
	"testing"

	"github.com/gobenpark/gothought/messages"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestGeminiProvider_Generate(t *testing.T) {
	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		t.Skip("GEMINI_API_KEY not set, skipping integration test")
	}

	p := NewGeminiProvider("gemini-2.5-flash", WithAPIKey(apiKey))
	msgs, content, err := p.Generate(context.TODO(), nil, []messages.Message{
		{
			Role:       "user",
			ToolCallID: "",
			Message:    "안녕??",
			ToolCalls:  nil,
		},
	})
	require.NoError(t, err)
	fmt.Println(content)
	fmt.Println(msgs)
}

func TestNewGeminiProvider(t *testing.T) {
	tests := []struct {
		name      string
		apiKey    string
		model     string
		wantKey   string
		wantModel string
	}{
		{
			name:      "기본 모델로 생성",
			apiKey:    "test-key",
			model:     "gemini-pro",
			wantKey:   "test-key",
			wantModel: "gemini-pro",
		},
		{
			name:      "다른 모델로 생성",
			apiKey:    "test-key",
			model:     "gemini-2.5-flash",
			wantKey:   "test-key",
			wantModel: "gemini-2.5-flash",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			provider := NewGeminiProvider(tt.model, WithAPIKey(tt.apiKey))

			assert.NotNil(t, provider)
			assert.Equal(t, tt.wantKey, provider.apiKey)
			assert.Equal(t, tt.wantModel, provider.model)
		})
	}
}
