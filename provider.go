//go:generate mockgen -source=./provider.go -destination=./mock_provider.go -package=gothought

package gothought

import (
	"context"

	"github.com/gobenpark/gothought/messages"
	"github.com/gobenpark/gothought/tool"
)

type Provider interface {
	Generate(ctx context.Context, tools map[string]tool.Tool, msgs []messages.Message) (*messages.Message, string, error)
}

type StreamingCapable interface {
	GenerateStreaming(ctx context.Context, tools map[string]tool.Tool, msgs []messages.Message, callback func(messages.Message) error) error
}
