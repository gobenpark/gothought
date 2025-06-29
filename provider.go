//go:generate mockgen -source=./provider.go -destination=./mock_provider.go -package=gothought

package gothought

import (
	"context"

	"github.com/gobenpark/gothought/messages"
	"github.com/gobenpark/gothought/tools"
)

type Provider interface {
	Generate(ctx context.Context, tools map[string]tools.Tool, msgs []messages.Message) (*messages.Message, string, error)
}

type StreamingCapable interface {
	GenerateStreaming(ctx context.Context, tools map[string]tools.Tool, msgs []messages.Message, callback func(messages.Message) error) error
}
