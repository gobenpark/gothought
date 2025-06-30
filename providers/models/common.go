package models

// Common interfaces and types that can be shared across providers

// UsageInfo represents token usage information
type UsageInfo interface {
	GetInputTokens() int
	GetOutputTokens() int
	GetTotalTokens() int
}

// StreamEvent represents a streaming event type
type StreamEventType string

const (
	StreamEventMessage      StreamEventType = "message"
	StreamEventMessageDelta StreamEventType = "message_delta"
	StreamEventMessageStop  StreamEventType = "message_stop"
	StreamEventError        StreamEventType = "error"
)

// CommonStreamEvent provides a base structure for streaming events
type CommonStreamEvent struct {
	Type    StreamEventType `json:"type"`
	Content string          `json:"content,omitempty"`
	Error   string          `json:"error,omitempty"`
}

// ModelCapability represents what a model can do
type ModelCapability struct {
	MaxTokens      int
	SupportsTools  bool
	SupportsVision bool
	SupportsStream bool
	TokenWindow    int
}

// ModelInfo contains information about a specific model
type ModelInfo struct {
	Provider     string
	ModelID      string
	DisplayName  string
	Capabilities ModelCapability
}
