package models

type CohereChatRequest struct {
	Message          string              `json:"message"`
	Model            string              `json:"model,omitempty"`
	ChatHistory      []CohereChatMessage `json:"chat_history,omitempty"`
	ConversationID   string              `json:"conversation_id,omitempty"`
	Stream           bool                `json:"stream,omitempty"`
	Temperature      float32             `json:"temperature,omitempty"`
	MaxTokens        int                 `json:"max_tokens,omitempty"`
	PromptTruncation string              `json:"prompt_truncation,omitempty"`
	Tools            []CohereTool        `json:"tools,omitempty"`
}

type CohereChatMessage struct {
	Role    string `json:"role"`
	Message string `json:"message"`
}

type CohereTool struct {
	Name                 string                 `json:"name"`
	Description          string                 `json:"description"`
	ParameterDefinitions map[string]interface{} `json:"parameter_definitions"`
}

type CohereChatResponse struct {
	Text           string           `json:"text"`
	GenerationID   string           `json:"generation_id"`
	ConversationID string           `json:"conversation_id"`
	Citations      []interface{}    `json:"citations"`
	ToolCalls      []CohereToolCall `json:"tool_calls,omitempty"`
	FinishReason   string           `json:"finish_reason,omitempty"`
}

type CohereToolCall struct {
	Name       string                 `json:"name"`
	Parameters map[string]interface{} `json:"parameters"`
}

type CohereStreamResponse struct {
	EventType      string `json:"event_type"`
	Text           string `json:"text,omitempty"`
	IsFinished     bool   `json:"is_finished,omitempty"`
	FinishReason   string `json:"finish_reason,omitempty"`
	GenerationID   string `json:"generation_id,omitempty"`
	ConversationID string `json:"conversation_id,omitempty"`
}
