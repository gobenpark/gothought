package models

type OllamaChatRequest struct {
	Model       string          `json:"model"`
	Messages    []OllamaMessage `json:"messages"`
	Temperature float32         `json:"temperature,omitempty"`
	Stream      bool            `json:"stream"`
	Tools       []OllamaTool    `json:"tools,omitempty"`
}

type OllamaMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type OllamaTool struct {
	Type     string                 `json:"type"`
	Function map[string]interface{} `json:"function"`
}

type OllamaResponse struct {
	Model     string        `json:"model"`
	CreatedAt string        `json:"created_at"`
	Message   OllamaMessage `json:"message"`
	Done      bool          `json:"done"`
	Response  string        `json:"response,omitempty"`
}
