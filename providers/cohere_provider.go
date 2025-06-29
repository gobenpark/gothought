package providers

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"

	"github.com/gobenpark/gothought/errors"
	"github.com/gobenpark/gothought/messages"
	"github.com/gobenpark/gothought/tool"
	"github.com/samber/lo"
	"github.com/tidwall/gjson"
)

type CohereProvider struct {
	apiKey        string
	model         string
	retryConfig   RetryConfig
	timeoutConfig TimeoutConfig
	temperature   float32
}

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

const (
	cohereAPIURL = "https://api.cohere.ai/v1/chat"
)

// NewCohereProvider creates a new Cohere provider with the specified model and API key
func NewCohereProvider(model string, options ...ProviderOption) *CohereProvider {
	if model == "" {
		model = "command" // Default Cohere model
	}

	provider := &CohereProvider{
		apiKey:        os.Getenv("COHERE_API_KEY"),
		model:         model,
		temperature:   0.7,
		retryConfig:   DefaultRetryConfig(),
		timeoutConfig: DefaultTimeoutConfig(),
	}

	for _, option := range options {
		option(provider)
	}

	return provider
}

func (c *CohereProvider) WithRetryConfig(config RetryConfig) *CohereProvider {
	c.retryConfig = config
	return c
}

func (c *CohereProvider) WithTimeoutConfig(config TimeoutConfig) *CohereProvider {
	c.timeoutConfig = config
	return c
}

// convertMessagesToCohere converts internal Message format to Cohere's chat history format
func (c *CohereProvider) convertMessagesToCohere(messages []messages.Message) (string, []CohereChatMessage) {
	if len(messages) == 0 {
		return "", nil
	}

	// Last message is the current user message
	lastMessage := messages[len(messages)-1]
	currentMessage := lastMessage.Message

	// Convert previous messages to chat history
	var chatHistory []CohereChatMessage
	for i, msg := range messages[:len(messages)-1] {
		// Skip the last message as it's the current prompt
		if i == len(messages)-1 {
			break
		}

		role := msg.Role
		switch role {
		case "user", "human":
			role = "USER"
		case "assistant", "AI":
			role = "CHATBOT"
		case "system":
			// Cohere doesn't have system role, prepend to first user message
			if len(chatHistory) == 0 {
				currentMessage = msg.Message + "\n\n" + currentMessage
				continue
			}
			role = "USER"
		case "tool":
			// Tool responses are typically from the system
			role = "CHATBOT"
		}

		chatHistory = append(chatHistory, CohereChatMessage{
			Role:    role,
			Message: msg.Message,
		})
	}

	return currentMessage, chatHistory
}

// convertToolsToCohere converts internal tool format to Cohere's tool format
func (c *CohereProvider) convertToolsToCohere(tools map[string]tool.Tool) []CohereTool {
	return lo.MapToSlice(tools, func(key string, value tool.Tool) CohereTool {
		schema := value.ParameterSchema()
		paramDefs := make(map[string]interface{})

		// Convert JSON schema to Cohere parameter definitions
		if properties, ok := schema["properties"].(map[string]interface{}); ok {
			for propName, propDef := range properties {
				if propDefMap, ok := propDef.(map[string]interface{}); ok {
					paramDefs[propName] = map[string]interface{}{
						"type":        propDefMap["type"],
						"description": propDefMap["description"],
						"required":    contains(schema["required"], propName),
					}
				}
			}
		}

		return CohereTool{
			Name:                 value.Name(),
			Description:          value.Description(),
			ParameterDefinitions: paramDefs,
		}
	})
}

// Helper function to check if a slice contains a string
func contains(slice interface{}, item string) bool {
	if arr, ok := slice.([]interface{}); ok {
		for _, v := range arr {
			if str, ok := v.(string); ok && str == item {
				return true
			}
		}
	}
	return false
}

func (c *CohereProvider) Generate(ctx context.Context, tools map[string]tool.Tool, msgs []messages.Message) (*messages.Message, string, error) {
	if c.apiKey == "" {
		return nil, "", errors.NewValidationError("api_key", "COHERE_API_KEY cannot be empty")
	}

	if err := ValidateMessages(msgs); err != nil {
		return nil, "", err
	}

	timeoutCtx, cancel := WithTimeout(ctx, c.timeoutConfig.RequestTimeout)
	defer cancel()

	type ProviderResult struct {
		Message      *messages.Message
		FinishReason string
	}

	result, err := WithRetry(timeoutCtx, c.retryConfig, func(retryCtx context.Context) (ProviderResult, error) {
		message, finishReason, err := c.generateWithoutRetry(retryCtx, tools, msgs)
		if err != nil {
			return ProviderResult{}, err
		}
		return ProviderResult{Message: message, FinishReason: finishReason}, nil
	})

	if err != nil {
		return nil, "", err
	}

	return result.Message, result.FinishReason, nil
}

func (c *CohereProvider) generateWithoutRetry(ctx context.Context, tools map[string]tool.Tool, msgs []messages.Message) (*messages.Message, string, error) {
	currentMessage, chatHistory := c.convertMessagesToCohere(msgs)

	body := CohereChatRequest{
		Message:          currentMessage,
		Model:            c.model,
		ChatHistory:      chatHistory,
		Stream:           false,
		Temperature:      0.3, // Default Cohere temperature
		PromptTruncation: "AUTO",
	}

	if len(tools) > 0 {
		body.Tools = c.convertToolsToCohere(tools)
	}

	jsonBody, err := json.Marshal(body)
	if err != nil {
		return nil, "", errors.NewProviderError("failed to marshal request body", err)
	}

	request, err := http.NewRequestWithContext(ctx, http.MethodPost, cohereAPIURL, bytes.NewReader(jsonBody))
	if err != nil {
		return nil, "", errors.NewNetworkError("failed to create HTTP request", err, 0)
	}

	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Authorization", "Bearer "+c.apiKey)

	client := &http.Client{}
	res, err := client.Do(request)
	if err != nil {
		return nil, "", errors.NewNetworkError("HTTP request failed", err, 0)
	}
	defer res.Body.Close()

	if res.StatusCode == 429 {
		buf := bytes.Buffer{}
		io.Copy(&buf, res.Body)
		return nil, "", errors.NewRateLimitError("rate limit exceeded", 60)
	}

	if res.StatusCode != 200 {
		buf := bytes.Buffer{}
		if _, err := io.Copy(&buf, res.Body); err != nil {
			return nil, "", errors.NewNetworkError("failed to read error response", err, res.StatusCode)
		}
		return nil, "", errors.NewNetworkError("API request failed", fmt.Errorf("status: %d, body: %s", res.StatusCode, buf.String()), res.StatusCode)
	}

	buf := &bytes.Buffer{}
	if _, err := io.Copy(buf, res.Body); err != nil {
		return nil, "", errors.NewNetworkError("failed to read response body", err, res.StatusCode)
	}

	re := gjson.ParseBytes(buf.Bytes())

	if !re.Get("text").Exists() {
		return nil, "", errors.NewParsingError("no text in response", nil)
	}

	content := re.Get("text").String()
	finishReason := re.Get("finish_reason").String()

	if finishReason == "" {
		finishReason = messages.FinishReasonStop
	}

	// Check for tool calls
	if re.Get("tool_calls").Exists() {
		var toolCalls []messages.ToolCalls
		for _, toolCall := range re.Get("tool_calls").Array() {
			name := toolCall.Get("name").String()
			params := toolCall.Get("parameters").String()

			toolCalls = append(toolCalls, messages.ToolCalls{
				ID:   fmt.Sprintf("cohere_%s", name), // Generate ID for Cohere
				Type: "function",
				Function: struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				}{
					Name:      name,
					Arguments: params,
				},
			})
		}

		if len(toolCalls) > 0 {
			return &messages.Message{
				Role:      "assistant",
				Message:   content,
				ToolCalls: toolCalls,
			}, messages.FinishReasonToolCalls, nil
		}
	}

	return &messages.Message{
		Role:    "assistant",
		Message: content,
	}, finishReason, nil
}

func (c *CohereProvider) GenerateStreaming(ctx context.Context, tools map[string]tool.Tool, msgs []messages.Message, callback func(messages.Message) error) error {
	if c.apiKey == "" {
		return errors.NewValidationError("api_key", "COHERE_API_KEY cannot be empty")
	}

	if callback == nil {
		return errors.NewValidationError("callback", "callback function cannot be nil")
	}

	if err := ValidateMessages(msgs); err != nil {
		return err
	}

	currentMessage, chatHistory := c.convertMessagesToCohere(msgs)

	body := CohereChatRequest{
		Message:          currentMessage,
		Model:            c.model,
		ChatHistory:      chatHistory,
		Stream:           true,
		Temperature:      0.3,
		PromptTruncation: "AUTO",
	}

	if len(tools) > 0 {
		body.Tools = c.convertToolsToCohere(tools)
	}

	jsonBody, err := json.Marshal(body)
	if err != nil {
		return errors.NewProviderError("failed to marshal streaming request body", err)
	}

	request, err := http.NewRequestWithContext(ctx, http.MethodPost, cohereAPIURL, bytes.NewReader(jsonBody))
	if err != nil {
		return errors.NewNetworkError("failed to create streaming HTTP request", err, 0)
	}

	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Authorization", "Bearer "+c.apiKey)
	request.Header.Set("Accept", "text/event-stream")

	client := &http.Client{}
	res, err := client.Do(request)
	if err != nil {
		return errors.NewNetworkError("streaming HTTP request failed", err, 0)
	}
	defer res.Body.Close()

	if res.StatusCode == 429 {
		bodyBytes, _ := io.ReadAll(res.Body)
		return errors.NewRateLimitError("streaming rate limit exceeded", 60).WithContext("response_body", string(bodyBytes))
	}

	if res.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(res.Body)
		return errors.NewNetworkError("streaming API request failed", fmt.Errorf("status: %d, body: %s", res.StatusCode, string(bodyBytes)), res.StatusCode)
	}

	reader := bufio.NewReader(res.Body)

	for {
		line, err := reader.ReadBytes('\n')
		if err != nil {
			if err == io.EOF {
				break
			}
			return errors.NewNetworkError("failed to read streaming response", err, 0)
		}

		line = bytes.TrimSpace(line)
		if len(line) == 0 {
			continue
		}

		// Handle Server-Sent Events format
		if bytes.HasPrefix(line, []byte("data: ")) {
			data := line[6:] // Remove "data: " prefix

			if string(data) == "[DONE]" {
				break
			}

			var chunkResponse CohereStreamResponse
			if err := json.Unmarshal(data, &chunkResponse); err != nil {
				return errors.NewParsingError("failed to parse streaming chunk", err).WithContext("chunk_data", string(data))
			}

			// Handle text-generation events
			if chunkResponse.EventType == "text-generation" && chunkResponse.Text != "" {
				message := messages.Message{
					Message: chunkResponse.Text,
				}

				if err := callback(message); err != nil {
					return errors.NewProviderError("streaming callback failed", err)
				}
			}

			// Check if streaming is finished
			if chunkResponse.IsFinished {
				break
			}
		}
	}

	return nil
}
