package gothought

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/gobenpark/gothought/tool"
	"github.com/samber/lo"
	"github.com/tidwall/gjson"
)

type OllamaProvider struct {
	baseURL       string
	model         string
	temperature   float32
	retryConfig   RetryConfig
	timeoutConfig TimeoutConfig
}

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

var _ Provider = (*OllamaProvider)(nil)
var _ StreamingCapable = (*OllamaProvider)(nil)

// NewOllamaProvider creates a new Ollama provider with the specified model and optional custom base URL
func NewOllamaProvider(model string, temperature float32, baseURL ...string) *OllamaProvider {
	url := "http://localhost:11434"
	if len(baseURL) > 0 && baseURL[0] != "" {
		url = strings.TrimSuffix(baseURL[0], "/")
	}

	return &OllamaProvider{
		baseURL:       url,
		model:         model,
		temperature:   temperature,
		retryConfig:   DefaultRetryConfig(),
		timeoutConfig: DefaultTimeoutConfig(),
	}
}

func (o *OllamaProvider) WithRetryConfig(config RetryConfig) *OllamaProvider {
	o.retryConfig = config
	return o
}

func (o *OllamaProvider) WithTimeoutConfig(config TimeoutConfig) *OllamaProvider {
	o.timeoutConfig = config
	return o
}

// convertMessagesToOllama converts internal Message format to Ollama's message format
func (o *OllamaProvider) convertMessagesToOllama(messages []Message) []OllamaMessage {
	return lo.Map(messages, func(msg Message, index int) OllamaMessage {
		role := msg.Role
		// Map internal roles to Ollama roles
		switch role {
		case "AI", "assistant":
			role = "assistant"
		case "user", "human":
			role = "user"
		case "system":
			role = "system"
		case "tool":
			// Ollama doesn't have tool role, convert to user message
			role = "user"
		}

		return OllamaMessage{
			Role:    role,
			Content: msg.Message,
		}
	})
}

// convertToolsToOllama converts internal tool format to Ollama's tool format
func (o *OllamaProvider) convertToolsToOllama(tools map[string]tool.Tool) []OllamaTool {
	return lo.MapToSlice(tools, func(key string, value tool.Tool) OllamaTool {
		return OllamaTool{
			Type: "function",
			Function: map[string]interface{}{
				"name":        value.Name(),
				"description": value.Description(),
				"parameters":  value.ParameterSchema(),
			},
		}
	})
}

func (o *OllamaProvider) Generate(ctx context.Context, tools map[string]tool.Tool, messages []Message) (*Message, string, error) {
	if o.model == "" {
		return nil, "", NewValidationError("model", "model cannot be empty")
	}

	if err := ValidateMessages(messages); err != nil {
		return nil, "", err
	}

	timeoutCtx, cancel := WithTimeout(ctx, o.timeoutConfig.RequestTimeout)
	defer cancel()

	type ProviderResult struct {
		Message      *Message
		FinishReason string
	}

	result, err := WithRetry(timeoutCtx, o.retryConfig, func(retryCtx context.Context) (ProviderResult, error) {
		message, finishReason, err := o.generateWithoutRetry(retryCtx, tools, messages)
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

func (o *OllamaProvider) generateWithoutRetry(ctx context.Context, tools map[string]tool.Tool, messages []Message) (*Message, string, error) {
	body := OllamaChatRequest{
		Model:       o.model,
		Messages:    o.convertMessagesToOllama(messages),
		Temperature: o.temperature,
		Stream:      false,
	}

	if len(tools) > 0 {
		body.Tools = o.convertToolsToOllama(tools)
	}

	jsonBody, err := json.Marshal(body)
	if err != nil {
		return nil, "", NewProviderError("failed to marshal request body", err)
	}

	url := fmt.Sprintf("%s/api/chat", o.baseURL)
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(jsonBody))
	if err != nil {
		return nil, "", NewNetworkError("failed to create HTTP request", err, 0)
	}

	request.Header.Set("Content-Type", "application/json")

	client := &http.Client{}
	res, err := client.Do(request)
	if err != nil {
		return nil, "", NewNetworkError("HTTP request failed", err, 0)
	}
	defer res.Body.Close()

	if res.StatusCode == 429 {
		buf := bytes.Buffer{}
		io.Copy(&buf, res.Body)
		return nil, "", NewRateLimitError("rate limit exceeded", 60)
	}

	if res.StatusCode != 200 {
		buf := bytes.Buffer{}
		if _, err := io.Copy(&buf, res.Body); err != nil {
			return nil, "", NewNetworkError("failed to read error response", err, res.StatusCode)
		}
		return nil, "", NewNetworkError("API request failed", fmt.Errorf("status: %d, body: %s", res.StatusCode, buf.String()), res.StatusCode)
	}

	buf := &bytes.Buffer{}
	if _, err := io.Copy(buf, res.Body); err != nil {
		return nil, "", NewNetworkError("failed to read response body", err, res.StatusCode)
	}

	re := gjson.ParseBytes(buf.Bytes())

	if !re.Get("message").Exists() {
		return nil, "", NewParsingError("no message in response", nil)
	}

	content := re.Get("message.content").String()
	done := re.Get("done").Bool()

	if !done {
		return nil, "", NewParsingError("response not complete", nil)
	}

	return &Message{
		Role:    "assistant",
		Message: content,
	}, FinishReasonStop, nil
}

func (o *OllamaProvider) GenerateStreaming(ctx context.Context, tools map[string]tool.Tool, messages []Message, callback func(Message) error) error {
	if o.model == "" {
		return NewValidationError("model", "model cannot be empty")
	}

	if callback == nil {
		return NewValidationError("callback", "callback function cannot be nil")
	}

	if err := ValidateMessages(messages); err != nil {
		return err
	}

	body := OllamaChatRequest{
		Model:       o.model,
		Messages:    o.convertMessagesToOllama(messages),
		Temperature: o.temperature,
		Stream:      true,
	}

	if len(tools) > 0 {
		body.Tools = o.convertToolsToOllama(tools)
	}

	jsonBody, err := json.Marshal(body)
	if err != nil {
		return NewProviderError("failed to marshal streaming request body", err)
	}

	url := fmt.Sprintf("%s/api/chat", o.baseURL)
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(jsonBody))
	if err != nil {
		return NewNetworkError("failed to create streaming HTTP request", err, 0)
	}

	request.Header.Set("Content-Type", "application/json")

	client := &http.Client{}
	res, err := client.Do(request)
	if err != nil {
		return NewNetworkError("streaming HTTP request failed", err, 0)
	}
	defer res.Body.Close()

	if res.StatusCode == 429 {
		bodyBytes, _ := io.ReadAll(res.Body)
		return NewRateLimitError("streaming rate limit exceeded", 60).WithContext("response_body", string(bodyBytes))
	}

	if res.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(res.Body)
		return NewNetworkError("streaming API request failed", fmt.Errorf("status: %d, body: %s", res.StatusCode, string(bodyBytes)), res.StatusCode)
	}

	reader := bufio.NewReader(res.Body)

	for {
		line, err := reader.ReadBytes('\n')
		if err != nil {
			if err == io.EOF {
				break
			}
			return NewNetworkError("failed to read streaming response", err, 0)
		}

		line = bytes.TrimSpace(line)
		if len(line) == 0 {
			continue
		}

		var chunkResponse OllamaResponse
		if err := json.Unmarshal(line, &chunkResponse); err != nil {
			return NewParsingError("failed to parse streaming chunk", err).WithContext("chunk_data", string(line))
		}

		// Handle streaming response content
		if chunkResponse.Message.Content != "" {
			message := Message{
				Message: chunkResponse.Message.Content,
			}

			if err := callback(message); err != nil {
				return NewProviderError("streaming callback failed", err)
			}
		}

		// Check if response is complete
		if chunkResponse.Done {
			break
		}
	}

	return nil
}
