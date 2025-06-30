package providers

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/gobenpark/gothought/errors"
	"github.com/gobenpark/gothought/messages"
	"github.com/gobenpark/gothought/providers/models"
	"github.com/gobenpark/gothought/tools"
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

// NewOllamaProvider creates a new Ollama provider with the specified model and optional custom base URL
func NewOllamaProvider(model string, options ...ProviderOption) *OllamaProvider {
	provider := &OllamaProvider{
		baseURL:       "http://localhost:11434",
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

func (o *OllamaProvider) WithRetryConfig(config RetryConfig) *OllamaProvider {
	o.retryConfig = config
	return o
}

func (o *OllamaProvider) WithTimeoutConfig(config TimeoutConfig) *OllamaProvider {
	o.timeoutConfig = config
	return o
}

// convertMessagesToOllama converts internal Message format to Ollama's message format
func (o *OllamaProvider) convertMessagesToOllama(msgs []messages.Message) []models.OllamaMessage {
	return lo.Map(msgs, func(msg messages.Message, index int) models.OllamaMessage {
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

		return models.OllamaMessage{
			Role:    role,
			Content: msg.Message,
		}
	})
}

// convertToolsToOllama converts internal tool format to Ollama's tool format
func (o *OllamaProvider) convertToolsToOllama(tls map[string]tools.Tool) []models.OllamaTool {
	return lo.MapToSlice(tls, func(key string, value tools.Tool) models.OllamaTool {
		return models.OllamaTool{
			Type: "function",
			Function: map[string]interface{}{
				"name":        value.Name(),
				"description": value.Description(),
				"parameters":  value.ParameterSchema(),
			},
		}
	})
}

func (o *OllamaProvider) Generate(ctx context.Context, tools map[string]tools.Tool, msgs []messages.Message) (*messages.Message, string, error) {
	if o.model == "" {
		return nil, "", errors.NewValidationError("model", "model cannot be empty")
	}

	if err := ValidateMessages(msgs); err != nil {
		return nil, "", err
	}

	timeoutCtx, cancel := WithTimeout(ctx, o.timeoutConfig.RequestTimeout)
	defer cancel()

	type ProviderResult struct {
		Message      *messages.Message
		FinishReason string
	}

	result, err := WithRetry(timeoutCtx, o.retryConfig, func(retryCtx context.Context) (ProviderResult, error) {
		message, finishReason, err := o.generateWithoutRetry(retryCtx, tools, msgs)
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

func (o *OllamaProvider) generateWithoutRetry(ctx context.Context, tools map[string]tools.Tool, msgs []messages.Message) (*messages.Message, string, error) {
	body := models.OllamaChatRequest{
		Model:       o.model,
		Messages:    o.convertMessagesToOllama(msgs),
		Temperature: o.temperature,
		Stream:      false,
	}

	if len(tools) > 0 {
		body.Tools = o.convertToolsToOllama(tools)
	}

	jsonBody, err := json.Marshal(body)
	if err != nil {
		return nil, "", errors.NewProviderError("failed to marshal request body", err)
	}

	url := fmt.Sprintf("%s/api/chat", o.baseURL)
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(jsonBody))
	if err != nil {
		return nil, "", errors.NewNetworkError("failed to create HTTP request", err, 0)
	}

	request.Header.Set("Content-Type", "application/json")

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

	if !re.Get("message").Exists() {
		return nil, "", errors.NewParsingError("no message in response", nil)
	}

	content := re.Get("message.content").String()
	done := re.Get("done").Bool()

	if !done {
		return nil, "", errors.NewParsingError("response not complete", nil)
	}

	return &messages.Message{
		Role:    "assistant",
		Message: content,
	}, messages.FinishReasonStop, nil
}

func (o *OllamaProvider) GenerateStreaming(ctx context.Context, tools map[string]tools.Tool, msgs []messages.Message, callback func(messages.Message) error) error {
	if o.model == "" {
		return errors.NewValidationError("model", "model cannot be empty")
	}

	if callback == nil {
		return errors.NewValidationError("callback", "callback function cannot be nil")
	}

	if err := ValidateMessages(msgs); err != nil {
		return err
	}

	body := models.OllamaChatRequest{
		Model:       o.model,
		Messages:    o.convertMessagesToOllama(msgs),
		Temperature: o.temperature,
		Stream:      true,
	}

	if len(tools) > 0 {
		body.Tools = o.convertToolsToOllama(tools)
	}

	jsonBody, err := json.Marshal(body)
	if err != nil {
		return errors.NewProviderError("failed to marshal streaming request body", err)
	}

	url := fmt.Sprintf("%s/api/chat", o.baseURL)
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(jsonBody))
	if err != nil {
		return errors.NewNetworkError("failed to create streaming HTTP request", err, 0)
	}

	request.Header.Set("Content-Type", "application/json")

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

		var chunkResponse models.OllamaResponse
		if err := json.Unmarshal(line, &chunkResponse); err != nil {
			return errors.NewParsingError("failed to parse streaming chunk", err).WithContext("chunk_data", string(line))
		}

		// Handle streaming response content
		if chunkResponse.Message.Content != "" {
			message := messages.Message{
				Message: chunkResponse.Message.Content,
			}

			if err := callback(message); err != nil {
				return errors.NewProviderError("streaming callback failed", err)
			}
		}

		// Check if response is complete
		if chunkResponse.Done {
			break
		}
	}

	return nil
}
