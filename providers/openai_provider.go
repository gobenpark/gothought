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
	"time"

	goErrors "github.com/gobenpark/gothought/errors"
	"github.com/gobenpark/gothought/messages"
	"github.com/gobenpark/gothought/providers/models"
	"github.com/gobenpark/gothought/tools"
	"github.com/samber/lo"
	"github.com/tidwall/gjson"
)

const (
	openAIAPIURL      = "https://api.openai.com/v1/chat/completions"
	openAIDefaultTemp = 0.7
	openAIDefaultTopP = 1
	openAIFreqPenalty = 0.5
	openAIPresPenalty = 0.5
	openAITimeout     = 30 * time.Second
)

type OpenAIProvider struct {
	model         string
	apiKey        string
	temperature   float32
	httpClient    *http.Client
	retryConfig   RetryConfig
	timeoutConfig TimeoutConfig
}

func NewOpenAIProvider(model string, options ...ProviderOption) *OpenAIProvider {
	provider := &OpenAIProvider{
		model:       model,
		apiKey:      os.Getenv("OPENAI_API_KEY"),
		temperature: openAIDefaultTemp,
		httpClient: &http.Client{
			Timeout: openAITimeout,
		},
		retryConfig:   DefaultRetryConfig(),
		timeoutConfig: DefaultTimeoutConfig(),
	}

	for _, option := range options {
		option(provider)
	}

	return provider
}

func (o *OpenAIProvider) convertMessages(msgs []messages.Message) []models.OpenAIMessage {
	return lo.Map(msgs, func(item messages.Message, index int) models.OpenAIMessage {
		msg := models.OpenAIMessage{
			Role:       item.Role,
			Content:    item.Message,
			ToolCallID: item.ToolCallID,
		}

		// Convert tool calls to OpenAI format
		if len(item.ToolCalls) > 0 {
			msg.ToolCalls = lo.Map(item.ToolCalls, func(tc messages.ToolCalls, _ int) models.OpenAIToolCall {
				return models.OpenAIToolCall{
					ID:   tc.ID,
					Type: tc.Type,
					Function: struct {
						Name      string `json:"name"`
						Arguments string `json:"arguments"`
					}{
						Name:      tc.Function.Name,
						Arguments: tc.Function.Arguments,
					},
				}
			})
		}

		return msg
	})
}

func (o *OpenAIProvider) convertTools(tls map[string]tools.Tool) []models.OpenAITool {
	return lo.MapToSlice(tls, func(key string, value tools.Tool) models.OpenAITool {
		return models.OpenAITool{
			Type: "function",
			Function: struct {
				Name        string      `json:"name"`
				Description string      `json:"description"`
				Parameters  interface{} `json:"parameters"`
			}{
				Name:        value.Name(),
				Description: value.Description(),
				Parameters:  value.ParameterSchema(),
			},
		}
	})
}

func (o *OpenAIProvider) createRequest(tools map[string]tools.Tool, msgs []messages.Message, stream bool) models.OpenAIRequest {
	request := models.OpenAIRequest{
		Model:       o.model,
		Messages:    o.convertMessages(msgs),
		Temperature: o.temperature,
		Stream:      stream,
	}

	if len(tools) > 0 {
		request.Tools = o.convertTools(tools)
	}

	return request
}

func (o *OpenAIProvider) Generate(ctx context.Context, tools map[string]tools.Tool, msgs []messages.Message) (*messages.Message, string, error) {
	if o.apiKey == "" {
		return nil, "", goErrors.NewValidationError("api_key", "OPENAI_API_KEY cannot be empty")
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

func (o *OpenAIProvider) generateWithoutRetry(ctx context.Context, tools map[string]tools.Tool, msgs []messages.Message) (*messages.Message, string, error) {
	request := o.createRequest(tools, msgs, false)

	resp, err := o.makeAPIRequest(ctx, request)
	if err != nil {
		return nil, "", err
	}
	defer func() {
		if closeErr := resp.Body.Close(); closeErr != nil {
			// Log error but don't fail the request
		}
	}()

	if err := o.checkResponseStatus(resp); err != nil {
		return nil, "", err
	}

	return o.parseResponse(resp.Body)
}

func (o *OpenAIProvider) makeAPIRequest(ctx context.Context, request models.OpenAIRequest) (*http.Response, error) {
	reqBody, err := json.Marshal(request)
	if err != nil {
		return nil, goErrors.NewProviderError("failed to marshal request", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, openAIAPIURL, bytes.NewReader(reqBody))
	if err != nil {
		return nil, goErrors.NewNetworkError("failed to create request", err, 0)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+o.apiKey)

	return o.httpClient.Do(req)
}

func (o *OpenAIProvider) checkResponseStatus(resp *http.Response) error {
	if resp.StatusCode == http.StatusOK {
		return nil
	}

	bodyBytes, _ := io.ReadAll(resp.Body)

	switch resp.StatusCode {
	case http.StatusTooManyRequests:
		return goErrors.NewRateLimitError("rate limit exceeded", 60)
	case http.StatusUnauthorized:
		return goErrors.NewValidationError("api_key", "invalid or missing API key")
	default:
		return goErrors.NewNetworkError(
			fmt.Sprintf("API request failed: %s", string(bodyBytes)),
			nil,
			resp.StatusCode,
		)
	}
}

func (o *OpenAIProvider) parseResponse(body io.Reader) (*messages.Message, string, error) {
	buf := &bytes.Buffer{}
	if _, err := io.Copy(buf, body); err != nil {
		return nil, "", goErrors.NewNetworkError("failed to read response body", err, 0)
	}

	re := gjson.ParseBytes(buf.Bytes())

	if !re.Get("choices").Exists() {
		return nil, "", goErrors.NewParsingError("no choices in response", nil)
	}

	for _, choice := range re.Get("choices").Array() {
		finishReason := choice.Get("finish_reason").String()

		switch finishReason {
		case "stop":
			return o.parseTextResponse(choice)
		case "tool_calls":
			return o.parseToolResponse(choice)
		case "length":
			return nil, "", goErrors.NewProviderError("response truncated due to length limit", nil)
		case "content_filter":
			return nil, "", goErrors.NewProviderError("response blocked by content filter", nil)
		default:
			return nil, "", goErrors.NewParsingError("unexpected finish reason", nil).
				WithContext("finish_reason", finishReason)
		}
	}

	return nil, "", goErrors.NewParsingError("unexpected response format", nil)
}

func (o *OpenAIProvider) parseTextResponse(choice gjson.Result) (*messages.Message, string, error) {
	content := choice.Get("message.content").String()
	return &messages.Message{
		Role:    "assistant",
		Message: content,
	}, messages.FinishReasonStop, nil
}

func (o *OpenAIProvider) parseToolResponse(choice gjson.Result) (*messages.Message, string, error) {
	assistantMessage := messages.Message{
		Role: "assistant",
	}

	var toolCalls []messages.ToolCalls
	toolCallsArray := choice.Get("message.tool_calls").Array()

	if len(toolCallsArray) == 0 {
		return nil, "", goErrors.NewParsingError("tool_calls finish reason but no tool calls found", nil)
	}

	for _, toolItem := range toolCallsArray {
		toolCall := messages.ToolCalls{
			ID:   toolItem.Get("id").String(),
			Type: toolItem.Get("type").String(),
		}
		toolCall.Function.Name = toolItem.Get("function.name").String()
		toolCall.Function.Arguments = toolItem.Get("function.arguments").String()

		if toolCall.ID == "" || toolCall.Function.Name == "" {
			return nil, "", goErrors.NewParsingError("invalid tool call format", nil)
		}

		toolCalls = append(toolCalls, toolCall)
	}

	assistantMessage.ToolCalls = toolCalls
	return &assistantMessage, messages.FinishReasonToolCalls, nil
}

func (o *OpenAIProvider) GenerateStreaming(ctx context.Context, tools map[string]tools.Tool, msgs []messages.Message, callback func(message messages.Message) error) error {
	if o.apiKey == "" {
		return goErrors.NewValidationError("api_key", "OPENAI_API_KEY cannot be empty")
	}

	if callback == nil {
		return goErrors.NewValidationError("callback", "callback function cannot be nil")
	}

	if err := ValidateMessages(msgs); err != nil {
		return err
	}

	request := o.createRequest(tools, msgs, true)

	resp, err := o.makeAPIRequest(ctx, request)
	if err != nil {
		return err
	}
	defer func() {
		if closeErr := resp.Body.Close(); closeErr != nil {
			// Log error but don't fail the request
		}
	}()

	if err := o.checkResponseStatus(resp); err != nil {
		return err
	}

	return o.processStreamingResponse(resp.Body, callback)
}

func (o *OpenAIProvider) processStreamingResponse(body io.Reader, callback func(message messages.Message) error) error {
	reader := bufio.NewReader(body)

	for {
		line, err := reader.ReadBytes('\n')
		if err != nil {
			if err == io.EOF {
				break
			}
			return goErrors.NewNetworkError("failed to read streaming response", err, 0)
		}

		if err := o.processStreamingLine(line, callback); err != nil {
			if err == io.EOF {
				return nil
			}
			return err
		}
	}

	return nil
}

func (o *OpenAIProvider) processStreamingLine(line []byte, callback func(message messages.Message) error) error {
	line = bytes.TrimSpace(line)
	if !bytes.HasPrefix(line, []byte("data: ")) {
		return nil
	}

	data := line[6:]
	if string(data) == "[DONE]" {
		return io.EOF
	}

	var streamResp models.OpenAIStreamResponse
	if err := json.Unmarshal(data, &streamResp); err != nil {
		return goErrors.NewParsingError("failed to parse streaming chunk", err).
			WithContext("chunk_data", string(data))
	}

	if len(streamResp.Choices) > 0 && streamResp.Choices[0].Delta.Content != "" {
		message := messages.Message{
			Message: streamResp.Choices[0].Delta.Content,
		}

		if err := callback(message); err != nil {
			return goErrors.NewProviderError("streaming callback failed", err)
		}
	}

	return nil
}
