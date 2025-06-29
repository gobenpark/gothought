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
	"github.com/gobenpark/gothought/tool"
	"github.com/samber/lo"
	"github.com/tidwall/gjson"
)

const (
	claudeAPIURL       = "https://api.anthropic.com/v1/messages"
	claudeAPIVersion   = "2023-06-01"
	defaultMaxTokens   = 4096
	defaultTemperature = 0.7
	httpTimeout        = 30 * time.Second
)

type ClaudeProvider struct {
	model         string
	apiKey        string
	temperature   float32
	maxTokens     int
	httpClient    *http.Client
	retryConfig   RetryConfig
	timeoutConfig TimeoutConfig
}

func NewClaudeProvider(model string, options ...ProviderOption) *ClaudeProvider {
	provider := &ClaudeProvider{
		model:       model,
		apiKey:      os.Getenv("ANTHROPIC_API_KEY"),
		temperature: defaultTemperature,
		maxTokens:   defaultMaxTokens,
		httpClient: &http.Client{
			Timeout: httpTimeout,
		},
		retryConfig:   DefaultRetryConfig(),
		timeoutConfig: DefaultTimeoutConfig(),
	}

	for _, option := range options {
		option(provider)
	}

	return provider
}

func (c *ClaudeProvider) convertMessages(messages []messages.Message) ([]models.ClaudeMessage, string) {
	var claudeMessages []models.ClaudeMessage
	var systemMessage string

	for _, msg := range messages {
		if msg.Role == "system" {
			systemMessage = msg.Message
			continue
		}

		claudeMsg := c.buildClaudeMessage(msg)
		if len(claudeMsg.Content) > 0 {
			claudeMessages = append(claudeMessages, claudeMsg)
		}
	}

	return claudeMessages, systemMessage
}

func (c *ClaudeProvider) buildClaudeMessage(msg messages.Message) models.ClaudeMessage {
	claudeMsg := models.ClaudeMessage{
		Role: msg.Role,
	}

	switch {
	case len(msg.ToolCalls) > 0:
		claudeMsg.Content = c.buildToolCallContent(msg.ToolCalls)
	case msg.ToolCallID != "":
		claudeMsg.Content = []models.ClaudeContent{{
			Type:      "tool_result",
			ToolUseID: msg.ToolCallID,
			Text:      msg.Message,
		}}
	case msg.Message != "":
		claudeMsg.Content = []models.ClaudeContent{{
			Type: "text",
			Text: msg.Message,
		}}
	}

	return claudeMsg
}

func (c *ClaudeProvider) buildToolCallContent(toolCalls []messages.ToolCalls) []models.ClaudeContent {
	content := make([]models.ClaudeContent, 0, len(toolCalls))

	for _, toolCall := range toolCalls {
		var input interface{}
		if toolCall.Function.Arguments != "" {
			if err := json.Unmarshal([]byte(toolCall.Function.Arguments), &input); err != nil {
				// Log error but continue processing
				input = map[string]interface{}{
					"error": fmt.Sprintf("failed to parse arguments: %v", err),
				}
			}
		}

		content = append(content, models.ClaudeContent{
			Type:  "tool_use",
			ID:    toolCall.ID,
			Name:  toolCall.Function.Name,
			Input: input,
		})
	}

	return content
}

func (c *ClaudeProvider) convertTools(tools map[string]tool.Tool) []models.ClaudeTool {
	return lo.MapToSlice(tools, func(key string, value tool.Tool) models.ClaudeTool {
		return models.ClaudeTool{
			Name:        value.Name(),
			Description: value.Description(),
			InputSchema: value.ParameterSchema(),
		}
	})
}

func (c *ClaudeProvider) Generate(ctx context.Context, tools map[string]tool.Tool, msgs []messages.Message) (*messages.Message, string, error) {
	if c.apiKey == "" {
		return nil, "", goErrors.NewValidationError("api_key", "ANTHROPIC_API_KEY cannot be empty")
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

func (c *ClaudeProvider) generateWithoutRetry(ctx context.Context, tools map[string]tool.Tool, msgs []messages.Message) (*messages.Message, string, error) {
	claudeMessages, systemMessage := c.convertMessages(msgs)

	request := models.ClaudeRequest{
		Model:       c.model,
		MaxTokens:   c.maxTokens,
		Messages:    claudeMessages,
		System:      systemMessage,
		Temperature: c.temperature,
		Tools:       c.convertTools(tools),
		Stream:      false,
	}

	resp, err := c.makeAPIRequest(ctx, request)
	if err != nil {
		return nil, "", err
	}
	defer func() {
		if closeErr := resp.Body.Close(); closeErr != nil {
			// Log error but don't fail the request
			// In production, use proper logging
		}
	}()

	if err := c.checkResponseStatus(resp); err != nil {
		return nil, "", err
	}

	return c.parseResponse(resp.Body)
}

func (c *ClaudeProvider) makeAPIRequest(ctx context.Context, request models.ClaudeRequest) (*http.Response, error) {
	reqBody, err := json.Marshal(request)
	if err != nil {
		return nil, goErrors.NewProviderError("failed to marshal request", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, claudeAPIURL, bytes.NewReader(reqBody))
	if err != nil {
		return nil, goErrors.NewNetworkError("failed to create request", err, 0)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", c.apiKey)
	req.Header.Set("anthropic-version", claudeAPIVersion)

	return c.httpClient.Do(req)
}

func (c *ClaudeProvider) checkResponseStatus(resp *http.Response) error {
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

func (c *ClaudeProvider) parseResponse(body io.Reader) (*messages.Message, string, error) {
	buf := &bytes.Buffer{}
	if _, err := io.Copy(buf, body); err != nil {
		return nil, "", goErrors.NewNetworkError("failed to read response body", err, 0)
	}

	re := gjson.ParseBytes(buf.Bytes())
	stopReason := re.Get("stop_reason").String()

	switch stopReason {
	case "end_turn":
		return c.parseTextResponse(re)
	case "tool_use":
		return c.parseToolResponse(re)
	default:
		return nil, "", goErrors.NewParsingError("unexpected stop reason", nil).
			WithContext("stop_reason", stopReason)
	}
}

func (c *ClaudeProvider) parseTextResponse(re gjson.Result) (*messages.Message, string, error) {
	textContent := ""
	for _, content := range re.Get("content").Array() {
		if content.Get("type").String() == "text" {
			textContent += content.Get("text").String()
		}
	}

	return &messages.Message{
		Role:    "assistant",
		Message: textContent,
	}, messages.FinishReasonStop, nil
}

func (c *ClaudeProvider) parseToolResponse(re gjson.Result) (*messages.Message, string, error) {
	assistantMessage := messages.Message{
		Role: "assistant",
	}
	var toolCalls []messages.ToolCalls

	for _, content := range re.Get("content").Array() {
		if content.Get("type").String() == "tool_use" {
			toolCall := messages.ToolCalls{
				ID:   content.Get("id").String(),
				Type: "function",
			}
			toolCall.Function.Name = content.Get("name").String()

			inputJSON, _ := json.Marshal(content.Get("input").Value())
			toolCall.Function.Arguments = string(inputJSON)

			toolCalls = append(toolCalls, toolCall)
		}
	}

	assistantMessage.ToolCalls = toolCalls
	return &assistantMessage, messages.FinishReasonToolCalls, nil
}

func (c *ClaudeProvider) GenerateStreaming(ctx context.Context, tools map[string]tool.Tool, msgs []messages.Message, callback func(message messages.Message) error) error {
	if c.apiKey == "" {
		return goErrors.NewValidationError("api_key", "ANTHROPIC_API_KEY cannot be empty")
	}

	if callback == nil {
		return goErrors.NewValidationError("callback", "callback function cannot be nil")
	}

	if err := ValidateMessages(msgs); err != nil {
		return err
	}

	claudeMessages, systemMessage := c.convertMessages(msgs)

	request := models.ClaudeRequest{
		Model:       c.model,
		MaxTokens:   c.maxTokens,
		Messages:    claudeMessages,
		System:      systemMessage,
		Temperature: c.temperature,
		Tools:       c.convertTools(tools),
		Stream:      true,
	}

	resp, err := c.makeAPIRequest(ctx, request)
	if err != nil {
		return err
	}
	defer func() {
		if closeErr := resp.Body.Close(); closeErr != nil {
			// Log error but don't fail the request
		}
	}()

	if err := c.checkResponseStatus(resp); err != nil {
		return err
	}

	return c.processStreamingResponse(resp.Body, callback)
}

func (c *ClaudeProvider) processStreamingResponse(body io.Reader, callback func(message messages.Message) error) error {
	reader := bufio.NewReader(body)

	for {
		line, err := reader.ReadBytes('\n')
		if err != nil {
			if err == io.EOF {
				break
			}
			return goErrors.NewNetworkError("failed to read streaming response", err, 0)
		}

		if err := c.processStreamingLine(line, callback); err != nil {
			if err == io.EOF {
				return nil
			}
			return err
		}
	}

	return nil
}

func (c *ClaudeProvider) processStreamingLine(line []byte, callback func(message messages.Message) error) error {
	line = bytes.TrimSpace(line)
	if !bytes.HasPrefix(line, []byte("data: ")) {
		return nil
	}

	data := line[6:]
	re := gjson.ParseBytes(data)
	eventType := re.Get("type").String()

	switch eventType {
	case "content_block_delta":
		return c.handleContentBlockDelta(re, callback)
	case "message_stop":
		return io.EOF
	}

	return nil
}

func (c *ClaudeProvider) handleContentBlockDelta(re gjson.Result, callback func(message messages.Message) error) error {
	if re.Get("delta.type").String() == "text_delta" {
		text := re.Get("delta.text").String()
		if text != "" {
			message := messages.Message{
				Message: text,
			}
			return callback(message)
		}
	}
	return nil
}
