package gothought

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"

	"github.com/gobenpark/gothought/tool"
	"github.com/samber/lo"
	"github.com/tidwall/gjson"
)

type OpenAIBody struct {
	Model            string                   `json:"model"`
	Messages         []OpenAIMessage          `json:"messages"`
	Temperature      float32                  `json:"temperature,omitempty"`
	TopP             int                      `json:"top_p,omitempty"`
	FrequencyPenalty float32                  `json:"frequency_penalty,omitempty"`
	PresencePenalty  float32                  `json:"presence_penalty,omitempty"`
	Stream           bool                     `json:"stream"`
	Tools            []map[string]interface{} `json:"tools"`
	ToolChoice       string                   `json:"tool_choice,omitempty"`
}

type OpenAIMessage struct {
	Role       string      `json:"role"`
	Content    string      `json:"content"`
	ToolCallID string      `json:"tool_call_id"`
	ToolCalls  []ToolCalls `json:"tool_calls"`
}

type ToolCalls struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

type OpenAIResponse struct {
	Id      string `json:"id"`
	Object  string `json:"object"`
	Created int    `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Index   int `json:"index"`
		Message struct {
			Role        string        `json:"role"`
			Content     string        `json:"content"`
			Refusal     interface{}   `json:"refusal"`
			Annotations []interface{} `json:"annotations"`
		} `json:"message"`
		Logprobs     interface{} `json:"logprobs"`
		FinishReason string      `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		PromptTokens        int `json:"prompt_tokens"`
		CompletionTokens    int `json:"completion_tokens"`
		TotalTokens         int `json:"total_tokens"`
		PromptTokensDetails struct {
			CachedTokens int `json:"cached_tokens"`
			AudioTokens  int `json:"audio_tokens"`
		} `json:"prompt_tokens_details"`
		CompletionTokensDetails struct {
			ReasoningTokens          int `json:"reasoning_tokens"`
			AudioTokens              int `json:"audio_tokens"`
			AcceptedPredictionTokens int `json:"accepted_prediction_tokens"`
			RejectedPredictionTokens int `json:"rejected_prediction_tokens"`
		} `json:"completion_tokens_details"`
	} `json:"usage"`
	ServiceTier       string `json:"service_tier"`
	SystemFingerprint string `json:"system_fingerprint"`
}

type OpenAIProvider struct {
	model         string
	apiKey        string
	temperature   float32
	retryConfig   RetryConfig
	timeoutConfig TimeoutConfig
}

var _ Provider = (*OpenAIProvider)(nil)

func NewOpenAIProvider(model string, options ...ProviderOption) *OpenAIProvider {
	provider := &OpenAIProvider{
		model:         model,
		apiKey:        os.Getenv("OPENAI_API_KEY"),
		temperature:   0.7,
		retryConfig:   DefaultRetryConfig(),
		timeoutConfig: DefaultTimeoutConfig(),
	}

	for _, option := range options {
		option(provider)
	}

	return provider
}

func (o *OpenAIProvider) WithRetryConfig(config RetryConfig) *OpenAIProvider {
	o.retryConfig = config
	return o
}

func (o *OpenAIProvider) WithTimeoutConfig(config TimeoutConfig) *OpenAIProvider {
	o.timeoutConfig = config
	return o
}

func (o *OpenAIProvider) generateBody(tools map[string]tool.Tool, messages []Message, stream bool) OpenAIBody {
	body := OpenAIBody{
		Model: o.model,
		Messages: lo.Map(messages, func(item Message, index int) OpenAIMessage {
			msg := OpenAIMessage{
				Role:       item.Role,
				Content:    item.Message,
				ToolCallID: item.ToolCallID,
				ToolCalls:  item.ToolCalls,
			}

			return msg
		}),
		Temperature:      o.temperature,
		TopP:             1,
		FrequencyPenalty: 0.5,
		PresencePenalty:  0.5,
		Stream:           stream,
	}

	body.Tools = lo.MapToSlice(tools, func(key string, value tool.Tool) map[string]interface{} {
		return map[string]interface{}{
			"type": "function",
			"function": map[string]interface{}{
				"name":        value.Name(),
				"description": value.Description(),
				"parameters":  value.ParameterSchema(),
			},
		}
	})
	body.ToolChoice = "auto"
	return body
}

func (o *OpenAIProvider) Generate(ctx context.Context, tools map[string]tool.Tool, messages []Message) (*Message, string, error) {
	if err := ValidateProviderConfig(ProviderConfig{
		Model:       o.model,
		APIKey:      o.apiKey,
		Temperature: o.temperature,
	}); err != nil {
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

func (o *OpenAIProvider) generateWithoutRetry(ctx context.Context, tools map[string]tool.Tool, messages []Message) (*Message, string, error) {
	body := o.generateBody(tools, messages, false)

	bt, err := json.MarshalIndent(body, "", "\t")
	if err != nil {
		return nil, "", NewProviderError("failed to marshal request body", err)
	}
	fmt.Println(string(bt))

	request, err := http.NewRequestWithContext(ctx, http.MethodPost, "https://api.openai.com/v1/chat/completions", bytes.NewReader(bt))
	if err != nil {
		return nil, "", NewNetworkError("failed to create HTTP request", err, 0)
	}

	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Authorization", "Bearer "+o.apiKey)

	res, err := http.DefaultClient.Do(request)
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
		return nil, "", NewNetworkError("API request failed", errors.New(buf.String()), res.StatusCode)
	}

	buf := &bytes.Buffer{}
	if _, err := io.Copy(buf, res.Body); err != nil {
		return nil, "", NewNetworkError("failed to read response body", err, res.StatusCode)
	}

	re := gjson.ParseBytes(buf.Bytes())

	if !re.Get("choices").Exists() {
		return nil, "", NewParsingError("no choices in response", nil)
	}

	for _, choice := range re.Get("choices").Array() {
		finishReason := choice.Get("finish_reason").String()

		switch finishReason {
		case "stop":
			content := choice.Get("message.content").String()
			return &Message{
				Message: content,
			}, FinishReasonStop, nil

		case "tool_calls":
			assistantMessage := Message{
				Role: "assistant",
			}
			toolCalls := []ToolCalls{}

			toolCallsArray := choice.Get("message.tool_calls").Array()
			if len(toolCallsArray) == 0 {
				return nil, "", NewParsingError("tool_calls finish reason but no tool calls found", nil)
			}

			for _, toolItem := range toolCallsArray {
				toolCall := ToolCalls{
					ID:   toolItem.Get("id").String(),
					Type: toolItem.Get("type").String(),
				}
				toolCall.Function.Name = toolItem.Get("function.name").String()
				toolCall.Function.Arguments = toolItem.Get("function.arguments").String()

				if toolCall.ID == "" || toolCall.Function.Name == "" {
					return nil, "", NewParsingError("invalid tool call format", nil)
				}

				toolCalls = append(toolCalls, toolCall)
			}
			assistantMessage.ToolCalls = toolCalls

			return &assistantMessage, FinishReasonToolCalls, nil

		case "length":
			return nil, "", NewProviderError("response truncated due to length limit", nil)
		case "content_filter":
			return nil, "", NewProviderError("response blocked by content filter", nil)
		}
	}
	return nil, "", NewParsingError("unexpected response format", nil)
}

func (o *OpenAIProvider) GenerateStreaming(ctx context.Context, tools map[string]tool.Tool, messages []Message, callback func(message Message) error) error {
	if err := ValidateProviderConfig(ProviderConfig{
		Model:       o.model,
		APIKey:      o.apiKey,
		Temperature: o.temperature,
	}); err != nil {
		return err
	}

	body := o.generateBody(tools, messages, true)

	bt, err := json.Marshal(body)
	if err != nil {
		return NewProviderError("failed to marshal streaming request body", err)
	}

	request, err := http.NewRequestWithContext(ctx, http.MethodPost, "https://api.openai.com/v1/chat/completions", bytes.NewReader(bt))
	if err != nil {
		return NewNetworkError("failed to create streaming HTTP request", err, 0)
	}

	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Authorization", "Bearer "+o.apiKey)

	res, err := http.DefaultClient.Do(request)
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
		if !bytes.HasPrefix(line, []byte("data: ")) {
			continue
		}

		data := line[6:]

		if string(data) == "[DONE]" {
			break
		}

		var chunkResponse struct {
			Choices []struct {
				Delta struct {
					Content string `json:"content"`
					Role    string `json:"role"`
				} `json:"delta"`
			} `json:"choices"`
		}

		if err := json.Unmarshal(data, &chunkResponse); err != nil {
			return NewParsingError("failed to parse streaming chunk", err).WithContext("chunk_data", string(data))
		}

		if len(chunkResponse.Choices) > 0 && chunkResponse.Choices[0].Delta.Content != "" {
			message := Message{
				Message: chunkResponse.Choices[0].Delta.Content,
			}

			if err := callback(message); err != nil {
				return NewProviderError("streaming callback failed", err)
			}
		}
	}

	return nil
}
