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
	model       string
	apiKey      string
	temperature float32
}

var _ Provider = (*OpenAIProvider)(nil)

func NewOpenAIProvider(model string, apikey string, temperature float32) *OpenAIProvider {
	return &OpenAIProvider{apiKey: apikey, temperature: temperature, model: model}
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
	body := o.generateBody(tools, messages, false)

	bt, err := json.MarshalIndent(body, "", "\t")
	if err != nil {
		return nil, "", err
	}
	fmt.Println(string(bt))

	request, err := http.NewRequestWithContext(ctx, http.MethodPost, "https://api.openai.com/v1/chat/completions", bytes.NewReader(bt))
	if err != nil {
		return nil, "", err
	}

	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Authorization", "Bearer "+o.apiKey)

	res, err := http.DefaultClient.Do(request)
	if err != nil {
		return nil, "", err
	}

	if res.StatusCode != 200 {
		buf := bytes.Buffer{}
		if _, err := io.Copy(&buf, res.Body); err != nil {
			return nil, "", err
		}
		return nil, "", errors.New(buf.String())
	}

	buf := &bytes.Buffer{}
	if _, err := io.Copy(buf, res.Body); err != nil {
		return nil, "", err
	}

	re := gjson.ParseBytes(buf.Bytes())
	for _, choice := range re.Get("choices").Array() {

		switch choice.Get("finish_reason").String() {
		case "stop":
			return &Message{
				Message: choice.Get("message.content").String(),
			}, FinishReasonStop, nil

		/*
			tool format example:
			{
				"id": "call_Su8cd9iLod6gNvdPnbhxL2Oa",
				"type": "function",
				"function": {
				  "name": "brave_web_search",
				  "arguments": "{\"query\":\"current weather in Paris today\"}"
				}
			}
		*/
		case "tool_calls":

			assistantMessage := Message{
				Role: "assistant",
			}
			toolCalls := []ToolCalls{}
			for _, toolItem := range choice.Get("message.tool_calls").Array() {
				toolCall := ToolCalls{
					ID:   toolItem.Get("id").String(),
					Type: toolItem.Get("type").String(),
				}
				toolCall.Function.Name = toolItem.Get("function.name").String()
				toolCall.Function.Arguments = toolItem.Get("function.arguments").String()
				toolCalls = append(toolCalls, toolCall)
			}
			assistantMessage.ToolCalls = toolCalls

			return &assistantMessage, FinishReasonToolCalls, nil
		}
	}
	return nil, "", nil
}

func (o *OpenAIProvider) GenerateStreaming(ctx context.Context, tools map[string]tool.Tool, messages []Message, callback func(message Message) error) error {
	body := o.generateBody(tools, messages, true)

	bt, err := json.Marshal(body)
	if err != nil {
		return err
	}

	request, err := http.NewRequestWithContext(ctx, http.MethodPost, "https://api.openai.com/v1/chat/completions", bytes.NewReader(bt))
	if err != nil {
		return err
	}

	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Authorization", "Bearer "+o.apiKey)

	res, err := http.DefaultClient.Do(request)
	if err != nil {
		return err
	}
	defer res.Body.Close()

	if res.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(res.Body)
		return fmt.Errorf("API returned non-200 status code: %d, body: %s", res.StatusCode, string(bodyBytes))
	}

	reader := bufio.NewReader(res.Body)

	for {
		line, err := reader.ReadBytes('\n')
		if err != nil {
			if err == io.EOF {
				break
			}
			return err
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
			return err
		}

		if len(chunkResponse.Choices) > 0 && chunkResponse.Choices[0].Delta.Content != "" {
			message := Message{
				Message: chunkResponse.Choices[0].Delta.Content,
			}

			if err := callback(message); err != nil {
				return err
			}
		}
	}

	return nil
}
