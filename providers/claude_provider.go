package providers

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

	"github.com/gobenpark/gothought/messages"
	"github.com/gobenpark/gothought/tool"
	"github.com/samber/lo"
	"github.com/tidwall/gjson"
)

type ClaudeMessage struct {
	Role    string          `json:"role"`
	Content []ClaudeContent `json:"content"`
}

type ClaudeContent struct {
	Type      string      `json:"type"`
	Text      string      `json:"text,omitempty"`
	ToolUseID string      `json:"tool_use_id,omitempty"`
	ID        string      `json:"id,omitempty"`
	Name      string      `json:"name,omitempty"`
	Input     interface{} `json:"input,omitempty"`
}

type ClaudeRequest struct {
	Model       string          `json:"model"`
	MaxTokens   int             `json:"max_tokens"`
	Messages    []ClaudeMessage `json:"messages"`
	System      string          `json:"system,omitempty"`
	Temperature float32         `json:"temperature,omitempty"`
	Tools       []ClaudeTool    `json:"tools,omitempty"`
	Stream      bool            `json:"stream,omitempty"`
}

type ClaudeTool struct {
	Name        string      `json:"name"`
	Description string      `json:"description"`
	InputSchema interface{} `json:"input_schema"`
}

type ClaudeResponse struct {
	ID           string          `json:"id"`
	Type         string          `json:"type"`
	Role         string          `json:"role"`
	Content      []ClaudeContent `json:"content"`
	Model        string          `json:"model"`
	StopReason   string          `json:"stop_reason"`
	StopSequence string          `json:"stop_sequence"`
	Usage        struct {
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
	} `json:"usage"`
}

type ClaudeProvider struct {
	model       string
	apiKey      string
	temperature float32
	maxTokens   int
}

func NewClaudeProvider(model string, options ...ProviderOption) *ClaudeProvider {
	provider := &ClaudeProvider{
		model:       model,
		apiKey:      os.Getenv("ANTHROPIC_API_KEY"),
		temperature: 0.7,
		maxTokens:   4096,
	}

	for _, option := range options {
		option(provider)
	}

	return provider
}

func (c *ClaudeProvider) convertMessages(messages []messages.Message) ([]ClaudeMessage, string) {
	var claudeMessages []ClaudeMessage
	var systemMessage string

	for _, msg := range messages {
		if msg.Role == "system" {
			systemMessage = msg.Message
			continue
		}

		claudeMsg := ClaudeMessage{
			Role: msg.Role,
		}

		if len(msg.ToolCalls) > 0 {
			for _, toolCall := range msg.ToolCalls {
				var input interface{}
				if toolCall.Function.Arguments != "" {
					json.Unmarshal([]byte(toolCall.Function.Arguments), &input)
				}

				claudeMsg.Content = append(claudeMsg.Content, ClaudeContent{
					Type:  "tool_use",
					ID:    toolCall.ID,
					Name:  toolCall.Function.Name,
					Input: input,
				})
			}
		} else if msg.ToolCallID != "" {
			claudeMsg.Content = append(claudeMsg.Content, ClaudeContent{
				Type:      "tool_result",
				ToolUseID: msg.ToolCallID,
				Text:      msg.Message,
			})
		} else if msg.Message != "" {
			claudeMsg.Content = append(claudeMsg.Content, ClaudeContent{
				Type: "text",
				Text: msg.Message,
			})
		}

		if len(claudeMsg.Content) > 0 {
			claudeMessages = append(claudeMessages, claudeMsg)
		}
	}

	return claudeMessages, systemMessage
}

func (c *ClaudeProvider) convertTools(tools map[string]tool.Tool) []ClaudeTool {
	return lo.MapToSlice(tools, func(key string, value tool.Tool) ClaudeTool {
		return ClaudeTool{
			Name:        value.Name(),
			Description: value.Description(),
			InputSchema: value.ParameterSchema(),
		}
	})
}

func (c *ClaudeProvider) Generate(ctx context.Context, tools map[string]tool.Tool, msgs []messages.Message) (*messages.Message, string, error) {
	claudeMessages, systemMessage := c.convertMessages(msgs)

	request := ClaudeRequest{
		Model:       c.model,
		MaxTokens:   c.maxTokens,
		Messages:    claudeMessages,
		System:      systemMessage,
		Temperature: c.temperature,
		Tools:       c.convertTools(tools),
		Stream:      false,
	}

	reqBody, err := json.Marshal(request)
	if err != nil {
		return nil, "", err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, "https://api.anthropic.com/v1/messages", bytes.NewReader(reqBody))
	if err != nil {
		return nil, "", err
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", c.apiKey)
	req.Header.Set("anthropic-version", "2023-06-01")

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return nil, "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, "", fmt.Errorf("API returned error: %s (status code: %d)", string(bodyBytes), resp.StatusCode)
	}

	buf := &bytes.Buffer{}
	if _, err := io.Copy(buf, resp.Body); err != nil {
		return nil, "", err
	}

	re := gjson.ParseBytes(buf.Bytes())

	switch re.Get("stop_reason").String() {
	case "end_turn":
		textContent := ""
		for _, content := range re.Get("content").Array() {
			if content.Get("type").String() == "text" {
				textContent += content.Get("text").String()
			}
		}
		return &messages.Message{
			Message: textContent,
		}, messages.FinishReasonStop, nil

	case "tool_use":
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

	return nil, "", errors.New("unexpected response format")
}

func (c *ClaudeProvider) GenerateStreaming(ctx context.Context, tools map[string]tool.Tool, msgs []messages.Message, callback func(message messages.Message) error) error {
	claudeMessages, systemMessage := c.convertMessages(msgs)

	request := ClaudeRequest{
		Model:       c.model,
		MaxTokens:   c.maxTokens,
		Messages:    claudeMessages,
		System:      systemMessage,
		Temperature: c.temperature,
		Tools:       c.convertTools(tools),
		Stream:      true,
	}

	reqBody, err := json.Marshal(request)
	if err != nil {
		return err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, "https://api.anthropic.com/v1/messages", bytes.NewReader(reqBody))
	if err != nil {
		return err
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", c.apiKey)
	req.Header.Set("anthropic-version", "2023-06-01")

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("API returned error: %s (status code: %d)", string(bodyBytes), resp.StatusCode)
	}

	reader := bufio.NewReader(resp.Body)

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

		re := gjson.ParseBytes(data)
		eventType := re.Get("type").String()

		switch eventType {
		case "content_block_delta":
			if re.Get("delta.type").String() == "text_delta" {
				text := re.Get("delta.text").String()
				if text != "" {
					message := messages.Message{
						Message: text,
					}
					if err := callback(message); err != nil {
						return err
					}
				}
			}
		case "message_stop":
			return nil
		}
	}

	return nil
}
