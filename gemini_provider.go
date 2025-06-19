package gothought

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/gobenpark/gothought/tool"
)

type GeminiResponse struct {
	Candidates []struct {
		Content struct {
			Parts []struct {
				Text string `json:"text"`
			} `json:"parts"`
			Role string `json:"role"`
		} `json:"content"`
		FinishReason string `json:"finishReason"`
		Index        int    `json:"index"`
	} `json:"candidates"`
	UsageMetadata struct {
		PromptTokenCount     int `json:"promptTokenCount"`
		CandidatesTokenCount int `json:"candidatesTokenCount"`
		TotalTokenCount      int `json:"totalTokenCount"`
		PromptTokensDetails  []struct {
			Modality   string `json:"modality"`
			TokenCount int    `json:"tokenCount"`
		} `json:"promptTokensDetails"`
	} `json:"usageMetadata"`
	ModelVersion string `json:"modelVersion"`
	ResponseId   string `json:"responseId"`
}

type GeminiProvider struct {
	apiKey string
	model  string // 모델 필드 추가
}

var _ Provider = (*GeminiProvider)(nil)

// NewGeminiProvider 생성자 수정
func NewGeminiProvider(apiKey string, model string) *GeminiProvider {
	return &GeminiProvider{
		apiKey: apiKey,
		model:  model,
	}
}

func (g *GeminiProvider) Generate(ctx context.Context, tools map[string]tool.Tool, messages []Message) (*Message, string, error) {

	content := ""
	for _, msg := range messages {
		content += msg.Message + "\n"
	}

	requestBody := map[string]interface{}{
		"contents": []map[string]interface{}{
			{
				"parts": []map[string]string{
					{
						"text": content,
					},
				},
			},
		},
		"generationConfig": map[string]interface{}{
			"thinkingConfig": map[string]interface{}{
				"thinkingBudget": 0,
			},
		},
	}

	jsonBody, err := json.Marshal(requestBody)
	if err != nil {
		return nil, "", fmt.Errorf("failed to marshal request body: %v", err)
	}

	// URL에 모델명 동적 추가
	url := fmt.Sprintf("https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent?key=%s",
		g.model, g.apiKey)

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonBody))
	if err != nil {
		return nil, "", fmt.Errorf("failed to create request: %v", err)
	}

	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return nil, "", fmt.Errorf("failed to send request: %v", err)
	}
	defer resp.Body.Close()

	buf := &bytes.Buffer{}
	if _, err := io.Copy(buf, resp.Body); err != nil {
		return nil, "", err
	}

	var generativeContent GeminiResponse
	if err := json.NewDecoder(buf).Decode(&generativeContent); err != nil {
		return nil, "", fmt.Errorf("failed to unmarshal response: %v", err)
	}

	if len(generativeContent.Candidates) == 0 || len(generativeContent.Candidates[0].Content.Parts) == 0 {
		return nil, "", fmt.Errorf("empty response from Gemini API")
	}

	return &Message{
		Role:    "assistant",
		Message: generativeContent.Candidates[0].Content.Parts[0].Text,
	}, "stop", nil
}
