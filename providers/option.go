package providers

import "strings"

// Provider 공통 옵션
type ProviderOption func(interface{})

// 공통 Provider 옵션들
func WithAPIKey(apiKey string) ProviderOption {
	return func(p interface{}) {
		switch provider := p.(type) {
		case *OpenAIProvider:
			provider.apiKey = apiKey
		case *ClaudeProvider:
			provider.apiKey = apiKey
		case *GeminiProvider:
			provider.apiKey = apiKey
		case *CohereProvider:
			provider.apiKey = apiKey
		}
	}
}

func WithTemperature(temperature float32) ProviderOption {
	return func(p interface{}) {
		switch provider := p.(type) {
		case *OpenAIProvider:
			provider.temperature = temperature
		case *ClaudeProvider:
			provider.temperature = temperature
		case *GeminiProvider:
			provider.temperature = temperature
		case *CohereProvider:
			provider.temperature = temperature
		case *OllamaProvider:
			provider.temperature = temperature
		}
	}
}

func WithRetryConfig(config RetryConfig) ProviderOption {
	return func(p interface{}) {
		switch provider := p.(type) {
		case *OpenAIProvider:
			provider.retryConfig = config
		}
	}
}

func WithTimeoutConfig(config TimeoutConfig) ProviderOption {
	return func(p interface{}) {
		switch provider := p.(type) {
		case *OpenAIProvider:
			provider.timeoutConfig = config
		}
	}
}

// Claude 전용 옵션
func WithMaxTokens(maxTokens int) ProviderOption {
	return func(p interface{}) {
		switch provider := p.(type) {
		case *ClaudeProvider:
			provider.maxTokens = maxTokens
		}
	}
}

// Ollama 전용 옵션
func WithOllamaURL(url string) ProviderOption {
	return func(p interface{}) {
		switch provider := p.(type) {
		case *OllamaProvider:
			provider.baseURL = strings.TrimSuffix(url, "/")
		}
	}
}
