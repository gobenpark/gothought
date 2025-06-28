package gothought

import "strings"

// LanguageModel Option
type Option func(c *LanguageModel)

// WithIteration max Iterations of LLM Agent loop
func WithIteration(iter int) Option {
	return func(c *LanguageModel) {
		c.maxIterations = iter
	}
}

// WithContextManager sets a custom context manager for the language model
func WithContextManager(cm ContextManager) Option {
	return func(c *LanguageModel) {
		c.contextManager = cm
	}
}

// WithContextConfig sets the context configuration for the default context manager
func WithContextConfig(config ContextConfig) Option {
	return func(c *LanguageModel) {
		c.contextManager = NewContextManager(config)
	}
}

// WithMemoryLimit sets the maximum number of messages to keep in memory
func WithMemoryLimit(maxMessages int) Option {
	return func(c *LanguageModel) {
		// If no context manager exists, create one with default config
		if c.contextManager == nil {
			config := DefaultContextConfig()
			config.MaxMessages = maxMessages
			c.contextManager = NewContextManager(config)
		} else if dcm, ok := c.contextManager.(*DefaultContextManager); ok {
			// Update existing DefaultContextManager config
			dcm.UpdateMaxMessages(maxMessages)
		} else {
			// For custom context managers, create a new default one
			config := DefaultContextConfig()
			config.MaxMessages = maxMessages
			c.contextManager = NewContextManager(config)
		}
	}
}

// WithTokenLimit sets the target maximum tokens for context
func WithTokenLimit(maxTokens int) Option {
	return func(c *LanguageModel) {
		// If no context manager exists, create one with default config
		if c.contextManager == nil {
			config := DefaultContextConfig()
			config.MaxTokens = maxTokens
			c.contextManager = NewContextManager(config)
		} else if dcm, ok := c.contextManager.(*DefaultContextManager); ok {
			// Update existing DefaultContextManager config
			dcm.UpdateMaxTokens(maxTokens)
		} else {
			// For custom context managers, create a new default one
			config := DefaultContextConfig()
			config.MaxTokens = maxTokens
			c.contextManager = NewContextManager(config)
		}
	}
}

// WithPersistentStorage enables persistent storage with the given storage backend
func WithPersistentStorage(storage StorageBackend) Option {
	return func(c *LanguageModel) {
		// If no context manager exists, create one with default config
		if c.contextManager == nil {
			config := DefaultContextConfig()
			config.StorageBackend = storage
			c.contextManager = NewContextManager(config)
		} else if dcm, ok := c.contextManager.(*DefaultContextManager); ok {
			// Update existing DefaultContextManager config
			dcm.UpdateStorageBackend(storage)
		} else {
			// For custom context managers, create a new default one
			config := DefaultContextConfig()
			config.StorageBackend = storage
			c.contextManager = NewContextManager(config)
		}
	}
}

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
