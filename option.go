package gothought

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
