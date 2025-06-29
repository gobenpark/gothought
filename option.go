package gothought

import "github.com/gobenpark/gothought/memory"

// LanguageModel Option
type Option func(c *LanguageModel)

// WithIteration max Iterations of LLM Agent loop
func WithIteration(iter int) Option {
	return func(c *LanguageModel) {
		c.maxIterations = iter
	}
}

// WithContextManager sets a custom context manager for the language model
func WithContextManager(cm memory.MemoryManager) Option {
	return func(c *LanguageModel) {
		c.contextManager = cm
	}
}

// WithContextConfig sets the context configuration for the default context manager
func WithContextConfig(config memory.MemoryConfig) Option {
	return func(c *LanguageModel) {
		c.contextManager = memory.NewMemoryManager(config)
	}
}

// WithMemoryLimit sets the
