package gothought

import (
	"github.com/gobenpark/gothought/memory"
	"go.uber.org/zap"
)

// LanguageModel Option
type Option func(c *LanguageModel)

// WithIteration max Iterations of LLM Agent loop
func WithIteration(iter int) Option {
	return func(c *LanguageModel) {
		c.maxIterations = iter
	}
}

// WithContextManager sets a custom context manager for the language model
func WithMemoryManager(cm memory.MemoryManager) Option {
	return func(c *LanguageModel) {
		c.memoryManager = cm
	}
}

// WithDebug enables debug logging using zap development logger
func WithDebug() Option {
	return func(c *LanguageModel) {
		logger, _ := zap.NewDevelopment()
		c.logger = logger
	}
}

// WithLogger sets a custom zap logger for the language model
func WithLogger(logger *zap.Logger) Option {
	return func(c *LanguageModel) {
		c.logger = logger
	}
}
