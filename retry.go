package gothought

import (
	"context"
	"errors"
	"math"
	"time"
)

type RetryConfig struct {
	MaxAttempts     int
	InitialDelay    time.Duration
	MaxDelay        time.Duration
	BackoffFactor   float64
	RetryableErrors []ErrorType
}

func DefaultRetryConfig() RetryConfig {
	return RetryConfig{
		MaxAttempts:   3,
		InitialDelay:  time.Second,
		MaxDelay:      30 * time.Second,
		BackoffFactor: 2.0,
		RetryableErrors: []ErrorType{
			ErrorTypeNetwork,
			ErrorTypeTimeout,
			ErrorTypeRateLimit,
		},
	}
}

func IsRetryableError(err error, retryableTypes []ErrorType) bool {
	var ge *GothoughtError
	if !errors.As(err, &ge) {
		return false
	}

	for _, retryableType := range retryableTypes {
		if ge.Type == retryableType {
			return true
		}
	}
	return false
}

func calculateBackoffDelay(attempt int, config RetryConfig) time.Duration {
	delay := float64(config.InitialDelay) * math.Pow(config.BackoffFactor, float64(attempt-1))
	if delay > float64(config.MaxDelay) {
		delay = float64(config.MaxDelay)
	}
	return time.Duration(delay)
}

func WithTimeout(ctx context.Context, timeout time.Duration) (context.Context, context.CancelFunc) {
	if timeout <= 0 {
		timeout = 30 * time.Second
	}
	return context.WithTimeout(ctx, timeout)
}

type RetryOperation[T any] func(context.Context) (T, error)

func WithRetry[T any](ctx context.Context, config RetryConfig, operation RetryOperation[T]) (T, error) {
	var lastErr error
	var result T

	for attempt := 1; attempt <= config.MaxAttempts; attempt++ {
		result, err := operation(ctx)
		if err == nil {
			return result, nil
		}

		lastErr = err

		if !IsRetryableError(err, config.RetryableErrors) {
			return result, err
		}

		if attempt == config.MaxAttempts {
			break
		}

		delay := calculateBackoffDelay(attempt, config)

		if IsRateLimitError(err) {
			var ge *GothoughtError
			if errors.As(err, &ge) {
				if retryAfter, ok := ge.Context["retry_after"].(int); ok && retryAfter > 0 {
					delay = time.Duration(retryAfter) * time.Second
				}
			}
		}

		select {
		case <-ctx.Done():
			return result, NewTimeoutError("retry cancelled due to context timeout", ctx.Err())
		case <-time.After(delay):
		}
	}

	return result, NewProviderError("operation failed after retries", lastErr).WithContext("attempts", config.MaxAttempts)
}

type TimeoutConfig struct {
	RequestTimeout time.Duration
	StreamTimeout  time.Duration
	DefaultTimeout time.Duration
}

func DefaultTimeoutConfig() TimeoutConfig {
	return TimeoutConfig{
		RequestTimeout: 30 * time.Second,
		StreamTimeout:  60 * time.Second,
		DefaultTimeout: 30 * time.Second,
	}
}
