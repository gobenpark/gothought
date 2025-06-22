package gothought

import (
	"context"
	"errors"
	"testing"
	"time"
)

func TestIsRetryableError(t *testing.T) {
	retryableTypes := []ErrorType{ErrorTypeNetwork, ErrorTypeTimeout, ErrorTypeRateLimit}

	tests := []struct {
		name        string
		err         error
		shouldRetry bool
	}{
		{
			name:        "Network error",
			err:         NewNetworkError("network failed", nil, 500),
			shouldRetry: true,
		},
		{
			name:        "Timeout error",
			err:         NewTimeoutError("timeout", nil),
			shouldRetry: true,
		},
		{
			name:        "Rate limit error",
			err:         NewRateLimitError("rate limited", 60),
			shouldRetry: true,
		},
		{
			name:        "Provider error",
			err:         NewProviderError("provider failed", nil),
			shouldRetry: false,
		},
		{
			name:        "Validation error",
			err:         NewValidationError("field", "invalid"),
			shouldRetry: false,
		},
		{
			name:        "Regular error",
			err:         errors.New("regular error"),
			shouldRetry: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := IsRetryableError(tt.err, retryableTypes)
			if result != tt.shouldRetry {
				t.Errorf("Expected %v, got %v", tt.shouldRetry, result)
			}
		})
	}
}

func TestCalculateBackoffDelay(t *testing.T) {
	config := RetryConfig{
		InitialDelay:  time.Second,
		MaxDelay:      30 * time.Second,
		BackoffFactor: 2.0,
	}

	tests := []struct {
		attempt     int
		expectedMin time.Duration
		expectedMax time.Duration
	}{
		{1, time.Second, time.Second},
		{2, 2 * time.Second, 2 * time.Second},
		{3, 4 * time.Second, 4 * time.Second},
		{10, 30 * time.Second, 30 * time.Second}, // Should be capped at MaxDelay
	}

	for _, tt := range tests {
		t.Run("", func(t *testing.T) {
			delay := calculateBackoffDelay(tt.attempt, config)
			if delay < tt.expectedMin || delay > tt.expectedMax {
				t.Errorf("Attempt %d: expected delay between %v and %v, got %v",
					tt.attempt, tt.expectedMin, tt.expectedMax, delay)
			}
		})
	}
}

func TestWithTimeout(t *testing.T) {
	t.Run("With positive timeout", func(t *testing.T) {
		ctx := context.Background()
		timeout := 5 * time.Second

		timeoutCtx, cancel := WithTimeout(ctx, timeout)
		defer cancel()

		deadline, ok := timeoutCtx.Deadline()
		if !ok {
			t.Error("Expected context to have deadline")
		}

		expectedDeadline := time.Now().Add(timeout)
		if deadline.Before(expectedDeadline.Add(-time.Second)) || deadline.After(expectedDeadline.Add(time.Second)) {
			t.Errorf("Deadline not within expected range")
		}
	})

	t.Run("With zero timeout", func(t *testing.T) {
		ctx := context.Background()

		timeoutCtx, cancel := WithTimeout(ctx, 0)
		defer cancel()

		deadline, ok := timeoutCtx.Deadline()
		if !ok {
			t.Error("Expected context to have deadline")
		}

		expectedDeadline := time.Now().Add(30 * time.Second)
		if deadline.Before(expectedDeadline.Add(-time.Second)) || deadline.After(expectedDeadline.Add(time.Second)) {
			t.Errorf("Should use default 30s timeout")
		}
	})
}

func TestWithRetry(t *testing.T) {
	t.Run("Success on first attempt", func(t *testing.T) {
		ctx := context.Background()
		config := RetryConfig{
			MaxAttempts:     3,
			InitialDelay:    time.Millisecond,
			RetryableErrors: []ErrorType{ErrorTypeNetwork},
		}

		attempts := 0
		result, err := WithRetry(ctx, config, func(ctx context.Context) (string, error) {
			attempts++
			return "success", nil
		})

		if err != nil {
			t.Errorf("Expected no error, got %v", err)
		}

		if result != "success" {
			t.Errorf("Expected 'success', got %v", result)
		}

		if attempts != 1 {
			t.Errorf("Expected 1 attempt, got %d", attempts)
		}
	})

	t.Run("Success after retries", func(t *testing.T) {
		ctx := context.Background()
		config := RetryConfig{
			MaxAttempts:     3,
			InitialDelay:    time.Millisecond,
			RetryableErrors: []ErrorType{ErrorTypeNetwork},
		}

		attempts := 0
		result, err := WithRetry(ctx, config, func(ctx context.Context) (string, error) {
			attempts++
			if attempts < 3 {
				return "", NewNetworkError("network error", nil, 500)
			}
			return "success", nil
		})

		if err != nil {
			t.Errorf("Expected no error, got %v", err)
		}

		if result != "success" {
			t.Errorf("Expected 'success', got %v", result)
		}

		if attempts != 3 {
			t.Errorf("Expected 3 attempts, got %d", attempts)
		}
	})

	t.Run("Non-retryable error", func(t *testing.T) {
		ctx := context.Background()
		config := RetryConfig{
			MaxAttempts:     3,
			InitialDelay:    time.Millisecond,
			RetryableErrors: []ErrorType{ErrorTypeNetwork},
		}

		attempts := 0
		_, err := WithRetry(ctx, config, func(ctx context.Context) (string, error) {
			attempts++
			return "", NewValidationError("field", "invalid")
		})

		if err == nil {
			t.Error("Expected error")
		}

		// The error should be the original validation error, not wrapped by retry
		var ge *GothoughtError
		if !errors.As(err, &ge) || ge.Type != ErrorTypeValidation {
			var errType string
			if ge != nil {
				errType = string(ge.Type)
			} else {
				errType = "nil"
			}
			t.Errorf("Expected validation error, got %T with type %v", err, errType)
		}

		if attempts != 1 {
			t.Errorf("Expected 1 attempt (no retries), got %d", attempts)
		}
	})

	t.Run("Max attempts exceeded", func(t *testing.T) {
		ctx := context.Background()
		config := RetryConfig{
			MaxAttempts:     2,
			InitialDelay:    time.Millisecond,
			RetryableErrors: []ErrorType{ErrorTypeNetwork},
		}

		attempts := 0
		_, err := WithRetry(ctx, config, func(ctx context.Context) (string, error) {
			attempts++
			return "", NewNetworkError("network error", nil, 500)
		})

		if err == nil {
			t.Error("Expected error")
		}

		if !IsProviderError(err) {
			t.Errorf("Expected provider error (retry wrapper), got %T", err)
		}

		if attempts != 2 {
			t.Errorf("Expected 2 attempts, got %d", attempts)
		}
	})
}

func TestDefaultConfigs(t *testing.T) {
	t.Run("DefaultRetryConfig", func(t *testing.T) {
		config := DefaultRetryConfig()

		if config.MaxAttempts != 3 {
			t.Errorf("Expected MaxAttempts 3, got %d", config.MaxAttempts)
		}

		if config.InitialDelay != time.Second {
			t.Errorf("Expected InitialDelay 1s, got %v", config.InitialDelay)
		}

		expectedRetryable := []ErrorType{ErrorTypeNetwork, ErrorTypeTimeout, ErrorTypeRateLimit}
		if len(config.RetryableErrors) != len(expectedRetryable) {
			t.Errorf("Expected %d retryable error types, got %d", len(expectedRetryable), len(config.RetryableErrors))
		}
	})

	t.Run("DefaultTimeoutConfig", func(t *testing.T) {
		config := DefaultTimeoutConfig()

		if config.RequestTimeout != 30*time.Second {
			t.Errorf("Expected RequestTimeout 30s, got %v", config.RequestTimeout)
		}

		if config.StreamTimeout != 60*time.Second {
			t.Errorf("Expected StreamTimeout 60s, got %v", config.StreamTimeout)
		}
	})
}
