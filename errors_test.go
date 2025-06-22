package gothought

import (
	"errors"
	"testing"
)

func TestGothoughtError(t *testing.T) {
	t.Run("Error message formatting", func(t *testing.T) {
		err := NewProviderError("test message", nil)
		expected := "provider error: test message"
		if err.Error() != expected {
			t.Errorf("Expected %q, got %q", expected, err.Error())
		}
	})

	t.Run("Error with cause", func(t *testing.T) {
		cause := errors.New("original error")
		err := NewProviderError("test message", cause)
		expected := "provider error: test message (caused by: original error)"
		if err.Error() != expected {
			t.Errorf("Expected %q, got %q", expected, err.Error())
		}
	})

	t.Run("Error with context", func(t *testing.T) {
		err := NewProviderError("test message", nil).WithContext("key", "value")
		if err.Context["key"] != "value" {
			t.Errorf("Expected context key 'key' to be 'value', got %v", err.Context["key"])
		}
	})

	t.Run("Error unwrapping", func(t *testing.T) {
		cause := errors.New("original error")
		err := NewProviderError("test message", cause)

		if errors.Unwrap(err) != cause {
			t.Errorf("Expected unwrapped error to be the original cause")
		}
	})
}

func TestErrorTypes(t *testing.T) {
	tests := []struct {
		name        string
		err         *GothoughtError
		checkFunc   func(error) bool
		shouldMatch bool
	}{
		{
			name:        "Provider error check",
			err:         NewProviderError("test", nil),
			checkFunc:   IsProviderError,
			shouldMatch: true,
		},
		{
			name:        "Tool error check",
			err:         NewToolError("test_tool", "test", nil),
			checkFunc:   IsToolError,
			shouldMatch: true,
		},
		{
			name:        "Validation error check",
			err:         NewValidationError("field", "test"),
			checkFunc:   IsValidationError,
			shouldMatch: true,
		},
		{
			name:        "Network error check",
			err:         NewNetworkError("test", nil, 500),
			checkFunc:   IsNetworkError,
			shouldMatch: true,
		},
		{
			name:        "Timeout error check",
			err:         NewTimeoutError("test", nil),
			checkFunc:   IsTimeoutError,
			shouldMatch: true,
		},
		{
			name:        "Rate limit error check",
			err:         NewRateLimitError("test", 60),
			checkFunc:   IsRateLimitError,
			shouldMatch: true,
		},
		{
			name:        "Wrong type check",
			err:         NewProviderError("test", nil),
			checkFunc:   IsToolError,
			shouldMatch: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.checkFunc(tt.err)
			if result != tt.shouldMatch {
				t.Errorf("Expected %v, got %v", tt.shouldMatch, result)
			}
		})
	}
}

func TestSpecificErrorTypes(t *testing.T) {
	t.Run("ToolError with context", func(t *testing.T) {
		err := NewToolError("brave_search", "search failed", errors.New("network error"))

		if err.Type != ErrorTypeTool {
			t.Errorf("Expected error type to be %v, got %v", ErrorTypeTool, err.Type)
		}

		if err.Context["tool_name"] != "brave_search" {
			t.Errorf("Expected tool_name context to be 'brave_search', got %v", err.Context["tool_name"])
		}
	})

	t.Run("NetworkError with status code", func(t *testing.T) {
		err := NewNetworkError("API request failed", nil, 429)

		if err.HTTPStatus != 429 {
			t.Errorf("Expected HTTP status to be 429, got %d", err.HTTPStatus)
		}

		if err.Context["status_code"] != 429 {
			t.Errorf("Expected status_code context to be 429, got %v", err.Context["status_code"])
		}
	})

	t.Run("RateLimitError with retry after", func(t *testing.T) {
		err := NewRateLimitError("rate limited", 120)

		if err.Context["retry_after"] != 120 {
			t.Errorf("Expected retry_after context to be 120, got %v", err.Context["retry_after"])
		}
	})

	t.Run("MaxIterationsError with context", func(t *testing.T) {
		err := NewMaxIterationsError(10)

		if err.Type != ErrorTypeMaxIterations {
			t.Errorf("Expected error type to be %v, got %v", ErrorTypeMaxIterations, err.Type)
		}

		if err.Context["max_iterations"] != 10 {
			t.Errorf("Expected max_iterations context to be 10, got %v", err.Context["max_iterations"])
		}
	})
}

func TestGetHTTPStatusFromError(t *testing.T) {
	tests := []struct {
		name           string
		err            error
		expectedStatus int
	}{
		{
			name:           "Network error with status",
			err:            NewNetworkError("test", nil, 404),
			expectedStatus: 404,
		},
		{
			name:           "Error without status",
			err:            NewProviderError("test", nil),
			expectedStatus: 500,
		},
		{
			name:           "Non-Gothought error",
			err:            errors.New("regular error"),
			expectedStatus: 500,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			status := GetHTTPStatusFromError(tt.err)
			if status != tt.expectedStatus {
				t.Errorf("Expected status %d, got %d", tt.expectedStatus, status)
			}
		})
	}
}
