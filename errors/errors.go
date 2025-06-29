package errors

import (
	"errors"
	"fmt"
	"net/http"
)

type ErrorType string

const (
	ErrorTypeProvider      ErrorType = "provider"
	ErrorTypeTool          ErrorType = "tool"
	ErrorTypeValidation    ErrorType = "validation"
	ErrorTypeNetwork       ErrorType = "network"
	ErrorTypeParsing       ErrorType = "parsing"
	ErrorTypeTimeout       ErrorType = "timeout"
	ErrorTypeRateLimit     ErrorType = "rate_limit"
	ErrorTypeMaxIterations ErrorType = "max_iterations"
)

type GothoughtError struct {
	Type       ErrorType
	Message    string
	Cause      error
	Context    map[string]interface{}
	HTTPStatus int
}

func (e *GothoughtError) Error() string {
	if e.Cause != nil {
		return fmt.Sprintf("%s error: %s (caused by: %v)", e.Type, e.Message, e.Cause)
	}
	return fmt.Sprintf("%s error: %s", e.Type, e.Message)
}

func (e *GothoughtError) Unwrap() error {
	return e.Cause
}

func (e *GothoughtError) WithContext(key string, value interface{}) *GothoughtError {
	if e.Context == nil {
		e.Context = make(map[string]interface{})
	}
	e.Context[key] = value
	return e
}

func NewProviderError(message string, cause error) *GothoughtError {
	return &GothoughtError{
		Type:    ErrorTypeProvider,
		Message: message,
		Cause:   cause,
	}
}

func NewToolError(toolName, message string, cause error) *GothoughtError {
	err := &GothoughtError{
		Type:    ErrorTypeTool,
		Message: message,
		Cause:   cause,
		Context: map[string]interface{}{
			"tool_name": toolName,
		},
	}
	return err
}

func NewValidationError(field, message string) *GothoughtError {
	return &GothoughtError{
		Type:    ErrorTypeValidation,
		Message: message,
		Context: map[string]interface{}{
			"field": field,
		},
	}
}

func NewNetworkError(message string, cause error, statusCode int) *GothoughtError {
	return &GothoughtError{
		Type:       ErrorTypeNetwork,
		Message:    message,
		Cause:      cause,
		HTTPStatus: statusCode,
		Context: map[string]interface{}{
			"status_code": statusCode,
		},
	}
}

func NewParsingError(message string, cause error) *GothoughtError {
	return &GothoughtError{
		Type:    ErrorTypeParsing,
		Message: message,
		Cause:   cause,
	}
}

func NewTimeoutError(message string, cause error) *GothoughtError {
	return &GothoughtError{
		Type:    ErrorTypeTimeout,
		Message: message,
		Cause:   cause,
	}
}

func NewRateLimitError(message string, retryAfter int) *GothoughtError {
	return &GothoughtError{
		Type:    ErrorTypeRateLimit,
		Message: message,
		Context: map[string]interface{}{
			"retry_after": retryAfter,
		},
	}
}

func NewMaxIterationsError(maxIterations int) *GothoughtError {
	return &GothoughtError{
		Type:    ErrorTypeMaxIterations,
		Message: fmt.Sprintf("maximum iterations (%d) reached without completion", maxIterations),
		Context: map[string]interface{}{
			"max_iterations": maxIterations,
		},
	}
}

func IsProviderError(err error) bool {
	var ge *GothoughtError
	return errors.As(err, &ge) && ge.Type == ErrorTypeProvider
}

func IsToolError(err error) bool {
	var ge *GothoughtError
	return errors.As(err, &ge) && ge.Type == ErrorTypeTool
}

func IsValidationError(err error) bool {
	var ge *GothoughtError
	return errors.As(err, &ge) && ge.Type == ErrorTypeValidation
}

func IsNetworkError(err error) bool {
	var ge *GothoughtError
	return errors.As(err, &ge) && ge.Type == ErrorTypeNetwork
}

func IsTimeoutError(err error) bool {
	var ge *GothoughtError
	return errors.As(err, &ge) && ge.Type == ErrorTypeTimeout
}

func IsRateLimitError(err error) bool {
	var ge *GothoughtError
	return errors.As(err, &ge) && ge.Type == ErrorTypeRateLimit
}

func GetHTTPStatusFromError(err error) int {
	var ge *GothoughtError
	if errors.As(err, &ge) && ge.HTTPStatus != 0 {
		return ge.HTTPStatus
	}
	return http.StatusInternalServerError
}
