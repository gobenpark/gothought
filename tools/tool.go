package tools

import "context"

// Tool interface defines the contract for integrating external functionalities with an LLM-based system.
// Implementations of this interface can provide various capabilities such as data retrieval,
// computation, external API access, etc. that extend the core language model functionality.
type Tool interface {
	// Description returns a human-readable explanation of the tool's purpose and usage.
	// This description may be used in prompts to help the LLM understand when and how to use the tool.
	Description() string

	// Name returns a unique identifier for the tool.
	// This should be a concise string that can be used to reference the tool in code.
	Name() string

	// Call executes the tool's functionality with the given context and returns the result.
	// Returns the tool's output as a string or an error if the execution fails.
	Call(ctx context.Context, params string) (string, error)

	// ParameterSchema returns a JSON schema object that defines the parameters expected by this tool.
	// The schema follows the JSON Schema format (http://json-schema.org/) and is used by LLM-based
	// systems to understand how to correctly call the tool with appropriate parameters.
	ParameterSchema() map[string]interface{}
}
