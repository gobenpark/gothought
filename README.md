<p align="center">
  <img src="./go_thought.png" alt="gothought Logo" width="300">
</p>

[![Go Reference](https://pkg.go.dev/badge/github.com/gobenpark/gothought.svg)](https://pkg.go.dev/github.com/gobenpark/gothought)
# gothought

A lightweight, intuitive library for building LLM-powered applications and agents in Go.

## What is gothought?

gothought provides a simple, fluent API for interacting with Large Language Models and building LLM agents with tools. Unlike more complex frameworks, gothought focuses on minimizing boilerplate code while maintaining flexibility and extensibility.

```go
// Initialize with an OpenAI provider
provider := providers.NewOpenAIProvider("gpt-4o", 
    providers.WithAPIKey(os.Getenv("OPENAI_API_KEY")),
    providers.WithTemperature(0.7),
)

// Create a language model with the provider
model := gothought.NewLanguageModel(provider)

// Build your prompt chain and execute
response, err := model.
    SystemPrompt("You are a helpful assistant.").
    HumanPrompt("Tell me about Go programming.").
    Q(context.Background())
```

## Why gothought?

While solutions like langchain-go offer comprehensive features, they often require significant configuration and understanding of complex abstractions. gothought aims to solve common challenges in LLM application development with:

- **Minimal Setup**: Get started with just a few lines of code
- **Fluent API**: Intuitive chain-style syntax for building prompts
- **Extensible Design**: Support for multiple LLM providers through a unified interface
- **Type Safety**: Leverage Go's type system for reliable code
- **Tool Integration**: Easily add tools for building capable agents
- **Centralized Agent Loop**: Simplified agent architecture with provider-agnostic design

## Features

- Simple, chainable API for prompt construction
- Support for different message roles (system, human, AI)
- Provider interface for easy provider implementation
- Functional options for flexible configuration
- Agent loop for tool-using LLM applications
- Streaming support for real-time responses
- Multiple LLM providers (OpenAI, Claude, Gemini, Ollama, Cohere)
- Tool interface for extending LLM capabilities
- Go template-based prompt templates with `{{.Variable}}` syntax
- Comprehensive error handling and retry mechanisms
- Type-safe structured output parsing
- **Modular Architecture**: Clean separation with dedicated `models` and `messages` packages
- **Enhanced Type Consistency**: Centralized type definitions for improved maintainability

## Installation

```bash
go get github.com/gobenpark/gothought
```

## Quickstart

```go
package main

import (
    "context"
    "fmt"
    "os"
    
    "github.com/gobenpark/gothought"
    "github.com/gobenpark/gothought/providers"
)

func main() {
    // Initialize provider with model, API key and temperature
    provider := providers.NewOpenAIProvider("gpt-4o", 
        providers.WithAPIKey(os.Getenv("OPENAI_API_KEY")),
        providers.WithTemperature(0.7),
    )
    
    // Create language model with the provider
    model := gothought.NewLanguageModel(provider)
    
    // Build your prompt chain and execute
    response, err := model.
        SystemPrompt("You are a helpful coding assistant.").
        HumanPrompt("How do I read a file in Go?").
        Q(context.Background())
        
    if err != nil {
        fmt.Println("Error:", err)
        return
    }
    
    fmt.Println("Response:", response.Message)
}
```

### Using Tools with the LLM

```go
// Initialize provider
provider := gothought.NewOpenAIProvider(
    "gpt-4o", 
    os.Getenv("OPENAI_API_KEY"),
    0.7,
)

// Create language model
model := gothought.NewLanguageModel(provider)

// Add a tool to the model
braveSearchTool := tools.NewBraveSearchTool(os.Getenv("BRAVE_API_KEY"))
model.AddTool(braveSearchTool)

// Now the LLM can use the tool to answer questions
response, err := model.
    SystemPrompt("You are a helpful assistant with access to search tools.").
    HumanPrompt("What were the major tech news headlines yesterday?").
    Q(context.Background())
```

### Streaming Responses

```go
// Initialize provider
provider := gothought.NewOpenAIProvider(
    "gpt-4o", 
    os.Getenv("OPENAI_API_KEY"),
    0.7,
)

// Create language model
model := gothought.NewLanguageModel(provider)

// Use streaming API
err := model.
    SystemPrompt("You are a helpful assistant.").
    HumanPrompt("Tell me a story about space exploration.").
    QStream(context.Background(), func(msg gothought.Message) error {
        fmt.Print(msg.Message) // Print message chunks as they arrive
        return nil
    })
```

## Supported LLM Providers

gothought supports multiple LLM providers through a unified interface:

### OpenAI
```go
provider := gothought.NewOpenAIProvider("gpt-4o", os.Getenv("OPENAI_API_KEY"), 0.7)
```

### Claude (Anthropic)
```go
provider := gothought.NewClaudeProvider("claude-3-5-sonnet-20241022", os.Getenv("ANTHROPIC_API_KEY"))
```

### Google Gemini
```go
provider := gothought.NewGeminiProvider("gemini-2.5-pro", os.Getenv("GEMINI_API_KEY"))
```

### Ollama (Local LLMs)
```go
// Default localhost:11434
provider := gothought.NewOllamaProvider("llama3.1", 0.7)

// Custom Ollama server
provider := gothought.NewOllamaProvider("mistral", 0.7, "http://my-ollama-server:11434")
```

### Cohere
```go
provider := gothought.NewCohereProvider("command", os.Getenv("COHERE_API_KEY"))
```

All providers support the same fluent API and streaming capabilities!

### Using Prompt Templates

gothought supports Go's native `text/template` syntax for dynamic prompt generation:

```go
// Create a template
template, err := gothought.NewPromptTemplate("greeting", "Hello {{.Name}}, you are a {{.Role}}!")
if err != nil {
    panic(err)
}

// Use with LanguageModel
data := map[string]interface{}{
    "Name": "Alice",
    "Role": "developer",
}

response, err := model.
    SystemPromptTemplate(template, data).
    HumanPrompt("What's your favorite programming language?").
    Q(context.Background())

// Or use convenience method for one-time templates
response, err := model.
    SystemPromptf("You are a {{.Role}} assistant specializing in {{.Domain}}", map[string]interface{}{
        "Role": "helpful",
        "Domain": "Go programming",
    }).
    HumanPrompt("Explain goroutines").
    Q(context.Background())
```

## Supported Tools

- Brave Search - Web search capabilities
- Custom tools - Easily implement your own tools by implementing the Tool interface

## Creating Custom Tools

Extending the LLM with custom tools is simple:

```go
// Implement the Tool interface
type MyCustomTool struct {
    // Your tool fields
}

func (t *MyCustomTool) Name() string {
    return "my_custom_tool"
}

func (t *MyCustomTool) Description() string {
    return "A custom tool that does something useful"
}

func (t *MyCustomTool) ParameterSchema() map[string]interface{} {
    return map[string]interface{}{
        "type": "object",
        "properties": map[string]interface{}{
            "input": map[string]interface{}{
                "type": "string",
                "description": "The input for the tool",
            },
        },
        "required": []string{"input"},
    }
}

func (t *MyCustomTool) Call(ctx context.Context, params string) (string, error) {
    // Parse params and implement your tool logic
    var request struct {
        Input string `json:"input"`
    }
    if err := json.Unmarshal([]byte(params), &request); err != nil {
        return "", err
    }
    
    // Do something with request.Input
    result := "Processed: " + request.Input
    
    return result, nil
}
```

## Roadmap

### Completed ✅
- ✅ **Multiple LLM Providers**: OpenAI, Claude, Gemini, Ollama, Cohere
- ✅ **Prompt Templates**: Go template syntax with `{{.Variable}}` placeholders
- ✅ **Error Handling**: Comprehensive error types and retry mechanisms
- ✅ **Streaming Support**: Real-time response streaming for all providers
- ✅ **Tool System**: Extensible tool interface with function calling
- ✅ **Modular Architecture**: Separated provider models and message types into dedicated packages
- ✅ **Type Consistency**: Centralized type definitions across all providers for better maintainability

### In Progress 🚧
- 🚧 **Token Management**: Token counting and cost tracking system

### Future Plans 📋
- Context management for multi-turn conversations
- More built-in tools for common tasks
- Middleware support for request/response processing
- Caching mechanisms for improved performance
- Enhanced validation and error handling
- Performance optimizations
- Memory management for long conversations
- Plugin system for custom extensions

## Contributing

Contributions, suggestions, and feature requests are welcome! Feel free to open issues or submit pull requests as the project evolves.
