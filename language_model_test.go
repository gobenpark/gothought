package gothought

import (
	"context"
	"fmt"
	"os"
	"testing"
	"text/template"

	"github.com/gobenpark/gothought/tool"
	"github.com/stretchr/testify/require"
	"go.uber.org/mock/gomock"
)

func TestOpenAI(t *testing.T) {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		t.Skip("OPENAI_API_KEY not set, skipping integration test")
	}

	braveAPIKey := os.Getenv("BRAVE_API_KEY")
	if braveAPIKey == "" {
		t.Skip("BRAVE_API_KEY not set, skipping integration test")
	}

	op := NewOpenAIProvider("gpt-4o-mini", apiKey, 0.7)
	cli := NewLanguageModel(op, WithIteration(10))

	cli.AddTool(tool.NewBraveSearchTool(braveAPIKey))
	result, err := cli.
		SystemPrompt("you are a web search ai").
		HumanPrompt(`
오늘 날씨는 어때?
`).
		Q(context.TODO())

	require.NoError(t, err)
	fmt.Println(result)
}

func TestTemplate(t *testing.T) {
	funcMap := template.FuncMap{
		"hasKey": func(m map[string]string, key string) bool {
			_, ok := m[key]
			return ok
		},
	}
	tm := template.New("template").Funcs(funcMap)
	temp, err := tm.Parse(`Hello {{.Name}}
{{if hasKey . "Ben" }}
hello
{{- end}}

`)
	require.NoError(t, err)
	temp.Execute(os.Stdout, map[string]string{
		"Name": "ben",
		"Ben":  "",
	})
}

func TestPromptTemplate_NewPromptTemplate(t *testing.T) {
	t.Run("valid template", func(t *testing.T) {
		template, err := NewPromptTemplate("test", "Hello {{.Name}}")
		require.NoError(t, err)
		require.NotNil(t, template)
		require.Equal(t, "test", template.Name())
	})

	t.Run("empty name", func(t *testing.T) {
		_, err := NewPromptTemplate("", "Hello {{.Name}}")
		require.Error(t, err)
		require.Contains(t, err.Error(), "template name cannot be empty")
	})

	t.Run("invalid template syntax", func(t *testing.T) {
		_, err := NewPromptTemplate("test", "Hello {{.Name")
		require.Error(t, err)
		require.Contains(t, err.Error(), "invalid template syntax")
	})
}

func TestPromptTemplate_Execute(t *testing.T) {
	t.Run("execute with struct", func(t *testing.T) {
		type User struct {
			Name string
			Role string
		}

		template, err := NewPromptTemplate("user", "Hello {{.Name}}, you are a {{.Role}}")
		require.NoError(t, err)

		result, err := template.Execute(User{Name: "Alice", Role: "developer"})
		require.NoError(t, err)
		require.Equal(t, "Hello Alice, you are a developer", result)
	})

	t.Run("execute with map", func(t *testing.T) {
		template, err := NewPromptTemplate("map", "Hello {{.Name}}, you are {{.Age}} years old")
		require.NoError(t, err)

		data := map[string]interface{}{
			"Name": "Bob",
			"Age":  30,
		}

		result, err := template.Execute(data)
		require.NoError(t, err)
		require.Equal(t, "Hello Bob, you are 30 years old", result)
	})

	t.Run("missing variable shows no value", func(t *testing.T) {
		template, err := NewPromptTemplate("missing", "Hello {{.Name}}, {{.Missing}}")
		require.NoError(t, err)

		data := map[string]interface{}{"Name": "Charlie"}
		result, err := template.Execute(data)
		require.NoError(t, err)
		require.Equal(t, "Hello Charlie, <no value>", result)
	})
}

func TestPromptTemplate_WithFuncs(t *testing.T) {
	t.Run("custom functions", func(t *testing.T) {
		funcMap := template.FuncMap{
			"upper": func(s string) string { return "[" + s + "]" },
			"add":   func(a, b int) int { return a + b },
		}

		templ, err := NewPromptTemplateWithFuncs(
			"funcs",
			"Hello {{upper .Name}}, result: {{add .A .B}}",
			funcMap,
		)
		require.NoError(t, err)

		data := map[string]interface{}{
			"Name": "test",
			"A":    5,
			"B":    3,
		}

		result, err := templ.Execute(data)
		require.NoError(t, err)
		require.Equal(t, "Hello [test], result: 8", result)
	})
}

func TestPromptTemplate_Clone(t *testing.T) {
	template, err := NewPromptTemplate("original", "Hello {{.Name}}")
	require.NoError(t, err)

	cloned, err := template.Clone()
	require.NoError(t, err)
	require.NotNil(t, cloned)
	require.Contains(t, cloned.Name(), "clone")

	// Both should work independently
	data := map[string]interface{}{"Name": "Test"}

	origResult, err := template.Execute(data)
	require.NoError(t, err)

	clonedResult, err := cloned.Execute(data)
	require.NoError(t, err)

	require.Equal(t, origResult, clonedResult)
}

func TestLanguageModel_TemplateIntegration(t *testing.T) {
	ctrl := gomock.NewController(t)
	mockProvider := NewMockProvider(ctrl)

	t.Run("HumanPromptTemplate", func(t *testing.T) {
		template, err := NewPromptTemplate("greeting", "Hello {{.Name}}, you are a {{.Role}}")
		require.NoError(t, err)

		expectedMessages := []Message{
			{
				Role:    "user",
				Message: "Hello Alice, you are a developer",
			},
		}

		mockProvider.EXPECT().Generate(gomock.Any(), gomock.Any(), expectedMessages).Return(
			&Message{Role: "assistant", Message: "Hi Alice!"},
			"stop",
			nil,
		)

		model := NewLanguageModel(mockProvider)
		data := map[string]interface{}{"Name": "Alice", "Role": "developer"}

		res, err := model.HumanPromptTemplate(template, data).Q(context.TODO())
		require.NoError(t, err)
		require.Equal(t, "Hi Alice!", res.Message)
	})

	t.Run("SystemPromptTemplate", func(t *testing.T) {
		template, err := NewPromptTemplate("system", "You are a {{.Role}} assistant")
		require.NoError(t, err)

		expectedMessages := []Message{
			{
				Role:    "system",
				Message: "You are a helpful assistant",
			},
			{
				Role:    "user",
				Message: "Hello",
			},
		}

		mockProvider.EXPECT().Generate(gomock.Any(), gomock.Any(), expectedMessages).Return(
			&Message{Role: "assistant", Message: "Hello there!"},
			"stop",
			nil,
		)

		model := NewLanguageModel(mockProvider)
		data := map[string]interface{}{"Role": "helpful"}

		res, err := model.
			SystemPromptTemplate(template, data).
			HumanPrompt("Hello").
			Q(context.TODO())
		require.NoError(t, err)
		require.Equal(t, "Hello there!", res.Message)
	})

	t.Run("HumanPromptf convenience method", func(t *testing.T) {
		expectedMessages := []Message{
			{
				Role:    "user",
				Message: "Hello Bob, you are 30 years old",
			},
		}

		mockProvider.EXPECT().Generate(gomock.Any(), gomock.Any(), expectedMessages).Return(
			&Message{Role: "assistant", Message: "Nice to meet you!"},
			"stop",
			nil,
		)

		model := NewLanguageModel(mockProvider)
		data := map[string]interface{}{"Name": "Bob", "Age": 30}

		res, err := model.HumanPromptf("Hello {{.Name}}, you are {{.Age}} years old", data).Q(context.TODO())
		require.NoError(t, err)
		require.Equal(t, "Nice to meet you!", res.Message)
	})

	t.Run("SystemPromptf convenience method", func(t *testing.T) {
		expectedMessages := []Message{
			{
				Role:    "system",
				Message: "You are a coding assistant",
			},
		}

		mockProvider.EXPECT().Generate(gomock.Any(), gomock.Any(), expectedMessages).Return(
			&Message{Role: "assistant", Message: "Ready to help!"},
			"stop",
			nil,
		)

		model := NewLanguageModel(mockProvider)
		data := map[string]interface{}{"Type": "coding"}

		res, err := model.SystemPromptf("You are a {{.Type}} assistant", data).Q(context.TODO())
		require.NoError(t, err)
		require.Equal(t, "Ready to help!", res.Message)
	})

	t.Run("template error handling", func(t *testing.T) {
		expectedMessages := []Message{
			{
				Role:    "user",
				Message: "Hello Test, <no value>",
			},
		}

		mockProvider.EXPECT().Generate(gomock.Any(), gomock.Any(), expectedMessages).Return(
			&Message{Role: "assistant", Message: "Error handled"},
			"stop",
			nil,
		)

		model := NewLanguageModel(mockProvider)
		data := map[string]interface{}{"Name": "Test"}

		// This should handle missing variables gracefully by showing <no value>
		res, err := model.HumanPromptf("Hello {{.Name}}, {{.Missing}}", data).Q(context.TODO())
		require.NoError(t, err)
		require.Equal(t, "Error handled", res.Message)
	})
}

func TestOpenAIProvider_Generate(t *testing.T) {
	ctrl := gomock.NewController(t)
	op := NewMockProvider(ctrl)

	expectOpenAIMessage := []Message{
		{
			Role:    "system",
			Message: "you are a ai",
		},
		{
			Role:    "user",
			Message: "what time is today?",
		},
	}

	op.EXPECT().Generate(gomock.Any(), gomock.Any(), expectOpenAIMessage).Return(&Message{
		Role:    "assistant",
		Message: "time is 00:00",
	}, "stop", nil)
	model := NewLanguageModel(op)

	res, err := model.SystemPrompt("you are a ai").HumanPrompt("what time is today?").Q(context.TODO())
	require.NoError(t, err)
	require.Equal(t, "time is 00:00", res.Message)
}
