package gothought

import (
	"context"
	"fmt"
	"html/template"
	"os"
	"testing"

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
