package main

import (
	"context"
	"fmt"

	"github.com/gobenpark/gothought"
	"github.com/gobenpark/gothought/providers"
	"github.com/gobenpark/gothought/tools"
	"github.com/joho/godotenv"
)

func main() {
	err := godotenv.Load(".env")
	if err != nil {
		panic(err)
	}

	pv := providers.NewOpenAIProvider("gpt-4o-mini")
	m := gothought.NewLanguageModel(pv, gothought.WithDebug())
	m.AddTool(tools.NewCommander())
	res, err := m.HumanPrompt("which files on /Users/ben").Q(context.TODO())
	if err != nil {
		panic(err)
	}
	fmt.Println(res.Message)
}
