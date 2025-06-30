package main

import (
	"context"
	"fmt"

	"github.com/gobenpark/gothought"
	"github.com/gobenpark/gothought/providers"
	"github.com/gobenpark/gothought/tools"
)

func main() {

	m := gothought.NewLanguageModel(providers.NewOpenAIProvider("gpt-4o-mini"))
	m.AddTool(tools.NewCommander())
	res, err := m.HumanPrompt("which files on /Users/ben").Q(context.TODO())
	if err != nil {
		panic(err)
	}
	fmt.Println(res.Message)

}
