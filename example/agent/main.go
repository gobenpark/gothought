package main

import (
	"github.com/gobenpark/gothought"
	"github.com/gobenpark/gothought/providers"
)

func main() {

	gothought.NewLanguageModel(providers.NewOpenAIProvider("chatgpt-4o-mini"))

}
