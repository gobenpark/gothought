package main

import "github.com/gobenpark/gothought"

func main() {

	gothought.NewLanguageModel(gothought.NewOpenAIProvider("chatgpt-4o-mini"))

}
