package main

import (
	"context"
	"fmt"

	"github.com/gobenpark/gothought"
	"github.com/gobenpark/gothought/messages"
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
	m := gothought.NewLanguageModel(pv, gothought.WithDebug()) // No memory by default

	// Test 1: Without tools (should work with streaming)
	fmt.Println("=== Test 1: Streaming WITHOUT tools ===")
	callbackCalled := false
	err = m.HumanPrompt("Tell me a short joke").QStream(context.TODO(), func(message messages.Message) error {
		callbackCalled = true
		fmt.Printf("Message: %s", message.Message)
		return nil
	})
	fmt.Printf("\nCallback was called: %v\n", callbackCalled)
	if err != nil {
		fmt.Printf("Error in test 1: %v\n", err)
	}

	// Test 2: With tools (might have issues with streaming)
	fmt.Println("\n=== Test 2: Streaming WITH tools ===")
	m.AddTool(tools.NewCommander())
	m.ClearConversation()
	callbackCalled2 := false
	err = m.HumanPrompt("List files in /tmp").QStream(context.TODO(), func(message messages.Message) error {
		callbackCalled2 = true
		fmt.Printf("%s", message.Message)
		return nil
	})
	fmt.Printf("\nCallback was called: %v\n", callbackCalled2)

	if err != nil {
		fmt.Printf("Error: %v\n", err)
		panic(err)
	}

	fmt.Println("Streaming completed!")

	// Test 3: Non-streaming with tools (should work)
	fmt.Println("\n=== Test 3: Non-streaming WITH tools ===")
	response, err := m.HumanPrompt("List files in /tmp").Q(context.TODO())
	if err != nil {
		fmt.Printf("Error in test 3: %v\n", err)
	} else {
		fmt.Printf("Response: %s\n", response.Message)
	}
}
