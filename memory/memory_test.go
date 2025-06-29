package memory

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/gobenpark/gothought/messages"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDefaultContextManager(t *testing.T) {
	t.Run("AddMessage and GetMessages", func(t *testing.T) {
		cm := NewMemoryManager(DefaultMemoryConfig())

		msg1 := messages.Message{Role: "user", Message: "Hello"}
		msg2 := messages.Message{Role: "assistant", Message: "Hi there!"}

		err := cm.AddMessage(msg1)
		require.NoError(t, err)

		err = cm.AddMessage(msg2)
		require.NoError(t, err)

		messages := cm.GetMessages()
		assert.Len(t, messages, 2)
		assert.Equal(t, msg1, messages[0])
		assert.Equal(t, msg2, messages[1])
	})

	t.Run("Clear", func(t *testing.T) {
		cm := NewMemoryManager(DefaultMemoryConfig())

		cm.AddMessage(messages.Message{Role: "user", Message: "Hello"})
		cm.AddMessage(messages.Message{Role: "assistant", Message: "Hi!"})

		assert.Len(t, cm.GetMessages(), 2)

		cm.Clear()
		assert.Len(t, cm.GetMessages(), 0)
	})

	t.Run("Message limit enforcement", func(t *testing.T) {
		config := DefaultMemoryConfig()
		config.MaxMessages = 3
		config.PreserveSystemMessages = false
		cm := NewMemoryManager(config)

		// Add 5 messages
		for i := 0; i < 5; i++ {
			cm.AddMessage(messages.Message{Role: "user", Message: "Message " + string(rune(i+'1'))})
		}

		messages := cm.GetMessages()
		assert.Len(t, messages, 3)
		// Should keep the most recent messages
		assert.Equal(t, "Message 3", messages[0].Message)
		assert.Equal(t, "Message 4", messages[1].Message)
		assert.Equal(t, "Message 5", messages[2].Message)
	})

	t.Run("Preserve system messages", func(t *testing.T) {
		config := DefaultMemoryConfig()
		config.MaxMessages = 3
		config.PreserveSystemMessages = true
		cm := NewMemoryManager(config)

		// Add system message
		cm.AddMessage(messages.Message{Role: "system", Message: "You are a helpful assistant"})

		// Add 4 user messages (should cause trimming)
		for i := 0; i < 4; i++ {
			cm.AddMessage(messages.Message{Role: "user", Message: "Message " + string(rune(i+'1'))})
		}

		messages := cm.GetMessages()
		assert.Len(t, messages, 3) // 1 system + 2 most recent user messages
		assert.Equal(t, "system", messages[0].Role)
		assert.Equal(t, "Message 3", messages[1].Message)
		assert.Equal(t, "Message 4", messages[2].Message)
	})
}

func TestFilteredMessages(t *testing.T) {
	t.Run("GetFilteredMessages with token limit", func(t *testing.T) {
		cm := NewMemoryManager(DefaultMemoryConfig())

		// Add messages with varying lengths
		cm.AddMessage(messages.Message{Role: "system", Message: "Short"})                                               // ~5 chars
		cm.AddMessage(messages.Message{Role: "user", Message: "This is a longer message"})                              // ~24 chars
		cm.AddMessage(messages.Message{Role: "assistant", Message: "This is an even longer message with more content"}) // ~47 chars
		cm.AddMessage(messages.Message{Role: "user", Message: "Short"})                                                 // ~5 chars

		// Request filtered messages with small token limit
		// Using 10 tokens = ~40 chars
		filtered, err := cm.GetFilteredMessages(10, "gpt-3.5-turbo")
		require.NoError(t, err)

		// Should include system message (preserved) + some recent messages
		assert.True(t, len(filtered) >= 1)

		// First message should be system message (if preserved)
		hasSystem := false
		for _, msg := range filtered {
			if msg.Role == "system" {
				hasSystem = true
				break
			}
		}
		assert.True(t, hasSystem)
	})
}

func TestTokenCount(t *testing.T) {
	t.Run("GetTokenCount estimation", func(t *testing.T) {
		cm := NewMemoryManager(DefaultMemoryConfig())

		// Add a message with known character count
		message := "This is exactly forty characters long!!" // 40 chars
		cm.AddMessage(messages.Message{Role: "user", Message: message})

		count, err := cm.GetTokenCount("gpt-3.5-turbo")
		require.NoError(t, err)

		// Should be approximately 40 * 0.25 = 10 tokens (allow for small variation)
		assert.InDelta(t, 10, count, 2)
	})
}

func TestMemoryStorage(t *testing.T) {
	t.Run("Save and Load", func(t *testing.T) {
		storage := NewMemoryStorage()
		messages := []messages.Message{
			{Role: "user", Message: "Hello"},
			{Role: "assistant", Message: "Hi there!"},
		}

		err := storage.Save("session1", messages)
		require.NoError(t, err)

		loaded, err := storage.Load("session1")
		require.NoError(t, err)
		assert.Equal(t, messages, loaded)
	})

	t.Run("Exists and Delete", func(t *testing.T) {
		storage := NewMemoryStorage()
		messages := []messages.Message{{Role: "user", Message: "Hello"}}

		assert.False(t, storage.Exists("session1"))

		storage.Save("session1", messages)
		assert.True(t, storage.Exists("session1"))

		err := storage.Delete("session1")
		require.NoError(t, err)
		assert.False(t, storage.Exists("session1"))
	})

	t.Run("Load non-existent session", func(t *testing.T) {
		storage := NewMemoryStorage()

		_, err := storage.Load("nonexistent")
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "not found")
	})
}

func TestFileStorage(t *testing.T) {
	tempDir := t.TempDir()

	t.Run("Save and Load", func(t *testing.T) {
		storage := NewFileStorage(tempDir)
		messages := []messages.Message{
			{Role: "user", Message: "Hello"},
			{Role: "assistant", Message: "Hi there!"},
		}

		err := storage.Save("session1", messages)
		require.NoError(t, err)

		// Check file was created
		assert.True(t, storage.Exists("session1"))

		loaded, err := storage.Load("session1")
		require.NoError(t, err)
		assert.Equal(t, messages, loaded)
	})

	t.Run("Delete", func(t *testing.T) {
		storage := NewFileStorage(tempDir)
		messages := []messages.Message{{Role: "user", Message: "Hello"}}

		storage.Save("session2", messages)
		assert.True(t, storage.Exists("session2"))

		err := storage.Delete("session2")
		require.NoError(t, err)
		assert.False(t, storage.Exists("session2"))
	})

	t.Run("Load non-existent session", func(t *testing.T) {
		storage := NewFileStorage(tempDir)

		_, err := storage.Load("nonexistent")
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "not found")
	})

	t.Run("Directory creation", func(t *testing.T) {
		nestedDir := filepath.Join(tempDir, "nested", "directory")
		storage := NewFileStorage(nestedDir)

		messages := []messages.Message{{Role: "user", Message: "Hello"}}
		err := storage.Save("session3", messages)
		require.NoError(t, err)

		// Check directory was created
		_, err = os.Stat(nestedDir)
		assert.NoError(t, err)
	})
}

//func TestContextCompression(t *testing.T) {
//	t.Run("CompressContext", func(t *testing.T) {
//		ctrl := gomock.NewController(t)
//		defer ctrl.Finish()
//
//		mockProvider := gothought.NewMockProvider(ctrl)
//
//		config := DefaultMemoryConfig()
//		config.PreserveRecentMessages = 2
//		cm := NewMemoryManager(config)
//
//		// Add system message
//		cm.AddMessage(messages.Message{Role: "system", Message: "You are a helpful assistant"})
//
//		// Add several user/assistant messages
//		cm.AddMessage(messages.Message{Role: "user", Message: "What is the weather?"})
//		cm.AddMessage(messages.Message{Role: "assistant", Message: "I don't have access to weather data."})
//		cm.AddMessage(messages.Message{Role: "user", Message: "What is 2+2?"})
//		cm.AddMessage(messages.Message{Role: "assistant", Message: "2+2 equals 4."})
//		cm.AddMessage(messages.Message{Role: "user", Message: "Tell me a joke"})
//		cm.AddMessage(messages.Message{Role: "assistant", Message: "Why did the chicken cross the road?"})
//
//		// Mock the summarization call
//		mockProvider.EXPECT().Generate(
//			gomock.Any(),
//			gomock.Nil(),
//			gomock.Any(),
//		).Return(&messages.Message{Message: "Previous conversation covered weather inquiry and math question."}, "", nil)
//
//		err := cm.CompressContext(context.Background(), mockProvider, 1000)
//		require.NoError(t, err)
//
//		messages := cm.GetMessages()
//
//		// Should have: system message + summary + 2 recent messages
//		assert.True(t, len(messages) >= 4)
//
//		// First should be system message
//		assert.Equal(t, "system", messages[0].Role)
//
//		// Second should be summary
//		assert.Equal(t, "system", messages[1].Role)
//		assert.Contains(t, messages[1].Message, "Previous conversation summary:")
//
//		// Last 2 should be the most recent messages
//		assert.Equal(t, "Tell me a joke", messages[len(messages)-2].Message)
//		assert.Equal(t, "Why did the chicken cross the road?", messages[len(messages)-1].Message)
//	})
//}
//
//func TestLanguageModelContextIntegration(t *testing.T) {
//	t.Run("Context manager integration", func(t *testing.T) {
//		ctrl := gomock.NewController(t)
//		defer ctrl.Finish()
//
//		mockProvider := gothought.NewMockProvider(ctrl)
//
//		config := DefaultMemoryConfig()
//		config.MaxMessages = 5
//
//		model := NewLanguageModel(mockProvider, WithContextConfig(config))
//
//		// Test context manager is initialized
//		assert.NotNil(t, model.GetContextManager())
//
//		// Test conversation save/load
//		model.SystemPrompt("You are a helpful assistant").
//			HumanPrompt("Hello").
//			AIPrompt("Hi there!")
//
//		err := model.SaveConversation("test-session")
//		require.NoError(t, err)
//
//		// Clear and load
//		model.ClearConversation()
//		assert.Len(t, model.GetContextManager().GetMessages(), 0)
//
//		err = model.LoadConversation("test-session")
//		require.NoError(t, err)
//		assert.Len(t, model.GetContextManager().GetMessages(), 3)
//	})
//
//	t.Run("Token count estimation", func(t *testing.T) {
//		ctrl := gomock.NewController(t)
//		defer ctrl.Finish()
//
//		mockProvider := gothought.NewMockProvider(ctrl)
//		model := NewLanguageModel(mockProvider)
//
//		model.HumanPrompt("This is exactly forty characters long!!")
//
//		count, err := model.GetConversationTokenCount("gpt-3.5-turbo")
//		require.NoError(t, err)
//		assert.InDelta(t, 10, count, 2) // 40 chars * 0.25 = ~10 tokens
//	})
//
//	t.Run("Optimized messages", func(t *testing.T) {
//		ctrl := gomock.NewController(t)
//		defer ctrl.Finish()
//
//		mockProvider := gothought.NewMockProvider(ctrl)
//		model := NewLanguageModel(mockProvider)
//
//		model.SystemPrompt("System prompt").
//			HumanPrompt("This is a longer message that takes more tokens").
//			AIPrompt("This is an even longer response that definitely uses more tokens than the previous message").
//			HumanPrompt("Short")
//
//		optimized, err := model.GetOptimizedMessages(20, "gpt-3.5-turbo")
//		require.NoError(t, err)
//
//		// Should have filtered out some messages due to token limit
//		assert.True(t, len(optimized) < 4)
//
//		// Should preserve system message
//		hasSystem := false
//		for _, msg := range optimized {
//			if msg.Role == "system" {
//				hasSystem = true
//				break
//			}
//		}
//		assert.True(t, hasSystem)
//	})
//}
//
//func TestContextManagerOptions(t *testing.T) {
//	t.Run("WithMemoryLimit", func(t *testing.T) {
//		ctrl := gomock.NewController(t)
//		defer ctrl.Finish()
//
//		mockProvider := gothought.NewMockProvider(ctrl)
//		model := NewLanguageModel(mockProvider, WithMemoryLimit(3))
//
//		// Add 5 messages
//		for i := 0; i < 5; i++ {
//			model.HumanPrompt("Message " + string(rune(i+'1')))
//		}
//
//		// Should only keep 3 messages
//		assert.Len(t, model.GetContextManager().GetMessages(), 3)
//	})
//
//	t.Run("WithTokenLimit", func(t *testing.T) {
//		ctrl := gomock.NewController(t)
//		defer ctrl.Finish()
//
//		mockProvider := gothought.NewMockProvider(ctrl)
//		model := gothought.NewLanguageModel(mockProvider, WithTokenLimit(100))
//
//		cm := model.GetContextManager().(*DefaultMemoryManager)
//		assert.Equal(t, 100, cm.GetConfig().MaxTokens)
//	})
//
//	t.Run("WithContextConfig for persistent storage", func(t *testing.T) {
//		ctrl := gomock.NewController(t)
//		defer ctrl.Finish()
//
//		mockProvider := gothought.NewMockProvider(ctrl)
//		tempDir := t.TempDir()
//		fileStorage := NewFileStorage(tempDir)
//
//		config := DefaultMemoryConfig()
//		config.StorageBackend = fileStorage
//
//		model := NewLanguageModel(mockProvider, WithContextConfig(config))
//
//		model.HumanPrompt("Test message")
//		err := model.SaveConversation("test-file-session")
//		require.NoError(t, err)
//
//		// Check file was created
//		assert.True(t, fileStorage.Exists("test-file-session"))
//	})
//}
