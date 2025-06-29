package gothought

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/gobenpark/gothought/messages"
)

const EstimatedCharsPerToken = 4.0 // Roughly 4 characters per token for English text

// ContextManager handles conversation context and memory management
type ContextManager interface {
	// AddMessage adds a message to the context
	AddMessage(message messages.Message) error

	// GetMessages returns messages within the context limits
	GetMessages() []messages.Message

	// GetFilteredMessages returns messages optimized for the given constraints
	GetFilteredMessages(maxTokens int, modelName string) ([]messages.Message, error)

	// Clear removes all messages from context
	Clear()

	// CompressContext summarizes older messages to reduce token usage
	CompressContext(ctx context.Context, provider Provider, maxTokens int) error

	// GetTokenCount estimates total tokens in current context
	GetTokenCount(modelName string) (int, error)

	// SaveContext persists the current context
	SaveContext(sessionID string) error

	// LoadContext restores context from storage
	LoadContext(sessionID string) error
}

// ContextConfig defines memory management configuration
type ContextConfig struct {
	// MaxMessages is the maximum number of messages to keep in memory
	MaxMessages int

	// MaxTokens is the target maximum tokens for context
	MaxTokens int

	// PreserveSystemMessages keeps system prompts always in context
	PreserveSystemMessages bool

	// PreserveRecentMessages number of recent messages to always keep
	PreserveRecentMessages int

	// CompressionRatio when compressing, target this ratio of original
	CompressionRatio float64

	// StorageBackend for persistence (memory, file, database)
	StorageBackend StorageBackend
}

// DefaultContextConfig returns sensible defaults
func DefaultContextConfig() ContextConfig {
	return ContextConfig{
		MaxMessages:            50,
		MaxTokens:              4000, // Conservative default for most models
		PreserveSystemMessages: true,
		PreserveRecentMessages: 5,
		CompressionRatio:       0.3,
		StorageBackend:         NewMemoryStorage(),
	}
}

// StorageBackend defines persistence interface
type StorageBackend interface {
	Save(sessionID string, messages []messages.Message) error
	Load(sessionID string) ([]messages.Message, error)
	Delete(sessionID string) error
	Exists(sessionID string) bool
}

// MemoryStorage implements in-memory storage
type MemoryStorage struct {
	sessions map[string][]messages.Message
}

// NewMemoryStorage creates a new in-memory storage backend
func NewMemoryStorage() *MemoryStorage {
	return &MemoryStorage{
		sessions: make(map[string][]messages.Message),
	}
}

func (m *MemoryStorage) Save(sessionID string, msgs []messages.Message) error {
	m.sessions[sessionID] = make([]messages.Message, len(msgs))
	copy(m.sessions[sessionID], msgs)
	return nil
}

func (m *MemoryStorage) Load(sessionID string) ([]messages.Message, error) {
	msgs, exists := m.sessions[sessionID]
	if !exists {
		return nil, fmt.Errorf("session %s not found", sessionID)
	}
	result := make([]messages.Message, len(msgs))
	copy(result, msgs)
	return result, nil
}

func (m *MemoryStorage) Delete(sessionID string) error {
	delete(m.sessions, sessionID)
	return nil
}

func (m *MemoryStorage) Exists(sessionID string) bool {
	_, exists := m.sessions[sessionID]
	return exists
}

// FileStorage implements file-based storage
type FileStorage struct {
	baseDir string
}

// NewFileStorage creates a new file-based storage backend
func NewFileStorage(baseDir string) *FileStorage {
	return &FileStorage{
		baseDir: baseDir,
	}
}

func (f *FileStorage) Save(sessionID string, messages []messages.Message) error {
	// Ensure base directory exists
	if err := os.MkdirAll(f.baseDir, 0755); err != nil {
		return fmt.Errorf("failed to create storage directory: %w", err)
	}

	// Serialize messages to JSON
	data, err := json.MarshalIndent(messages, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal messages: %w", err)
	}

	// Write to file
	filename := filepath.Join(f.baseDir, sessionID+".json")
	if err := os.WriteFile(filename, data, 0644); err != nil {
		return fmt.Errorf("failed to write session file: %w", err)
	}

	return nil
}

func (f *FileStorage) Load(sessionID string) ([]messages.Message, error) {
	filename := filepath.Join(f.baseDir, sessionID+".json")

	// Check if file exists
	if !f.Exists(sessionID) {
		return nil, fmt.Errorf("session %s not found", sessionID)
	}

	// Read file
	data, err := os.ReadFile(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to read session file: %w", err)
	}

	// Deserialize messages
	var messages []messages.Message
	if err := json.Unmarshal(data, &messages); err != nil {
		return nil, fmt.Errorf("failed to unmarshal messages: %w", err)
	}

	return messages, nil
}

func (f *FileStorage) Delete(sessionID string) error {
	filename := filepath.Join(f.baseDir, sessionID+".json")

	if !f.Exists(sessionID) {
		return nil // Already deleted or doesn't exist
	}

	if err := os.Remove(filename); err != nil {
		return fmt.Errorf("failed to delete session file: %w", err)
	}

	return nil
}

func (f *FileStorage) Exists(sessionID string) bool {
	filename := filepath.Join(f.baseDir, sessionID+".json")
	_, err := os.Stat(filename)
	return err == nil
}

// DefaultContextManager implements ContextManager
type DefaultContextManager struct {
	config   ContextConfig
	messages []messages.Message
}

// NewContextManager creates a new context manager with the given configuration
func NewContextManager(config ContextConfig) *DefaultContextManager {
	return &DefaultContextManager{
		config:   config,
		messages: make([]messages.Message, 0),
	}
}

// AddMessage implements ContextManager.AddMessage
func (cm *DefaultContextManager) AddMessage(message messages.Message) error {
	cm.messages = append(cm.messages, message)

	// Apply message limit
	if len(cm.messages) > cm.config.MaxMessages {
		cm.trimToMessageLimit()
	}

	return nil
}

// GetMessages implements ContextManager.GetMessages
func (cm *DefaultContextManager) GetMessages() []messages.Message {
	result := make([]messages.Message, len(cm.messages))
	copy(result, cm.messages)
	return result
}

// GetFilteredMessages implements ContextManager.GetFilteredMessages
func (cm *DefaultContextManager) GetFilteredMessages(maxTokens int, modelName string) ([]messages.Message, error) {
	// Simple token estimation for now
	// TODO: Integrate with token counting system when available
	if len(cm.messages) == 0 {
		return []messages.Message{}, nil
	}

	// If no token limit specified, use config default
	if maxTokens <= 0 {
		maxTokens = cm.config.MaxTokens
	}

	// For now, use simple character-based estimation
	// Roughly 4 characters per token for English text
	estimatedTokensPerChar := 1.0 / EstimatedCharsPerToken

	var result []messages.Message
	var totalChars int

	// Always include system messages if configured
	if cm.config.PreserveSystemMessages {
		for _, msg := range cm.messages {
			if msg.Role == "system" {
				result = append(result, msg)
				totalChars += len(msg.Message)
			}
		}
	}

	// Add messages from newest to oldest until we hit token limit
	nonSystemMessages := cm.getNonSystemMessages()

	// Build a temporary slice of selected messages in reverse order
	var selectedMessages []messages.Message
	for i := len(nonSystemMessages) - 1; i >= 0; i-- {
		msg := nonSystemMessages[i]
		msgChars := len(msg.Message)

		if float64(totalChars+msgChars)*estimatedTokensPerChar > float64(maxTokens) {
			break
		}

		selectedMessages = append(selectedMessages, msg)
		totalChars += msgChars
	}

	// Reverse the selected messages and append to result (more efficient)
	for i := len(selectedMessages) - 1; i >= 0; i-- {
		result = append(result, selectedMessages[i])
	}

	return result, nil
}

// Clear implements ContextManager.Clear
func (cm *DefaultContextManager) Clear() {
	cm.messages = make([]messages.Message, 0)
}

// CompressContext implements ContextManager.CompressContext
func (cm *DefaultContextManager) CompressContext(ctx context.Context, provider Provider, maxTokens int) error {
	if len(cm.messages) <= cm.config.PreserveRecentMessages+1 {
		return nil // Not enough messages to compress
	}

	// Find messages to compress (exclude system and recent messages)
	var toCompress []messages.Message
	var toKeep []messages.Message

	recentStart := len(cm.messages) - cm.config.PreserveRecentMessages

	for i, msg := range cm.messages {
		if msg.Role == "system" && cm.config.PreserveSystemMessages {
			toKeep = append(toKeep, msg)
		} else if i >= recentStart {
			toKeep = append(toKeep, msg)
		} else {
			toCompress = append(toCompress, msg)
		}
	}

	if len(toCompress) == 0 {
		return nil
	}

	// Create compression summary
	summary, err := cm.createCompressionSummary(ctx, provider, toCompress)
	if err != nil {
		return fmt.Errorf("failed to compress context: %w", err)
	}

	// Replace compressed messages with summary
	var newMessages []messages.Message

	// Add system messages first
	for _, msg := range toKeep {
		if msg.Role == "system" {
			newMessages = append(newMessages, msg)
		}
	}

	// Add compression summary
	newMessages = append(newMessages, messages.Message{
		Role:    "system",
		Message: fmt.Sprintf("Previous conversation summary: %s", summary),
	})

	// Add recent messages
	for _, msg := range toKeep {
		if msg.Role != "system" {
			newMessages = append(newMessages, msg)
		}
	}

	cm.messages = newMessages
	return nil
}

// GetTokenCount implements ContextManager.GetTokenCount
func (cm *DefaultContextManager) GetTokenCount(modelName string) (int, error) {
	// Simple character-based estimation for now
	// TODO: Integrate with proper token counting when available
	totalChars := 0
	for _, msg := range cm.messages {
		totalChars += len(msg.Message)
	}

	// Rough estimation: 4 characters per token for English
	return int(float64(totalChars) / EstimatedCharsPerToken), nil
}

// SaveContext implements ContextManager.SaveContext
func (cm *DefaultContextManager) SaveContext(sessionID string) error {
	return cm.config.StorageBackend.Save(sessionID, cm.messages)
}

// LoadContext implements ContextManager.LoadContext
func (cm *DefaultContextManager) LoadContext(sessionID string) error {
	messages, err := cm.config.StorageBackend.Load(sessionID)
	if err != nil {
		return err
	}
	cm.messages = messages
	return nil
}

// Helper methods

func (cm *DefaultContextManager) trimToMessageLimit() {
	if len(cm.messages) <= cm.config.MaxMessages {
		return
	}

	// Keep system messages and recent messages
	var systemMessages []messages.Message
	var otherMessages []messages.Message

	for _, msg := range cm.messages {
		if msg.Role == "system" && cm.config.PreserveSystemMessages {
			systemMessages = append(systemMessages, msg)
		} else {
			otherMessages = append(otherMessages, msg)
		}
	}

	// Calculate how many non-system messages we can keep
	maxOtherMessages := cm.config.MaxMessages - len(systemMessages)
	if maxOtherMessages < 0 {
		maxOtherMessages = 0
	}

	// Keep the most recent messages
	if len(otherMessages) > maxOtherMessages {
		otherMessages = otherMessages[len(otherMessages)-maxOtherMessages:]
	}

	// Combine system and other messages
	cm.messages = append(systemMessages, otherMessages...)
}

func (cm *DefaultContextManager) getNonSystemMessages() []messages.Message {
	var result []messages.Message
	for _, msg := range cm.messages {
		if msg.Role != "system" {
			result = append(result, msg)
		}
	}
	return result
}

func (cm *DefaultContextManager) createCompressionSummary(ctx context.Context, provider Provider, msgs []messages.Message) (string, error) {
	if len(msgs) == 0 {
		return "", nil
	}

	// Create a temporary conversation for summarization
	var conversationText string
	for _, msg := range msgs {
		conversationText += fmt.Sprintf("%s: %s\n", msg.Role, msg.Message)
	}

	// Create summarization prompt
	summaryMessages := []messages.Message{
		{
			Role:    "system",
			Message: "Summarize the following conversation concisely, preserving key information and context:",
		},
		{
			Role:    "human",
			Message: conversationText,
		},
	}

	// Generate summary
	response, _, err := provider.Generate(ctx, nil, summaryMessages)
	if err != nil {
		return "", err
	}

	return response.Message, nil
}

// UpdateMaxMessages updates the maximum message limit for DefaultContextManager
func (cm *DefaultContextManager) UpdateMaxMessages(maxMessages int) {
	cm.config.MaxMessages = maxMessages
	// Apply the new limit if needed
	if len(cm.messages) > maxMessages {
		cm.trimToMessageLimit()
	}
}

// UpdateMaxTokens updates the maximum token limit for DefaultContextManager
func (cm *DefaultContextManager) UpdateMaxTokens(maxTokens int) {
	cm.config.MaxTokens = maxTokens
}

// UpdateStorageBackend updates the storage backend for DefaultContextManager
func (cm *DefaultContextManager) UpdateStorageBackend(storage StorageBackend) {
	cm.config.StorageBackend = storage
}

// GetConfig returns a copy of the current configuration
func (cm *DefaultContextManager) GetConfig() ContextConfig {
	return cm.config
}

// ContextAware interface for providers that support context management
type ContextAware interface {
	GetContextManager() ContextManager
	SetContextManager(ContextManager)
}
