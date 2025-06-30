package memory

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/gobenpark/gothought/messages"
)

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
