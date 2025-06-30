package tools

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestNewCommander(t *testing.T) {
	commander := NewCommander()
	require.NotNil(t, commander)
}

func TestCommander_Name(t *testing.T) {
	commander := NewCommander()
	expected := "local_commander"
	require.Equal(t, expected, commander.Name())
}

func TestCommander_Description(t *testing.T) {
	commander := NewCommander()
	desc := commander.Description()
	require.Contains(t, desc, "file system operations")
}

func TestCommander_ParameterSchema(t *testing.T) {
	commander := NewCommander()
	schema := commander.ParameterSchema()

	require.NotNil(t, schema)
	require.Equal(t, "object", schema["type"])

	properties, ok := schema["properties"].(map[string]interface{})
	require.True(t, ok, "Schema should have properties field")

	query, ok := properties["query"].(map[string]interface{})
	require.True(t, ok, "Schema properties should have query field")

	queryProps, ok := query["properties"].(map[string]interface{})
	require.True(t, ok, "Query should have properties")

	command, ok := queryProps["command"].(map[string]interface{})
	require.True(t, ok, "Query properties should have command field")

	enum, ok := command["enum"].([]string)
	require.True(t, ok, "Command should have enum field")

	expectedCommands := []string{"read", "write", "mkdir", "mv", "cp", "rm", "ls"}
	require.Equal(t, len(expectedCommands), len(enum))
}

func TestCommander_Call_InvalidJSON(t *testing.T) {
	commander := NewCommander()
	ctx := context.Background()

	_, err := commander.Call(ctx, "invalid json")
	require.Error(t, err)
	require.Contains(t, err.Error(), "invalid command parameters")
}

func TestCommander_Call_MissingCommand(t *testing.T) {
	commander := NewCommander()
	ctx := context.Background()

	params := `{"query": {"path": "/test"}}`
	_, err := commander.Call(ctx, params)
	require.Error(t, err)
	require.Contains(t, err.Error(), "command parameter is required")
}

func TestCommander_Call_MissingPath(t *testing.T) {
	commander := NewCommander()
	ctx := context.Background()

	params := `{"query": {"command": "read"}}`
	_, err := commander.Call(ctx, params)
	require.Error(t, err)
	require.Contains(t, err.Error(), "path parameter is required")
}

func TestCommander_Call_UnsupportedCommand(t *testing.T) {
	commander := NewCommander()
	ctx := context.Background()

	params := `{"query": {"command": "unsupported", "path": "/test"}}`
	_, err := commander.Call(ctx, params)
	require.Error(t, err)
	require.Contains(t, err.Error(), "unsupported command")
}

func TestCommander_ExecuteRead(t *testing.T) {
	commander := NewCommander()
	ctx := context.Background()

	// Create a temporary file
	tmpDir := t.TempDir()
	testFile := filepath.Join(tmpDir, "test.txt")
	testContent := "Hello, World!"

	err := os.WriteFile(testFile, []byte(testContent), 0644)
	require.NoError(t, err)

	params := map[string]interface{}{
		"query": map[string]interface{}{
			"command": "read",
			"path":    testFile,
		},
	}

	paramsJSON, _ := json.Marshal(params)
	result, err := commander.Call(ctx, string(paramsJSON))

	require.NoError(t, err)
	require.Contains(t, result, testContent)
}

func TestCommander_ExecuteWrite(t *testing.T) {
	commander := NewCommander()
	ctx := context.Background()

	tmpDir := t.TempDir()
	testFile := filepath.Join(tmpDir, "write_test.txt")
	testContent := "Test content for write operation"

	params := map[string]interface{}{
		"query": map[string]interface{}{
			"command": "write",
			"path":    testFile,
			"content": testContent,
		},
	}

	paramsJSON, _ := json.Marshal(params)
	result, err := commander.Call(ctx, string(paramsJSON))

	require.NoError(t, err)
	require.Contains(t, result, "Successfully wrote content")

	// Verify file was actually written
	content, err := os.ReadFile(testFile)
	require.NoError(t, err)
	require.Equal(t, testContent, string(content))
}

func TestCommander_ExecuteWrite_MissingContent(t *testing.T) {
	commander := NewCommander()
	ctx := context.Background()

	params := `{"query": {"command": "write", "path": "/test"}}`
	_, err := commander.Call(ctx, params)
	require.Error(t, err)
	require.Contains(t, err.Error(), "content parameter is required")
}

func TestCommander_ExecuteMkdir(t *testing.T) {
	commander := NewCommander()
	ctx := context.Background()

	tmpDir := t.TempDir()
	newDir := filepath.Join(tmpDir, "new_directory")

	params := map[string]interface{}{
		"query": map[string]interface{}{
			"command": "mkdir",
			"path":    newDir,
		},
	}

	paramsJSON, _ := json.Marshal(params)
	result, err := commander.Call(ctx, string(paramsJSON))

	require.NoError(t, err)
	require.Contains(t, result, "Successfully created directory")

	// Verify directory was created
	_, err = os.Stat(newDir)
	require.NoError(t, err)
}

func TestCommander_ExecuteMove(t *testing.T) {
	commander := NewCommander()
	ctx := context.Background()

	tmpDir := t.TempDir()
	srcFile := filepath.Join(tmpDir, "source.txt")
	dstFile := filepath.Join(tmpDir, "destination.txt")

	// Create source file
	err := os.WriteFile(srcFile, []byte("test content"), 0644)
	require.NoError(t, err)

	params := map[string]interface{}{
		"query": map[string]interface{}{
			"command":     "mv",
			"path":        srcFile,
			"destination": dstFile,
		},
	}

	paramsJSON, _ := json.Marshal(params)
	result, err := commander.Call(ctx, string(paramsJSON))

	require.NoError(t, err)
	require.Contains(t, result, "Successfully moved")

	// Verify source file doesn't exist and destination does
	_, err = os.Stat(srcFile)
	require.True(t, os.IsNotExist(err), "Source file should not exist after move")

	_, err = os.Stat(dstFile)
	require.NoError(t, err, "Destination file should exist after move")
}

func TestCommander_ExecuteMove_MissingDestination(t *testing.T) {
	commander := NewCommander()
	ctx := context.Background()

	params := `{"query": {"command": "mv", "path": "/test"}}`
	_, err := commander.Call(ctx, params)
	require.Error(t, err)
	require.Contains(t, err.Error(), "destination parameter is required")
}

func TestCommander_ExecuteCopy(t *testing.T) {
	commander := NewCommander()
	ctx := context.Background()

	tmpDir := t.TempDir()
	srcFile := filepath.Join(tmpDir, "source.txt")
	dstFile := filepath.Join(tmpDir, "destination.txt")
	testContent := "test content for copy"

	// Create source file
	err := os.WriteFile(srcFile, []byte(testContent), 0644)
	require.NoError(t, err)

	params := map[string]interface{}{
		"query": map[string]interface{}{
			"command":     "cp",
			"path":        srcFile,
			"destination": dstFile,
		},
	}

	paramsJSON, _ := json.Marshal(params)
	result, err := commander.Call(ctx, string(paramsJSON))

	require.NoError(t, err)
	require.Contains(t, result, "Successfully copied")

	// Verify both files exist with same content
	srcContent, err := os.ReadFile(srcFile)
	require.NoError(t, err)

	dstContent, err := os.ReadFile(dstFile)
	require.NoError(t, err)

	require.Equal(t, string(srcContent), string(dstContent))
}

func TestCommander_ExecuteCopy_MissingDestination(t *testing.T) {
	commander := NewCommander()
	ctx := context.Background()

	params := `{"query": {"command": "cp", "path": "/test"}}`
	_, err := commander.Call(ctx, params)
	require.Error(t, err)
	require.Contains(t, err.Error(), "destination parameter is required")
}

func TestCommander_ExecuteRemove(t *testing.T) {
	commander := NewCommander()
	ctx := context.Background()

	tmpDir := t.TempDir()
	testFile := filepath.Join(tmpDir, "to_remove.txt")

	// Create file to remove
	err := os.WriteFile(testFile, []byte("content"), 0644)
	require.NoError(t, err)

	params := map[string]interface{}{
		"query": map[string]interface{}{
			"command": "rm",
			"path":    testFile,
		},
	}

	paramsJSON, _ := json.Marshal(params)
	result, err := commander.Call(ctx, string(paramsJSON))

	require.NoError(t, err)
	require.Contains(t, result, "Successfully removed")

	// Verify file was removed
	_, err = os.Stat(testFile)
	require.True(t, os.IsNotExist(err), "File should not exist after removal")
}

func TestCommander_ExecuteLs(t *testing.T) {
	commander := NewCommander()
	ctx := context.Background()

	tmpDir := t.TempDir()

	// Create test files and directory
	testFile := filepath.Join(tmpDir, "test_file.txt")
	testDir := filepath.Join(tmpDir, "test_dir")

	err := os.WriteFile(testFile, []byte("content"), 0644)
	require.NoError(t, err)

	err = os.Mkdir(testDir, 0755)
	require.NoError(t, err)

	params := map[string]interface{}{
		"query": map[string]interface{}{
			"command": "ls",
			"path":    tmpDir,
		},
	}

	paramsJSON, _ := json.Marshal(params)
	result, err := commander.Call(ctx, string(paramsJSON))

	require.NoError(t, err)
	require.Contains(t, result, "Contents of directory")
	require.Contains(t, result, "test_file.txt")
	require.Contains(t, result, "test_dir")
}

func TestCommander_ExecuteLs_EmptyDirectory(t *testing.T) {
	commander := NewCommander()
	ctx := context.Background()

	tmpDir := t.TempDir()

	params := map[string]interface{}{
		"query": map[string]interface{}{
			"command": "ls",
			"path":    tmpDir,
		},
	}

	paramsJSON, _ := json.Marshal(params)
	result, err := commander.Call(ctx, string(paramsJSON))

	require.NoError(t, err)
	require.Contains(t, result, "Directory is empty")
}

func TestCommander_ExecuteRead_NonexistentFile(t *testing.T) {
	commander := NewCommander()
	ctx := context.Background()

	params := map[string]interface{}{
		"query": map[string]interface{}{
			"command": "read",
			"path":    "/nonexistent/file.txt",
		},
	}

	paramsJSON, _ := json.Marshal(params)
	_, err := commander.Call(ctx, string(paramsJSON))

	require.Error(t, err)
	require.Contains(t, err.Error(), "failed to read file")
}
