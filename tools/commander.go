package tools

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
)

type Commander struct {
}

func NewCommander() *Commander {
	return &Commander{}
}

func (c *Commander) ParameterSchema() map[string]interface{} {
	return map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"query": map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"command": map[string]interface{}{
						"type":        "string",
						"enum":        []string{"read", "write", "mkdir", "mv", "cp", "rm", "ls"},
						"description": "Command to execute",
					},
					"path": map[string]interface{}{
						"type":        "string",
						"description": "File or directory path",
					},
					"content": map[string]interface{}{
						"type":        "string",
						"description": "Content to write when using write command",
					},
					"destination": map[string]interface{}{
						"type":        "string",
						"description": "Destination path for mv or cp commands",
					},
				},
				"required": []string{"command", "path"},
			},
		},
	}
}

func (c *Commander) Description() string {
	return `Executes local file system operations for file and directory management tasks.
Use this for performing basic file operations like reading, writing, creating directories, and managing file structures.
Ideal for development environments when you need quick file manipulation or project structure organization.
Supports essential commands: read, write, mkdir, mv, cp, rm for comprehensive file system control.
Provides safe local file system access with support for both relative and absolute paths.`
}

func (c *Commander) Name() string {
	return "local_commander"
}

func (c *Commander) Call(ctx context.Context, params string) (string, error) {
	var commandParams struct {
		Query struct {
			Command     string `json:"command"`
			Path        string `json:"path"`
			Content     string `json:"content,omitempty"`
			Destination string `json:"destination,omitempty"`
		} `json:"query"`
	}

	if err := json.Unmarshal([]byte(params), &commandParams); err != nil {
		return "", fmt.Errorf("invalid command parameters: %v", err)
	}

	// Validate required parameters
	if commandParams.Query.Command == "" {
		return "", errors.New("command parameter is required")
	}
	if commandParams.Query.Path == "" {
		return "", errors.New("path parameter is required")
	}

	// Execute the command based on type
	switch commandParams.Query.Command {
	case "read":
		return c.executeRead(commandParams.Query.Path)
	case "write":
		if commandParams.Query.Content == "" {
			return "", errors.New("content parameter is required for write command")
		}
		return c.executeWrite(commandParams.Query.Path, commandParams.Query.Content)
	case "mkdir":
		return c.executeMkdir(commandParams.Query.Path)
	case "mv":
		if commandParams.Query.Destination == "" {
			return "", errors.New("destination parameter is required for mv command")
		}
		return c.executeMove(commandParams.Query.Path, commandParams.Query.Destination)
	case "cp":
		if commandParams.Query.Destination == "" {
			return "", errors.New("destination parameter is required for cp command")
		}
		return c.executeCopy(commandParams.Query.Path, commandParams.Query.Destination)
	case "rm":
		return c.executeRemove(commandParams.Query.Path)
	case "ls":
		return c.executeLs(commandParams.Query.Path)
	default:
		return "", fmt.Errorf("unsupported command: %s", commandParams.Query.Command)
	}
}

func (c *Commander) executeRead(path string) (string, error) {
	content, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("failed to read file %s: %v", path, err)
	}
	return fmt.Sprintf("File content of '%s':\n%s", path, string(content)), nil
}

func (c *Commander) executeWrite(path, content string) (string, error) {
	err := os.WriteFile(path, []byte(content), 0644)
	if err != nil {
		return "", fmt.Errorf("failed to write file %s: %v", path, err)
	}
	return fmt.Sprintf("Successfully wrote content to '%s'", path), nil
}

func (c *Commander) executeMkdir(path string) (string, error) {
	err := os.MkdirAll(path, 0755)
	if err != nil {
		return "", fmt.Errorf("failed to create directory %s: %v", path, err)
	}
	return fmt.Sprintf("Successfully created directory '%s'", path), nil
}

func (c *Commander) executeMove(src, dst string) (string, error) {
	err := os.Rename(src, dst)
	if err != nil {
		return "", fmt.Errorf("failed to move %s to %s: %v", src, dst, err)
	}
	return fmt.Sprintf("Successfully moved '%s' to '%s'", src, dst), nil
}

func (c *Commander) executeCopy(src, dst string) (string, error) {
	srcFile, err := os.Open(src)
	if err != nil {
		return "", fmt.Errorf("failed to open source file %s: %v", src, err)
	}
	defer srcFile.Close()

	dstFile, err := os.Create(dst)
	if err != nil {
		return "", fmt.Errorf("failed to create destination file %s: %v", dst, err)
	}
	defer dstFile.Close()

	_, err = io.Copy(dstFile, srcFile)
	if err != nil {
		return "", fmt.Errorf("failed to copy file from %s to %s: %v", src, dst, err)
	}

	return fmt.Sprintf("Successfully copied '%s' to '%s'", src, dst), nil
}

func (c *Commander) executeRemove(path string) (string, error) {
	err := os.RemoveAll(path)
	if err != nil {
		return "", fmt.Errorf("failed to remove %s: %v", path, err)
	}
	return fmt.Sprintf("Successfully removed '%s'", path), nil
}

func (c *Commander) executeLs(path string) (string, error) {
	files, err := os.ReadDir(path)
	if err != nil {
		return "", fmt.Errorf("failed to list files in %s: %v", path, err)
	}

	var result strings.Builder
	result.WriteString(fmt.Sprintf("Contents of directory '%s':\n\n", path))

	if len(files) == 0 {
		result.WriteString("Directory is empty")
		return result.String(), nil
	}

	for _, file := range files {
		fileType := "file"
		if file.IsDir() {
			fileType = "directory"
		}

		info, err := file.Info()
		if err != nil {
			result.WriteString(fmt.Sprintf("%-20s [%s] (info unavailable)\n", file.Name(), fileType))
			continue
		}

		result.WriteString(fmt.Sprintf("%-20s [%s] %10d bytes %s\n",
			file.Name(),
			fileType,
			info.Size(),
			info.ModTime().Format("2006-01-02 15:04:05")))
	}

	return result.String(), nil
}
