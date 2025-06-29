package tools

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"

	"github.com/tidwall/gjson"
)

// BraveSearchResult represents the structure of search results returned by the Brave API
type BraveSearchResult struct {
	Type         string `json:"type"`
	Title        string `json:"title"`
	URL          string `json:"url"`
	Description  string `json:"description"`
	FaviconURL   string `json:"favicon_url,omitempty"`
	Age          string `json:"age,omitempty"`
	IsFamily     bool   `json:"is_family,omitempty"`
	IsImgFocused bool   `json:"is_img_focused,omitempty"`
}

// BraveSearchResponse represents the overall response from the Brave Search API
type BraveSearchResponse struct {
	Type    string              `json:"type"`
	Results []BraveSearchResult `json:"results"`
	Mixed   struct {
		News []BraveSearchResult `json:"news"`
	} `json:"mixed"`
}

// BraveSearchParams defines the parameters for a Brave search query
type BraveSearchParams struct {
	Query  string `json:"query"`
	Count  int    `json:"count"`
	Offset int    `json:"offset"`
}

// BraveSearchTool implements the Tool interface for searching with Brave
type BraveSearchTool struct {
	apiKey string
}

// NewBraveSearchTool creates a new instance of BraveSearchTool
func NewBraveSearchTool(key string) *BraveSearchTool {
	return &BraveSearchTool{
		apiKey: key,
	}
}

// ParameterSchema function the parameters structure for a Brave search query
func (b *BraveSearchTool) ParameterSchema() map[string]interface{} {
	return map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"query": map[string]interface{}{
				"type":        "string",
				"description": "The user’s search query term. Query can not be empty. Maximum of 400 characters and 50 words in the query.",
			},
			"count": map[string]interface{}{
				"type":        "integer",
				"description": "Number of results to return (default: 10, max: 20)",
				"default":     10,
			},
			"offset": map[string]interface{}{
				"type":        "integer",
				"description": "Pagination offset",
				"default":     0,
			},
		},
		"required": []string{"query"},
	}
}

// Description returns a description of the tool
func (b *BraveSearchTool) Description() string {
	return `Performs a web search using the Brave Search API, ideal for general queries, news, articles, and online content.
Use this for broad information gathering, recent events, or when you need diverse web sources.
Supports pagination, content filtering, and freshness controls.
Maximum 20 results per request, with offset for pagination.`
}

// Name returns the name of the tool
func (b *BraveSearchTool) Name() string {
	return "brave_web_search"
}

// Call executes a search query against the Brave API
func (b *BraveSearchTool) Call(ctx context.Context, params string) (string, error) {
	// Parse the input parameters
	var searchParams BraveSearchParams
	if err := json.Unmarshal([]byte(params), &searchParams); err != nil {
		return "", fmt.Errorf("invalid search parameters: %v", err)
	}

	// Validate parameters
	if searchParams.Query == "" {
		return "", errors.New("query parameter is required")
	}

	// Set default values if not provided
	if searchParams.Count <= 0 || searchParams.Count > 20 {
		searchParams.Count = 10 // Default to 10 results
	}

	// Build the request URL
	baseURL := "https://api.search.brave.com/res/v1/web/search"
	u, err := url.Parse(baseURL)
	if err != nil {
		return "", err
	}

	// Add query parameters
	q := u.Query()
	q.Add("q", searchParams.Query)
	q.Add("count", fmt.Sprintf("%d", searchParams.Count))
	if searchParams.Offset > 0 {
		q.Add("offset", fmt.Sprintf("%d", searchParams.Offset))
	}
	u.RawQuery = q.Encode()

	// Create a new request
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u.String(), nil)
	if err != nil {
		return "", err
	}

	// Add necessary headers
	req.Header.Add("Accept", "application/json")
	req.Header.Add("X-Subscription-Token", b.apiKey)

	// Execute the request
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	// Check for errors
	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("API returned error: %s (status code: %d)", string(bodyBytes), resp.StatusCode)
	}

	buf := &bytes.Buffer{}
	if _, err := io.Copy(buf, resp.Body); err != nil {
		return "", err
	}

	re := gjson.ParseBytes(buf.Bytes())

	// Format the results
	var resultBuilder strings.Builder
	resultBuilder.WriteString(fmt.Sprintf("Search results for '%s':\n\n", searchParams.Query))

	for i, result := range re.Get("web.results").Array() {
		resultBuilder.WriteString(fmt.Sprintf("%d. %s\n", i+1, result.Get("title").String()))
		resultBuilder.WriteString(fmt.Sprintf("   URL: %s\n", result.Get("url").String()))
		resultBuilder.WriteString(fmt.Sprintf("   %s\n\n", result.Get("description").String()))
	}

	return resultBuilder.String(), nil
}
