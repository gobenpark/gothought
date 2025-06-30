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

// WikiParams defines the parameters for a Wikipedia search query
type WikiParams struct {
	Query  string `json:"query"`
	Count  int    `json:"count"`
	Offset int    `json:"offset"`
}

// WikipediaTool implements the Tool interface for searching with Brave
type WikipediaTool struct {
	topK     int
	language string
}

// NewWikipediaTool creates a new instance of WikipediaTool
func NewWikipediaTool(topK int, language string) *WikipediaTool {
	return &WikipediaTool{topK: topK, language: language}
}

// ParameterSchema function the parameters structure for a Wiki search query
func (b *WikipediaTool) ParameterSchema() map[string]interface{} {
	return map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"query": map[string]interface{}{
				"type":        "string",
				"description": "The user’s search query term. Query can not be empty. Maximum of 400 characters and 50 words in the query.",
			},
		},
		"required": []string{"query"},
	}
}

// Description returns a description of the tool
func (b *WikipediaTool) Description() string {
	return `Searches Wikipedia for information, ideal for factual queries, encyclopedic knowledge, and detailed explanations.
Use this for accessing structured knowledge articles, historical information, or when you need reliable reference material.
Provides access to Wikipedia's extensive collection of articles across various subjects and languages.
Results include article summaries, links to related topics, and citations when available.
Maximum 15 results per request, with pagination options for browsing additional content.`
}

// Name returns the name of the tool
func (b *WikipediaTool) Name() string {
	return "wikipedia_search"
}

// Call executes a search query against the Brave API
func (b *WikipediaTool) Call(ctx context.Context, params string) (string, error) {
	// Parse the input parameters
	var searchParams WikiParams
	if err := json.Unmarshal([]byte(params), &searchParams); err != nil {
		return "", fmt.Errorf("invalid search parameters: %v", err)
	}

	// Validate parameters
	if searchParams.Query == "" {
		return "", errors.New("query parameter is required")
	}

	// Set default values if not provided
	if searchParams.Count <= 0 || searchParams.Count > 15 {
		searchParams.Count = 10 // Default to 10 results
	}

	// Build the request URL
	baseURL := fmt.Sprintf("https://%s.wikipedia.org/w/api.php", b.language)
	u, err := url.Parse(baseURL)
	if err != nil {
		return "", err
	}

	// Add query parameters for Wikipedia API
	q := u.Query()
	q.Add("action", "query")
	q.Add("format", "json")
	q.Add("generator", "search")
	q.Add("gsrsearch", searchParams.Query)
	q.Add("gsrlimit", fmt.Sprintf("%d", searchParams.Count))
	if searchParams.Offset > 0 {
		q.Add("gsroffset", fmt.Sprintf("%d", searchParams.Offset))
	}
	q.Add("prop", "extracts|info")
	q.Add("exintro", "1")
	q.Add("explaintext", "1")
	q.Add("inprop", "url")
	q.Add("exsentences", "3")
	u.RawQuery = q.Encode()

	// Create a new request
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u.String(), nil)
	if err != nil {
		return "", err
	}

	// Add necessary headers
	req.Header.Add("Accept", "application/json")
	req.Header.Add("User-Agent", "WikipediaTool/1.0")

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
	resultBuilder.WriteString(fmt.Sprintf("Wikipedia results for '%s':\n\n", searchParams.Query))

	// Wikipedia API returns results in pages object
	pages := re.Get("query.pages").Map()
	if len(pages) == 0 {
		return "No results found.", nil
	}

	// Process the results
	pageIDs := make([]string, 0, len(pages))
	for pageID := range pages {
		pageIDs = append(pageIDs, pageID)
	}

	for i, pageID := range pageIDs {
		result := pages[pageID]
		title := result.Get("title").String()
		extract := result.Get("extract").String()
		pageURL := result.Get("fullurl").String()

		resultBuilder.WriteString(fmt.Sprintf("%d. %s\n", i+1, title))
		resultBuilder.WriteString(fmt.Sprintf("   URL: %s\n", pageURL))
		resultBuilder.WriteString(fmt.Sprintf("   %s\n\n", extract))
	}

	return resultBuilder.String(), nil
}
