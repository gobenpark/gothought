package tools

import (
	"context"
	"fmt"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestWikipediaTool_Call(t *testing.T) {
	tool := NewWikipediaTool(3, "ko")
	result, err := tool.Call(context.TODO(), "{\"query\": \"경복궁\"}")
	require.NoError(t, err)
	fmt.Println(result)
}
