package messages

const (
	FinishReasonStop      = "stop"
	FinishReasonToolCalls = "tool_calls"
)

type Message struct {
	Role       string `json:"role"`
	ToolCallID string `json:"tool_call_id"`
	Message    string
	ToolCalls  []ToolCalls `json:"tool_calls"`
}

type ResponseMessage struct {
	Message string
}

type ToolCalls struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}
