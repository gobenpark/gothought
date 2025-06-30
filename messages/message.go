package messages

const (
	FinishReasonStop      = "stop"
	FinishReasonToolCalls = "tool_calls"
)

type Role string

const (
	RoleAssistant Role = "assistant"
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleTool      Role = "tool"
)

func (r Role) String() string {
	return string(r)
}

type Message struct {
	Role       Role   `json:"role"`
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
