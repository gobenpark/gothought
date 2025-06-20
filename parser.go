package gothought

import (
	"encoding/json"
	"errors"
	"reflect"
	"strings"
)

type JSONSchema struct {
	Type        string                `json:"type,omitempty"`
	Description string                `json:"description,omitempty"`
	Required    []string              `json:"required,omitempty"`
	Properties  map[string]JSONSchema `json:"properties,omitempty"`
}

func GenerateJSONSchema(v interface{}) JSONSchema {
	t := reflect.TypeOf(v)

	if t.Kind() == reflect.Ptr {
		t = t.Elem()
	}

	schema := JSONSchema{
		Type:       "object",
		Properties: make(map[string]JSONSchema),
	}

	for i := 0; i < t.NumField(); i++ {
		field := t.Field(i)

		jsonTag := field.Tag.Get("json")
		descriptionTag := field.Tag.Get("description")

		if jsonTag == "" || jsonTag == "-" {
			continue
		}

		jsonName := strings.Split(jsonTag, ",")[0]

		var propertyType string
		switch field.Type.Kind() {
		case reflect.String:
			propertyType = "string"
		case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
			reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
			propertyType = "integer"
		case reflect.Float32, reflect.Float64:
			propertyType = "number"
		case reflect.Bool:
			propertyType = "boolean"
		case reflect.Struct:
			nestedSchema := GenerateJSONSchema(reflect.New(field.Type).Elem().Interface())
			schema.Properties[jsonName] = nestedSchema
			continue
		case reflect.Slice, reflect.Array:
			propertyType = "array"
		}

		schema.Properties[jsonName] = JSONSchema{
			Type:        propertyType,
			Description: descriptionTag,
		}
	}
	return schema
}

func GenerateSchemaPrompt(v interface{}) string {
	var st strings.Builder

	st.WriteString("The output must be provided as a markdown code snippet, starting with ```json and ending with ```. Please generate JSON content that conforms to the JSON schema defined above:\n\n")

	result := GenerateJSONSchema(v)
	bt, err := json.MarshalIndent(&result, "", "\t")
	if err != nil {
		return ""
	}
	st.WriteString(string(bt))
	return st.String()
}

func ParsePrompt(v interface{}, text string) error {
	_, halfJson, ok := strings.Cut(text, "```json")
	if !ok {
		return errors.New("no ```json at start of output")
	}

	jsonString, _, ok := strings.Cut(halfJson, "```")
	if !ok {
		return errors.New("no ```json at end of output")
	}

	return json.Unmarshal([]byte(jsonString), v)
}
