package mcpserver

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/ditto-assistant/ditto-harness/pkg/harness"
	"github.com/ditto-assistant/ditto-harness/pkg/memory"
	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"
)

const (
	ServerName    = "ditto-harness-memory"
	ServerVersion = "0.1.0"
)

type Server struct {
	*server.MCPServer
	store  *memory.Store
	userID string
	kgID   string
}

type Options struct {
	Store  *memory.Store
	UserID string
	KGID   string
}

func New(opts Options) *Server {
	s := &Server{
		MCPServer: server.NewMCPServer(
			ServerName,
			ServerVersion,
			server.WithToolCapabilities(true),
		),
		store:  opts.Store,
		userID: opts.UserID,
		kgID:   opts.KGID,
	}
	s.registerTools()
	return s
}

func (s *Server) SetUser(userID, kgID string) {
	s.userID = userID
	s.kgID = kgID
}

func (s *Server) registerTools() {
	s.AddTool(mcp.NewTool("save_memory",
		mcp.WithDescription("Save a memory for the authenticated user."),
		mcp.WithString("prompt", mcp.Description("User-side memory text.")),
		mcp.WithString("response", mcp.Description("Assistant-side memory text.")),
		mcp.WithString("summary", mcp.Description("Compact memory summary.")),
		mcp.WithString("sessionId", mcp.Description("Optional session/thread id.")),
	), s.handleSaveMemory)

	s.AddTool(mcp.NewTool("search_memories",
		mcp.WithDescription("Search past memories by semantic similarity."),
		mcp.WithArray("queries", mcp.Required(), mcp.Items(map[string]any{"type": "string"})),
	), s.handleSearchMemories)

	s.AddTool(mcp.NewTool("search_subjects",
		mcp.WithDescription("Search the subject graph and return subject ids."),
		mcp.WithArray("queries", mcp.Required(), mcp.Items(map[string]any{"type": "string"})),
	), s.handleSearchSubjects)

	s.AddTool(mcp.NewTool("search_memories_in_subjects",
		mcp.WithDescription("Search memories linked to a subject."),
		mcp.WithString("subject_id", mcp.Required()),
		mcp.WithArray("queries", mcp.Required(), mcp.Items(map[string]any{"type": "string"})),
	), s.handleSearchMemoriesInSubjects)

	s.AddTool(mcp.NewTool("fetch_memories",
		mcp.WithDescription("Fetch full memories by pair id."),
		mcp.WithArray("pairIds", mcp.Required(), mcp.Items(map[string]any{"type": "string"})),
	), s.handleFetchMemories)
}

func (s *Server) handleSaveMemory(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	if err := s.ready(); err != nil {
		return mcp.NewToolResultError(err.Error()), nil
	}
	mem, err := s.store.SaveMemory(ctx, memory.SaveMemoryRequest{
		UserID:    s.userID,
		KGID:      s.kgID,
		SessionID: request.GetString("sessionId", ""),
		Prompt:    request.GetString("prompt", ""),
		Response:  request.GetString("response", ""),
		Summary:   request.GetString("summary", ""),
		Source:    "mcp",
	})
	if err != nil {
		return mcp.NewToolResultError(err.Error()), nil
	}
	return toolJSON(map[string]any{"memory": mem})
}

func (s *Server) handleSearchMemories(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	if err := s.ready(); err != nil {
		return mcp.NewToolResultError(err.Error()), nil
	}
	memories, err := s.store.SearchMemories(ctx, memory.SearchMemoriesRequest{
		UserID:  s.userID,
		KGID:    s.kgID,
		Queries: request.GetStringSlice("queries", nil),
		Limit:   request.GetInt("topK", 8),
	})
	if err != nil {
		return mcp.NewToolResultError(err.Error()), nil
	}
	return toolJSON(map[string]any{"memories": memory.SlimPreviews(memories, memory.DefaultPreviewLen)})
}

func (s *Server) handleSearchSubjects(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	if err := s.ready(); err != nil {
		return mcp.NewToolResultError(err.Error()), nil
	}
	subjects, err := s.store.SearchSubjects(ctx, memory.SearchSubjectsRequest{
		UserID:  s.userID,
		KGID:    s.kgID,
		Queries: request.GetStringSlice("queries", nil),
		Limit:   request.GetInt("topK", 8),
	})
	if err != nil {
		return mcp.NewToolResultError(err.Error()), nil
	}
	return toolJSON(map[string]any{"subjects": subjects})
}

func (s *Server) handleSearchMemoriesInSubjects(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	if err := s.ready(); err != nil {
		return mcp.NewToolResultError(err.Error()), nil
	}
	subjectID := request.GetString("subject_id", "")
	queries := request.GetStringSlice("queries", nil)
	typed := make([]memory.SubjectMemoryQuery, len(queries))
	for i, query := range queries {
		typed[i] = memory.SubjectMemoryQuery{SubjectID: subjectID, Query: query}
	}
	memories, err := s.store.SearchMemoriesInSubjects(ctx, memory.SearchMemoriesInSubjectsRequest{
		UserID:  s.userID,
		Queries: typed,
		Limit:   request.GetInt("topK", 8),
	})
	if err != nil {
		return mcp.NewToolResultError(err.Error()), nil
	}
	return toolJSON(map[string]any{"memories": memory.SlimPreviews(memories, memory.DefaultPreviewLen)})
}

func (s *Server) handleFetchMemories(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	if err := s.ready(); err != nil {
		return mcp.NewToolResultError(err.Error()), nil
	}
	memories, err := s.store.FetchMemories(ctx, memory.FetchMemoriesRequest{
		UserID:  s.userID,
		PairIDs: request.GetStringSlice("pairIds", nil),
	})
	if err != nil {
		return mcp.NewToolResultError(err.Error()), nil
	}
	return toolJSON(map[string]any{"memories": memory.SlimTruncated(memories, memory.DefaultFetchMaxBytes)})
}

func (s *Server) ready() error {
	if s.store == nil {
		return fmt.Errorf("memory store is not configured")
	}
	if s.userID == "" {
		return fmt.Errorf("not authenticated")
	}
	return nil
}

func toolJSON(v any) (*mcp.CallToolResult, error) {
	raw, err := json.Marshal(v)
	if err != nil {
		return nil, err
	}
	return mcp.NewToolResultText(string(raw)), nil
}

type MemoryTool struct {
	Name        string
	Description string
	Schema      json.RawMessage
	Handler     func(context.Context, json.RawMessage) (harness.ToolCallResponse, error)
}

func (t MemoryTool) Definition() harness.ToolDefinition {
	return harness.ToolDefinition{Name: t.Name, Description: t.Description, InputSchema: t.Schema}
}

func (t MemoryTool) Call(ctx context.Context, raw json.RawMessage) (harness.ToolCallResponse, error) {
	return t.Handler(ctx, raw)
}
