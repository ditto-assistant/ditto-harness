package mcpserver_test

import (
	"fmt"

	"github.com/ditto-assistant/ditto-harness/pkg/db"
	"github.com/ditto-assistant/ditto-harness/pkg/mcpserver"
	"github.com/ditto-assistant/ditto-harness/pkg/memory"
	"github.com/jackc/pgx/v5/pgxpool"
)

func ExampleNew() {
	var pool *pgxpool.Pool
	store := memory.NewStore(memory.Options{Queries: db.New(pool)})
	server := mcpserver.New(mcpserver.Options{
		Store:  store,
		UserID: "user_123",
	})
	fmt.Println(server != nil)
	// Output:
	// true
}
