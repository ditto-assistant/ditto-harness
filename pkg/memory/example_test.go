// SPDX-License-Identifier: AGPL-3.0-or-later
package memory_test

import (
	"context"
	"fmt"

	"github.com/ditto-assistant/ditto-harness/pkg/db"
	"github.com/ditto-assistant/ditto-harness/pkg/harness"
	"github.com/ditto-assistant/ditto-harness/pkg/memory"
	"github.com/jackc/pgx/v5/pgxpool"
)

type exampleEmbedder struct{}

func (exampleEmbedder) Embed(context.Context, harness.EmbedRequest) (harness.EmbedResponse, error) {
	return harness.EmbedResponse{
		Embeddings: [][]float32{make([]float32, 768)},
		Cost: &harness.CostedUsage{
			Usage: harness.Usage{Provider: "host", Model: "embedding-model", TotalTokens: 8},
			Cost:  harness.Cost{Currency: "USD", Amount: 0.00001},
		},
	}, nil
}

func ExampleNewStore() {
	var pool *pgxpool.Pool
	store := memory.NewStore(memory.Options{
		Queries:  db.New(pool),
		Embedder: exampleEmbedder{},
	})
	tools := memory.Tools(memory.ToolOptions{
		Store:  store,
		UserID: "user_123",
	})
	fmt.Println(len(tools))
	// Output:
	// 5
}
