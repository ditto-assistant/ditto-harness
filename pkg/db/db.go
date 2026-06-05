package db

import internaldb "github.com/ditto-assistant/ditto-harness/internal/db"

type DBTX = internaldb.DBTX
type Queries = internaldb.Queries

func New(conn DBTX) *Queries {
	return internaldb.New(conn)
}
