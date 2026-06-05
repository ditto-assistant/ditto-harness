package db

func (q *Queries) DB() DBTX {
	return q.db
}
