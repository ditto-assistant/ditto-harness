// SPDX-License-Identifier: AGPL-3.0-or-later
package db

func (q *Queries) DB() DBTX {
	return q.db
}
