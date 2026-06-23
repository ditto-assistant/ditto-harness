// SPDX-License-Identifier: AGPL-3.0-or-later
//! Integration coverage for the versioned migration runner against a real
//! on-disk database (the in-crate unit tests only use `:memory:`). Exercises
//! the realistic open -> close -> reopen path, where a fresh connection must
//! observe the persisted `schema_migrations` ledger and skip re-applying.

use ditto_harness::db::Db;

/// Reads the single-column `version` set from the persisted ledger.
async fn ledger_versions(db: &Db) -> Vec<i64> {
    let mut rows = db
        .connection()
        .query("SELECT version FROM schema_migrations ORDER BY version", ())
        .await
        .expect("query schema_migrations");
    let mut out = Vec::new();
    while let Some(row) = rows.next().await.expect("next ledger row") {
        match row.get_value(0).expect("version") {
            turso::Value::Integer(v) => out.push(v),
            other => panic!("expected integer version, got {other:?}"),
        }
    }
    out
}

async fn ledger_count(db: &Db) -> i64 {
    let mut rows = db
        .connection()
        .query("SELECT COUNT(*) FROM schema_migrations", ())
        .await
        .expect("count schema_migrations");
    let row = rows.next().await.expect("next").expect("count row");
    match row.get_value(0).expect("count") {
        turso::Value::Integer(v) => v,
        other => panic!("expected integer count, got {other:?}"),
    }
}

#[tokio::test]
async fn migrations_persist_and_reopen_is_a_noop() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("harness.db");
    let path = path.to_str().expect("utf8 path");

    // First open applies every bundled migration and records the ledger.
    let versions = {
        let db = Db::open(path).await.expect("first open");
        let versions = ledger_versions(&db).await;
        assert!(
            !versions.is_empty(),
            "at least the baseline must be applied"
        );
        // The baseline schema must be usable (FK + defaults wired up).
        db.upsert_user("u1").await.expect("insert user");
        versions
    };

    // Reopen a fresh connection on the same file: migrate() runs again but must
    // skip everything via the persisted ledger — no duplicate rows, same set.
    let db = Db::open(path).await.expect("reopen");
    assert_eq!(
        ledger_versions(&db).await,
        versions,
        "reopen must not change the applied set"
    );
    assert_eq!(
        ledger_count(&db).await,
        versions.len() as i64,
        "reopen must not insert duplicate ledger rows"
    );

    // Data written before the reopen must still be there (schema preserved).
    let mut rows = db
        .connection()
        .query("SELECT COUNT(*) FROM harness_users WHERE uid = 'u1'", ())
        .await
        .expect("count user");
    let row = rows.next().await.expect("next").expect("row");
    assert_eq!(
        row.get_value(0).expect("count"),
        turso::Value::Integer(1),
        "user inserted before reopen must survive"
    );
}
