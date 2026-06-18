// SPDX-License-Identifier: AGPL-3.0-or-later
package testpg

import (
	"context"
	"crypto/rand"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jackc/pgx/v5/pgxpool"
)

func NewPool(t *testing.T) *pgxpool.Pool {
	t.Helper()
	adminURL := os.Getenv("DITTO_HARNESS_TEST_DATABASE_URL")
	if adminURL == "" {
		t.Skip("DITTO_HARNESS_TEST_DATABASE_URL is not set")
	}
	ctx := context.Background()
	admin, err := pgxpool.New(ctx, adminURL)
	if err != nil {
		t.Fatalf("connect admin postgres: %v", err)
	}
	t.Cleanup(admin.Close)

	dbName := "ditto_harness_test_" + randSuffix(t)
	if _, err := admin.Exec(ctx, "CREATE DATABASE "+dbName); err != nil {
		t.Fatalf("create test database: %v", err)
	}
	t.Cleanup(func() {
		_, _ = admin.Exec(context.Background(), "DROP DATABASE IF EXISTS "+dbName+" WITH (FORCE)")
	})

	dbURL := replaceDBName(adminURL, dbName)
	pool, err := pgxpool.New(ctx, dbURL)
	if err != nil {
		t.Fatalf("connect test postgres: %v", err)
	}
	t.Cleanup(pool.Close)
	if err := ApplyMigrations(ctx, pool); err != nil {
		t.Fatalf("apply migrations: %v", err)
	}
	return pool
}

func ApplyMigrations(ctx context.Context, pool *pgxpool.Pool) error {
	migrationsDir, err := findMigrationsDir()
	if err != nil {
		return err
	}
	entries, err := os.ReadDir(migrationsDir)
	if err != nil {
		return err
	}
	for _, entry := range entries {
		name := entry.Name()
		if !strings.HasSuffix(name, ".up.sql") {
			continue
		}
		raw, err := os.ReadFile(filepath.Join(migrationsDir, name))
		if err != nil {
			return err
		}
		if _, err := pool.Exec(ctx, string(raw)); err != nil {
			return fmt.Errorf("%s: %w", name, err)
		}
	}
	return nil
}

func findMigrationsDir() (string, error) {
	wd, err := os.Getwd()
	if err != nil {
		return "", err
	}
	for {
		candidate := filepath.Join(wd, "db", "migrations")
		if info, err := os.Stat(candidate); err == nil && info.IsDir() {
			return candidate, nil
		}
		parent := filepath.Dir(wd)
		if parent == wd {
			return "", fmt.Errorf("db/migrations not found")
		}
		wd = parent
	}
}

func randSuffix(t *testing.T) string {
	t.Helper()
	var b [6]byte
	if _, err := rand.Read(b[:]); err != nil {
		t.Fatalf("random suffix: %v", err)
	}
	return fmt.Sprintf("%x", b[:])
}

func replaceDBName(url, dbName string) string {
	if idx := strings.LastIndex(url, "/"); idx >= 0 {
		if q := strings.Index(url[idx:], "?"); q >= 0 {
			return url[:idx+1] + dbName + url[idx+q:]
		}
		return url[:idx+1] + dbName
	}
	return url
}
