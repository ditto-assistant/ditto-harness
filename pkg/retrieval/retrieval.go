// SPDX-License-Identifier: AGPL-3.0-or-later
package retrieval

import (
	"context"
	"encoding/json"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/pgvector/pgvector-go"
)

type DBTX interface {
	Query(context.Context, string, ...any) (pgx.Rows, error)
	Exec(context.Context, string, ...any) (pgconn.CommandTag, error)
}

type Variant string

const (
	VariantLegacy Variant = "v1"
	VariantV2     Variant = "v2"
)

const (
	V2WeightCosine = iota
	V2WeightRecencyLinear
	V2WeightRecencyExp
	V2WeightSubjectFrequency
	V2WeightSubjectSemMatch
	V2WeightSessionContinuity
	V2WeightNeighborDensity
	V2NumWeights
)

type CompositeMemory struct {
	PairID            string  `json:"pairId"`
	CosineSimilarity  float64 `json:"cosineSimilarity"`
	RecencyScore      float64 `json:"recencyScore"`
	FrequencyScore    float64 `json:"frequencyScore"`
	CompositeScore    float64 `json:"compositeScore"`
	RecencyExp        float64 `json:"recencyExp,omitempty"`
	SubjectSemMatch   float64 `json:"subjectSemMatch,omitempty"`
	SessionContinuity float64 `json:"sessionContinuity,omitempty"`
	NeighborDensity   float64 `json:"neighborDensity,omitempty"`
}

func (m CompositeMemory) RetrievalMetadata(weights Weights, variant Variant, ids []string) map[string]any {
	return map[string]any{
		"variant":          variant,
		"weights":          weights,
		"retrievedPairIds": ids,
	}
}

type CompositeParams struct {
	Embedding         []float32
	UserID            string
	KGID              string
	SessionID         string
	CurrentSessionID  string
	MinTimestamp      time.Time
	Limit             int
	CandidatePoolSize int
	ExcludePairIDs    []string
	Weights           Weights
	Variant           Variant
	RequestPath       string
	Query             string
	LogEvent          bool
}

type Weights struct {
	Cosine            float64 `json:"cosine"`
	RecencyLinear     float64 `json:"recencyLinear"`
	RecencyExp        float64 `json:"recencyExp,omitempty"`
	SubjectFrequency  float64 `json:"subjectFrequency"`
	SubjectSemMatch   float64 `json:"subjectSemMatch,omitempty"`
	SessionContinuity float64 `json:"sessionContinuity,omitempty"`
	NeighborDensity   float64 `json:"neighborDensity,omitempty"`
	Scale             float64 `json:"scale,omitempty"`
}

func DefaultWeights() Weights {
	return Weights{
		Cosine:           0.65,
		RecencyLinear:    0.20,
		SubjectFrequency: 0.15,
		Scale:            1,
	}
}

type WeightPredictor interface {
	Predict(ctx context.Context, features Features) (Weights, error)
}

type Features struct {
	Query                 string    `json:"query,omitempty"`
	Now                   time.Time `json:"now,omitempty"`
	QueryEmbedding        []float32 `json:"-"`
	ShortTermMemoryCount  int       `json:"shortTermMemoryCount,omitempty"`
	CandidateMemoryCount  int       `json:"candidateMemoryCount,omitempty"`
	CurrentSessionID      string    `json:"currentSessionId,omitempty"`
	EmbeddingNorm         float64   `json:"embeddingNorm,omitempty"`
	HostApplicationSignal string    `json:"hostApplicationSignal,omitempty"`
}

type StaticPredictor struct {
	Weights Weights
}

func (p StaticPredictor) Predict(context.Context, Features) (Weights, error) {
	if p.Weights == (Weights{}) {
		return DefaultWeights(), nil
	}
	return p.Weights, nil
}

func CompositeRetrieve(ctx context.Context, db DBTX, p CompositeParams) ([]CompositeMemory, error) {
	if p.Limit <= 0 {
		p.Limit = 8
	}
	if p.CandidatePoolSize <= 0 {
		p.CandidatePoolSize = max(32, p.Limit*4)
	}
	if p.MinTimestamp.IsZero() {
		p.MinTimestamp = time.Unix(0, 0).UTC()
	}
	if p.Weights == (Weights{}) {
		p.Weights = DefaultWeights()
	}
	if p.Weights.Scale == 0 {
		p.Weights.Scale = 1
	}
	var (
		results []CompositeMemory
		err     error
	)
	if p.Variant == VariantV2 {
		results, err = compositeRetrieveV2(ctx, db, p)
	} else {
		results, err = compositeRetrieveV1(ctx, db, p)
	}
	if err != nil {
		return nil, err
	}
	if p.LogEvent {
		_ = LogEvent(ctx, db, p.UserID, p.KGID, p.SessionID, p.RequestPath, p.Query, p.Embedding, pairIDs(results), p.Weights, Features{
			Query:            p.Query,
			Now:              time.Now().UTC(),
			CurrentSessionID: p.CurrentSessionID,
		})
	}
	return results, nil
}

func compositeRetrieveV1(ctx context.Context, db DBTX, p CompositeParams) ([]CompositeMemory, error) {
	rows, err := db.Query(ctx, compositeSQLV1,
		pgvector.NewVector(p.Embedding), p.UserID, p.KGID, p.CandidatePoolSize,
		p.Weights.Cosine, p.Weights.RecencyLinear, p.Weights.SubjectFrequency,
		p.Limit, nilIfEmpty(p.ExcludePairIDs), p.SessionID, p.MinTimestamp,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []CompositeMemory
	for rows.Next() {
		var item CompositeMemory
		if err := rows.Scan(&item.PairID, &item.CosineSimilarity, &item.RecencyScore, &item.FrequencyScore, &item.CompositeScore); err != nil {
			return nil, err
		}
		out = append(out, item)
	}
	return out, rows.Err()
}

func compositeRetrieveV2(ctx context.Context, db DBTX, p CompositeParams) ([]CompositeMemory, error) {
	rows, err := db.Query(ctx, compositeSQLV2,
		pgvector.NewVector(p.Embedding), p.UserID, p.KGID, p.CandidatePoolSize,
		p.Weights.Cosine, p.Weights.RecencyLinear, p.Weights.SubjectFrequency,
		p.Limit, nilIfEmpty(p.ExcludePairIDs), p.SessionID, p.MinTimestamp,
		p.Weights.RecencyExp, p.Weights.SubjectSemMatch, p.Weights.SessionContinuity,
		p.Weights.NeighborDensity, 14*24*3600.0, p.CurrentSessionID, p.Weights.Scale,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []CompositeMemory
	for rows.Next() {
		var item CompositeMemory
		if err := rows.Scan(
			&item.PairID, &item.CosineSimilarity, &item.RecencyScore, &item.FrequencyScore,
			&item.RecencyExp, &item.SubjectSemMatch, &item.SessionContinuity,
			&item.NeighborDensity, &item.CompositeScore,
		); err != nil {
			return nil, err
		}
		out = append(out, item)
	}
	return out, rows.Err()
}

func LogEvent(ctx context.Context, db DBTX, userID, kgID, sessionID, requestPath, query string, embedding []float32, ids []string, weights Weights, aux Features) error {
	weightsJSON, err := json.Marshal(weights)
	if err != nil {
		return err
	}
	auxJSON, err := json.Marshal(aux)
	if err != nil {
		return err
	}
	_, err = db.Exec(ctx, `
INSERT INTO retrieval_events (user_id, kg_id, session_id, request_path, query, query_embedding, retrieved_pair_ids, weights, aux_features)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)`,
		userID, kgID, sessionID, requestPath, query, vectorOrNil(embedding), ids, weightsJSON, auxJSON)
	return err
}

const compositeSQLV1 = `
WITH candidates AS (
    SELECT mp.id, mp.firestore_pair_id, mp.timestamp,
           (1 - (mp.conversation_embedding <=> $1::vector)) AS cosine_sim
    FROM memory_pairs mp
    WHERE mp.user_id = $2 AND mp.kg_id = $3
      AND ($10::text = '' OR COALESCE(mp.session_id, 'main') = COALESCE(NULLIF($10::text, ''), 'main'))
      AND mp.timestamp >= $11
      AND mp.conversation_embedding IS NOT NULL
      AND ($9::text[] IS NULL OR mp.firestore_pair_id != ALL($9::text[]))
    ORDER BY mp.conversation_embedding <=> $1::vector
    LIMIT $4
),
pair_freq AS (
    SELECT smpl.pair_id, SUM(spc.link_count) AS total_frequency
    FROM subject_memory_pair_links smpl
    JOIN (
        SELECT subject_id, COUNT(*) AS link_count
        FROM subject_memory_pair_links GROUP BY subject_id
    ) spc ON smpl.subject_id = spc.subject_id
    WHERE smpl.pair_id IN (SELECT id FROM candidates)
    GROUP BY smpl.pair_id
),
bounds AS (
    SELECT MIN(timestamp) AS oldest, MAX(timestamp) AS newest FROM candidates
),
max_freq AS (
    SELECT COALESCE(MAX(total_frequency), 1) AS val FROM pair_freq
)
SELECT c.firestore_pair_id, c.cosine_sim,
    CASE WHEN b.newest = b.oldest THEN 1.0
         ELSE EXTRACT(EPOCH FROM (c.timestamp - b.oldest)) / EXTRACT(EPOCH FROM (b.newest - b.oldest))
    END AS recency_score,
    COALESCE(pf.total_frequency::float / mf.val, 0.0) AS frequency_score,
    ($5 * c.cosine_sim
   + $6 * CASE WHEN b.newest = b.oldest THEN 1.0
               ELSE EXTRACT(EPOCH FROM (c.timestamp - b.oldest)) / EXTRACT(EPOCH FROM (b.newest - b.oldest)) END
   + $7 * COALESCE(pf.total_frequency::float / mf.val, 0.0)) AS composite_score
FROM candidates c
CROSS JOIN bounds b CROSS JOIN max_freq mf
LEFT JOIN pair_freq pf ON c.id = pf.pair_id
ORDER BY composite_score DESC, c.timestamp DESC, c.firestore_pair_id DESC
LIMIT $8`

const compositeSQLV2 = `
WITH candidates AS (
    SELECT mp.id, mp.firestore_pair_id, mp.timestamp, mp.session_id,
           (1 - (mp.conversation_embedding <=> $1::vector)) AS cosine_sim,
           EXP(-EXTRACT(EPOCH FROM (NOW() - mp.timestamp)) / NULLIF($16, 0)) AS recency_exp
    FROM memory_pairs mp
    WHERE mp.user_id = $2 AND mp.kg_id = $3
      AND ($10::text = '' OR COALESCE(mp.session_id, 'main') = COALESCE(NULLIF($10::text, ''), 'main'))
      AND mp.timestamp >= $11
      AND mp.conversation_embedding IS NOT NULL
      AND ($9::text[] IS NULL OR mp.firestore_pair_id != ALL($9::text[]))
    ORDER BY mp.conversation_embedding <=> $1::vector
    LIMIT $4
),
pair_freq AS (
    SELECT smpl.pair_id, SUM(spc.link_count) AS total_frequency
    FROM subject_memory_pair_links smpl
    JOIN (
        SELECT subject_id, COUNT(*) AS link_count
        FROM subject_memory_pair_links GROUP BY subject_id
    ) spc ON smpl.subject_id = spc.subject_id
    WHERE smpl.pair_id IN (SELECT id FROM candidates)
    GROUP BY smpl.pair_id
),
subject_match AS (
    SELECT smpl.pair_id, MAX(1 - (s.embedding <=> $1::vector)) AS subj_sem
    FROM subject_memory_pair_links smpl
    JOIN subjects s ON s.id = smpl.subject_id
    WHERE smpl.pair_id IN (SELECT id FROM candidates)
      AND s.embedding IS NOT NULL
    GROUP BY smpl.pair_id
),
candidate_subjects AS (
    SELECT smpl.pair_id, smpl.subject_id
    FROM subject_memory_pair_links smpl
    WHERE smpl.pair_id IN (SELECT id FROM candidates)
),
neighbor_density AS (
    SELECT a.pair_id, COUNT(DISTINCT b.pair_id) AS density
    FROM candidate_subjects a
    JOIN candidate_subjects b ON a.subject_id = b.subject_id AND a.pair_id <> b.pair_id
    GROUP BY a.pair_id
),
density_max AS (
    SELECT COALESCE(MAX(density), 1) AS val FROM neighbor_density
),
bounds AS (
    SELECT MIN(timestamp) AS oldest, MAX(timestamp) AS newest FROM candidates
),
max_freq AS (
    SELECT COALESCE(MAX(total_frequency), 1) AS val FROM pair_freq
)
SELECT c.firestore_pair_id, c.cosine_sim,
    CASE WHEN b.newest = b.oldest THEN 1.0
         ELSE EXTRACT(EPOCH FROM (c.timestamp - b.oldest)) / EXTRACT(EPOCH FROM (b.newest - b.oldest))
    END AS recency_score,
    COALESCE(pf.total_frequency::float / mf.val, 0.0) AS frequency_score,
    COALESCE(c.recency_exp, 0.0) AS recency_exp,
    COALESCE(sm.subj_sem, 0.0) AS subj_sem,
    CASE WHEN COALESCE($17::text, '') = '' THEN 0.0
         WHEN COALESCE(c.session_id, '') = $17::text THEN 1.0
         ELSE 0.0 END AS session_continuity,
    COALESCE(nd.density::float / NULLIF(dm.val, 0), 0.0) AS neighbor_density,
    $18 * (
        $5 * c.cosine_sim
      + $6 * CASE WHEN b.newest = b.oldest THEN 1.0
                  ELSE EXTRACT(EPOCH FROM (c.timestamp - b.oldest)) / EXTRACT(EPOCH FROM (b.newest - b.oldest)) END
      + $7 * COALESCE(pf.total_frequency::float / mf.val, 0.0)
      + $12 * COALESCE(c.recency_exp, 0.0)
      + $13 * COALESCE(sm.subj_sem, 0.0)
      + $14 * CASE WHEN COALESCE($17::text, '') = '' THEN 0.0
                   WHEN COALESCE(c.session_id, '') = $17::text THEN 1.0 ELSE 0.0 END
      + $15 * COALESCE(nd.density::float / NULLIF(dm.val, 0), 0.0)
    ) AS composite_score
FROM candidates c
CROSS JOIN bounds b CROSS JOIN max_freq mf CROSS JOIN density_max dm
LEFT JOIN pair_freq pf ON c.id = pf.pair_id
LEFT JOIN subject_match sm ON c.id = sm.pair_id
LEFT JOIN neighbor_density nd ON c.id = nd.pair_id
ORDER BY composite_score DESC, c.timestamp DESC, c.firestore_pair_id DESC
LIMIT $8`

func nilIfEmpty(v []string) any {
	if len(v) == 0 {
		return nil
	}
	return v
}

func vectorOrNil(v []float32) any {
	if len(v) == 0 {
		return nil
	}
	return pgvector.NewVector(v)
}

func pairIDs(results []CompositeMemory) []string {
	out := make([]string, 0, len(results))
	for _, result := range results {
		if result.PairID != "" {
			out = append(out, result.PairID)
		}
	}
	return out
}
