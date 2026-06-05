-- name: UpsertUser :exec
INSERT INTO harness_users (uid) VALUES ($1)
ON CONFLICT (uid) DO NOTHING;

-- name: CreateMemoryPair :one
INSERT INTO memory_pairs (
    firestore_pair_id, user_id, kg_id, session_id, title, description,
    prompt, response, input, output, source, source_context, timestamp,
    timezone_offset, seed_memories, retrieval_metadata, conversation_embedding
) VALUES (
    $1, $2, $3, NULLIF($4, ''), NULLIF($5, ''), NULLIF($6, ''),
    NULLIF($7, ''), NULLIF($8, ''), $9, $10, NULLIF($11, ''), NULLIF($12, ''),
    $13, $14, $15, $16, $17
)
ON CONFLICT (user_id, firestore_pair_id) DO UPDATE SET
    kg_id = EXCLUDED.kg_id,
    session_id = EXCLUDED.session_id,
    title = EXCLUDED.title,
    description = EXCLUDED.description,
    prompt = EXCLUDED.prompt,
    response = EXCLUDED.response,
    input = EXCLUDED.input,
    output = EXCLUDED.output,
    source = EXCLUDED.source,
    source_context = EXCLUDED.source_context,
    timestamp = EXCLUDED.timestamp,
    timezone_offset = EXCLUDED.timezone_offset,
    seed_memories = EXCLUDED.seed_memories,
    retrieval_metadata = EXCLUDED.retrieval_metadata,
    conversation_embedding = EXCLUDED.conversation_embedding,
    updated_at = NOW()
RETURNING id, firestore_pair_id, user_id, kg_id, session_id, title, description,
    prompt, response, input, output, source, source_context, timestamp,
    timezone_offset, seed_memories, retrieval_metadata, conversation_embedding;

-- name: UpsertSubject :one
INSERT INTO subjects (user_id, kg_id, subject_text, description_text, is_key_subject, embedding)
VALUES ($1, $2, $3, NULLIF($4, ''), $5, $6)
ON CONFLICT (user_id, kg_id, subject_text) DO UPDATE SET
    description_text = COALESCE(EXCLUDED.description_text, subjects.description_text),
    is_key_subject = subjects.is_key_subject OR EXCLUDED.is_key_subject,
    embedding = COALESCE(EXCLUDED.embedding, subjects.embedding),
    updated_at = NOW()
RETURNING id, user_id, kg_id, subject_text, description_text, is_key_subject, embedding;

-- name: LinkSubjectMemoryPair :exec
INSERT INTO subject_memory_pair_links (subject_id, pair_id, user_id, kg_id)
VALUES ($1, $2, $3, $4)
ON CONFLICT (subject_id, pair_id) DO NOTHING;

-- name: FetchMemories :many
SELECT id, firestore_pair_id, user_id, kg_id, session_id, title, description,
    prompt, response, input, output, source, source_context, timestamp,
    timezone_offset, seed_memories, retrieval_metadata, conversation_embedding
FROM memory_pairs
WHERE user_id = $1 AND firestore_pair_id = ANY($2::text[])
ORDER BY array_position($2::text[], firestore_pair_id);

-- name: SearchMemories :many
SELECT id, firestore_pair_id, user_id, kg_id, session_id, title, description,
    prompt, response, input, output, source, source_context, timestamp,
    timezone_offset, seed_memories, retrieval_metadata, conversation_embedding,
    (1 - (conversation_embedding <=> $1::vector))::float8 AS similarity
FROM memory_pairs
WHERE user_id = $2 AND kg_id = $3
  AND conversation_embedding IS NOT NULL
  AND ($4::text = '' OR COALESCE(session_id, 'main') = COALESCE(NULLIF($4::text, ''), 'main'))
  AND ($5::text[] IS NULL OR firestore_pair_id != ALL($5::text[]))
  AND (1 - (conversation_embedding <=> $1::vector)) >= $6
ORDER BY conversation_embedding <=> $1::vector, timestamp DESC
LIMIT $7;

-- name: SearchSubjects :many
SELECT id, user_id, kg_id, subject_text, description_text, is_key_subject, embedding,
    (1 - (embedding <=> $1::vector))::float8 AS similarity,
    COUNT(smpl.pair_id)::bigint AS memory_count
FROM subjects s
LEFT JOIN subject_memory_pair_links smpl ON smpl.subject_id = s.id
WHERE s.user_id = $2 AND s.kg_id = $3
  AND s.embedding IS NOT NULL
  AND (1 - (s.embedding <=> $1::vector)) >= $4
GROUP BY s.id
ORDER BY s.embedding <=> $1::vector, memory_count DESC, s.updated_at DESC
LIMIT $5;

-- name: SearchMemoriesBySubject :many
SELECT mp.id, mp.firestore_pair_id, mp.user_id, mp.kg_id, mp.session_id, mp.title, mp.description,
    mp.prompt, mp.response, mp.input, mp.output, mp.source, mp.source_context, mp.timestamp,
    mp.timezone_offset, mp.seed_memories, mp.retrieval_metadata, mp.conversation_embedding,
    (1 - (mp.conversation_embedding <=> $1::vector))::float8 AS similarity
FROM memory_pairs mp
JOIN subject_memory_pair_links smpl ON smpl.pair_id = mp.id
WHERE smpl.subject_id = $2 AND mp.user_id = $3
  AND mp.conversation_embedding IS NOT NULL
  AND (1 - (mp.conversation_embedding <=> $1::vector)) >= $4
ORDER BY mp.conversation_embedding <=> $1::vector, mp.timestamp DESC
LIMIT $5;
