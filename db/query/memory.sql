-- name: UpsertUser :exec
INSERT INTO harness_users (uid) VALUES (sqlc.arg(uid))
ON CONFLICT (uid) DO NOTHING;

-- name: CreateMemoryPair :one
INSERT INTO memory_pairs (
    firestore_pair_id, user_id, kg_id, session_id, title, description,
    prompt, response, input, output, source, source_context, timestamp,
    timezone_offset, seed_memories, retrieval_metadata, conversation_embedding
) VALUES (
    sqlc.arg(firestore_pair_id), sqlc.arg(user_id), sqlc.arg(kg_id),
    NULLIF(sqlc.arg(session_id), ''), NULLIF(sqlc.arg(title), ''), NULLIF(sqlc.arg(description), ''),
    NULLIF(sqlc.arg(prompt), ''), NULLIF(sqlc.arg(response), ''), sqlc.arg(input), sqlc.arg(output),
    NULLIF(sqlc.arg(source), ''), NULLIF(sqlc.arg(source_context), ''),
    sqlc.arg(timestamp), sqlc.arg(timezone_offset), sqlc.arg(seed_memories),
    sqlc.arg(retrieval_metadata), sqlc.arg(conversation_embedding)
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
VALUES (
    sqlc.arg(user_id), sqlc.arg(kg_id), sqlc.arg(subject_text),
    NULLIF(sqlc.arg(description_text), ''), sqlc.arg(is_key_subject), sqlc.arg(embedding)
)
ON CONFLICT (user_id, kg_id, subject_text) DO UPDATE SET
    description_text = COALESCE(EXCLUDED.description_text, subjects.description_text),
    is_key_subject = subjects.is_key_subject OR EXCLUDED.is_key_subject,
    embedding = COALESCE(EXCLUDED.embedding, subjects.embedding),
    updated_at = NOW()
RETURNING id, user_id, kg_id, subject_text, description_text, is_key_subject, embedding;

-- name: LinkSubjectMemoryPair :exec
INSERT INTO subject_memory_pair_links (subject_id, pair_id, user_id, kg_id)
VALUES (sqlc.arg(subject_id), sqlc.arg(pair_id), sqlc.arg(user_id), sqlc.arg(kg_id))
ON CONFLICT (subject_id, pair_id) DO NOTHING;

-- name: FetchMemories :many
SELECT id, firestore_pair_id, user_id, kg_id, session_id, title, description,
    prompt, response, input, output, source, source_context, timestamp,
    timezone_offset, seed_memories, retrieval_metadata, conversation_embedding
FROM memory_pairs
WHERE user_id = sqlc.arg(user_id) AND firestore_pair_id = ANY(sqlc.arg(pair_ids)::text[])
ORDER BY array_position(sqlc.arg(pair_ids)::text[], firestore_pair_id);

-- name: SearchMemories :many
SELECT id, firestore_pair_id, user_id, kg_id, session_id, title, description,
    prompt, response, input, output, source, source_context, timestamp,
    timezone_offset, seed_memories, retrieval_metadata, conversation_embedding,
    (1 - (conversation_embedding <=> sqlc.arg(embedding)::vector))::float8 AS similarity
FROM memory_pairs
WHERE user_id = sqlc.arg(user_id) AND kg_id = sqlc.arg(kg_id)
  AND conversation_embedding IS NOT NULL
  AND (sqlc.arg(session_id)::text = '' OR COALESCE(session_id, 'main') = COALESCE(NULLIF(sqlc.arg(session_id)::text, ''), 'main'))
  AND (sqlc.arg(exclude_pair_ids)::text[] IS NULL OR firestore_pair_id != ALL(sqlc.arg(exclude_pair_ids)::text[]))
  AND (1 - (conversation_embedding <=> sqlc.arg(embedding)::vector)) >= sqlc.arg(min_similarity)::float8
ORDER BY conversation_embedding <=> sqlc.arg(embedding)::vector, timestamp DESC
LIMIT sqlc.arg(limit_count);

-- name: SearchSubjects :many
SELECT s.id, s.user_id, s.kg_id, s.subject_text, s.description_text, s.is_key_subject, s.embedding,
    (1 - (s.embedding <=> sqlc.arg(embedding)::vector))::float8 AS similarity,
    COUNT(smpl.pair_id)::bigint AS memory_count
FROM subjects s
LEFT JOIN subject_memory_pair_links smpl ON smpl.subject_id = s.id
WHERE s.user_id = sqlc.arg(user_id) AND s.kg_id = sqlc.arg(kg_id)
  AND s.embedding IS NOT NULL
  AND (1 - (s.embedding <=> sqlc.arg(embedding)::vector)) >= sqlc.arg(min_similarity)::float8
GROUP BY s.id
ORDER BY s.embedding <=> sqlc.arg(embedding)::vector, memory_count DESC, s.updated_at DESC
LIMIT sqlc.arg(limit_count);

-- name: SearchMemoriesBySubject :many
SELECT mp.id, mp.firestore_pair_id, mp.user_id, mp.kg_id, mp.session_id, mp.title, mp.description,
    mp.prompt, mp.response, mp.input, mp.output, mp.source, mp.source_context, mp.timestamp,
    mp.timezone_offset, mp.seed_memories, mp.retrieval_metadata, mp.conversation_embedding,
    (1 - (mp.conversation_embedding <=> sqlc.arg(embedding)::vector))::float8 AS similarity
FROM memory_pairs mp
JOIN subject_memory_pair_links smpl ON smpl.pair_id = mp.id
WHERE smpl.subject_id = sqlc.arg(subject_id) AND mp.user_id = sqlc.arg(user_id)
  AND mp.conversation_embedding IS NOT NULL
  AND (1 - (mp.conversation_embedding <=> sqlc.arg(embedding)::vector)) >= sqlc.arg(min_similarity)::float8
ORDER BY mp.conversation_embedding <=> sqlc.arg(embedding)::vector, mp.timestamp DESC
LIMIT sqlc.arg(limit_count);
