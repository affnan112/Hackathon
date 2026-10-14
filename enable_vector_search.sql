-- Run once in the Supabase SQL Editor.
-- Enables true semantic (meaning-based) search, which generalizes across
-- any wording variation — "reinheid" vs "netheid", "loslopen" vs "aan de
-- leiband" — without needing a hand-maintained synonym list.

-- pgvector is available by default on Supabase Postgres.
create extension if not exists vector;

-- 1536 dimensions matches OpenAI's text-embedding-3-small model.
alter table chunks add column if not exists embedding vector(1536);

-- Semantic search: takes the officer's question, already converted to an
-- embedding by the app, and returns the chunks closest to it in meaning
-- (cosine similarity), not by literal word overlap.
-- The parameter is typed `text` (not `vector`) and cast inside the
-- function — this avoids type-serialization quirks between different
-- Supabase client library versions when passing the embedding over RPC.
create or replace function match_chunks(query_embedding text, match_count int default 6)
returns table (
    id int,
    document_id int,
    page text,
    article text,
    text text,
    similarity float
)
language sql stable
as $$
    select
        c.id,
        c.document_id,
        c.page,
        c.article,
        c.text,
        1 - (c.embedding <=> query_embedding::vector) as similarity
    from chunks c
    where c.embedding is not null
    order by c.embedding <=> query_embedding::vector
    limit match_count;
$$;

-- At ~1,000-3,000 rows, a plain scan with the <=> operator is fast enough
-- (well under 100ms) — no index needed yet. If your corpus grows into the
-- tens of thousands, add: create index on chunks using ivfflat (embedding
-- vector_cosine_ops);
