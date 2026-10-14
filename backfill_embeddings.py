"""
One-time backfill: generate and store an OpenAI embedding for every chunk
that doesn't already have one. Run this once, after running
enable_vector_search.sql. Safe to re-run — it only processes chunks where
embedding IS NULL, so if it's interrupted partway through, just run it
again and it picks up where it left off.

Usage:
    $env:OPENAI_API_KEY = "sk-..."   (PowerShell)
    python backfill_embeddings.py

This will take a couple of minutes for ~1,250 chunks (mostly the sequential
database writes, not the embedding calls themselves — those are batched).
"""

import os
from openai import OpenAI
from db import get_client

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 100


def get_chunks_without_embedding() -> list[dict]:
    supabase = get_client()
    response = (
        supabase.table("chunks")
        .select("id, text")
        .is_("embedding", "null")
        .execute()
    )
    return response.data or []


def embed_batch(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def main():
    supabase = get_client()
    chunks = get_chunks_without_embedding()
    total = len(chunks)
    print(f"Found {total} chunks without an embedding.")

    if total == 0:
        print("Nothing to do.")
        return

    done = 0
    for i in range(0, total, BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        embeddings = embed_batch(texts)

        for chunk, embedding in zip(batch, embeddings):
            supabase.table("chunks").update({"embedding": embedding}).eq(
                "id", chunk["id"]
            ).execute()
            done += 1

        print(f"Embedded {done}/{total}")

    print("Done. Semantic search is now live for all chunks.")


if __name__ == "__main__":
    main()
