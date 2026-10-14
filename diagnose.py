"""
Diagnostic script — runs each retrieval layer individually and prints what
actually happens, including real exceptions (the app itself hides these on
purpose so it can gracefully fall back, which is right for production but
useless for debugging).

Usage:
    $env:OPENAI_API_KEY = "sk-..."   (PowerShell)
    python diagnose.py "Mag mijn hond loslopen op straat in Schoten?"
"""

import sys
from db import get_client, embed_text

question = sys.argv[1] if len(sys.argv) > 1 else "Mag mijn hond loslopen op straat in Schoten?"
print(f"Question: {question}\n")

print("=== 1. Embedding the question ===")
embedding = embed_text(question)
if embedding is None:
    print("FAILED: embed_text() returned None.")
    print("  -> Check OPENAI_API_KEY is set in THIS terminal session.")
else:
    print(f"OK: got embedding with {len(embedding)} dimensions")

print("\n=== 2. Checking how many chunks have embeddings in Supabase ===")
client = get_client()
try:
    with_emb = client.table("chunks").select("id", count="exact").not_.is_("embedding", "null").execute()
    without_emb = client.table("chunks").select("id", count="exact").is_("embedding", "null").execute()
    print(f"Chunks WITH embedding: {with_emb.count}")
    print(f"Chunks WITHOUT embedding: {without_emb.count}")
    if with_emb.count == 0:
        print("  -> Backfill has not run yet (or didn't complete). Run backfill_embeddings.py.")
except Exception as e:
    print(f"FAILED to check: {e}")

print("\n=== 3. Calling match_chunks RPC directly (semantic search) ===")
if embedding is not None:
    embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"
    try:
        resp = client.rpc(
            "match_chunks", {"query_embedding": embedding_str, "match_count": 5}
        ).execute()
        print(f"OK: {len(resp.data)} rows returned")
        for row in resp.data:
            print(f"  - chunk {row['id']}, doc {row['document_id']}, similarity {row['similarity']:.4f}: {row['text'][:80]}...")
    except Exception as e:
        print(f"FAILED: {e}")
else:
    print("SKIPPED (no embedding to search with)")

print("\n=== 4. Calling search_chunks RPC directly (lexical search) ===")
try:
    resp = client.rpc(
        "search_chunks", {"search_query": question, "result_limit": 5}
    ).execute()
    print(f"OK: {len(resp.data)} rows returned")
    for row in resp.data:
        print(f"  - chunk {row['id']}, doc {row['document_id']}, rank {row['rank']:.4f}: {row['text'][:80]}...")
except Exception as e:
    print(f"FAILED: {e}")
