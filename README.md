# PROV-AI · Assistent Lokale Economie

An AI assistant that answers regulatory questions for *medewerkers lokale economie* (local economy officers) in Flemish municipalities — built for **Challenge 2: "Answer Like the Expert"** at the Tectonic Hackathon, solved for a medewerker at **Provincie Antwerpen**.

Demo: https://youtu.be/YfeVOCXpGSw

## The problem

Local economy officers field questions from entrepreneurs — market stall permits, shop signage rules, terrace fees — and spend around 40 minutes per question digging through scattered regulations (gemeentelijk, provinciaal, Vlaams, federaal) to answer correctly. The information exists; finding and citing it reliably is the bottleneck.

## What this does

You ask a question in plain Dutch (e.g. *"Ik wil een vaste standplaats op de markt in Schoten. Hoe dien ik een aanvraag in?"*) and get back a structured briefing:

```
Vraag: ...
Bevinding: ...
Bewijs: "<exact quote from the source>"
Bron: <title, article, page, date> — <url>
Toepasselijkheid: ...
Onzekerheid: ...
```

Every answer is grounded only in retrieved source fragments — never free-generated — and every citation is re-verified against the actual retrieved text before being shown, rather than trusted on the model's word.

### Key design choices

- **Semantic search, not just keywords.** Questions are embedded (OpenAI `text-embedding-3-small`) and matched against embedded regulation chunks by meaning, so "reinheid" and "netheid" (synonyms) match each other even without a hand-maintained synonym list.
- **Verified citations.** The `Bron` line is rebuilt from the actual retrieved chunk's metadata rather than trusted from the model's output, and any quoted evidence is checked for verbatim presence in the retrieved text — if it can't be found, the answer is flagged with a system warning instead of silently shown as fact.
- **Status-aware answers.** Each source carries a status (`current`, `outdated_copy`, `undated`, `guidance`) and the model is instructed to always surface warnings when using an outdated or non-binding source.
- **Officer in the loop.** Officers confirm, correct, or reject every generated answer. Nothing is auto-sent to an entrepreneur. Confirmed/corrected decisions are logged and reused: a similar future question is answered with that prior decision as guidance.
- **Graceful retrieval fallback chain:** semantic search → PostgreSQL full-text search → keyword (ILIKE) matching — each tier only runs if the one before found nothing, so the app still responds if embeddings or full-text search fail.

## Architecture

```
Browser (index.html)
   │  fetch /ask, /review, /email, /documents
   ▼
Flask server (server.py)
   │
   ├─ assistant.py   → builds the grounded Dutch briefing via OpenAI (gpt-4o-mini),
   │                    verifies citations, logs the answer
   │
   └─ db.py          → Supabase client: retrieval chain (semantic → full-text →
                         keyword fallback), document/chunk/answer CRUD
                         │
                         ▼
                   Supabase (Postgres + pgvector)
                     documents · chunks · answers
```

## Files

| File | Purpose |
|---|---|
| `server.py` | Flask app — serves the UI and exposes `/ask`, `/email`, `/review`, `/documents` |
| `assistant.py` | Core logic: retrieval → grounded Dutch answer → citation verification → logging |
| `db.py` | Supabase client: semantic/full-text/keyword retrieval chain, document & answer storage |
| `index.html` | Single-page frontend — ask a question, review ranked sources, confirm/correct/reject, generate an email draft |
| `supabase_setup.sql` | Creates the `documents` and `answer_log` tables (run once) |
| `search_chunks.sql` | Postgres function for ranked full-text search (`ts_rank_cd`) |
| `enable_vector_search.sql` | Enables `pgvector`, adds the `embedding` column, creates `match_chunks()` for semantic search |
| `backfill_embeddings.py` | One-time script to embed any existing chunks that don't have an embedding yet |
| `diagnose.py` | Diagnostic script — runs each retrieval layer individually and prints real errors (the app itself fails gracefully and hides these on purpose) |
| `sources.py` | Early hardcoded prototype source list, superseded by the Supabase-backed `documents`/`chunks` tables |

## Setup

1. **Database** — in the Supabase SQL Editor, run in order:
   - `supabase_setup.sql`
   - `enable_vector_search.sql`
   - `search_chunks.sql`
2. **Dependencies**
   ```bash
   pip install flask supabase openai
   ```
3. **Environment variables**
   ```bash
   export OPENAI_API_KEY="sk-..."
   export SUPABASE_URL="https://<your-project>.supabase.co"
   export SUPABASE_KEY="<your-supabase-publishable-key>"
   ```
   (On Windows PowerShell: `$env:OPENAI_API_KEY = "sk-..."`.)
4. **Backfill embeddings** for any chunks added before semantic search was enabled:
   ```bash
   python backfill_embeddings.py
   ```
5. **Run**
   ```bash
   python server.py
   ```
   Open http://127.0.0.1:5000.

## Status

Hackathon prototype (one-day build). Row-level security is left open on the Supabase tables for demo convenience — not how this would be configured for production use, where officer sessions and write access would be authenticated and restricted.

## License

Built for the Tectonic Hackathon (KBC challenge track). No license specified.
