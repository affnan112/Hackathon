"""
db.py — Supabase client for the REAL schema already in your project:
  documents(id, filename, title, level, doc_date, status, public_url, warning, page_count)
  chunks(id, document_id, page, article, paragraphs, text, tsv)
  answers(id, question, finding, quote, chunk_id, decision, officer_note, created_at)
"""

import os
import re
from supabase import create_client, Client
from openai import OpenAI

SUPABASE_URL = os.environ.get(
    "SUPABASE_URL", "https://oqrhngrbbupkdlozzanw.supabase.co"
)
SUPABASE_KEY = os.environ.get(
    "SUPABASE_KEY", "sb_publishable_XchudDnVCLAiw1og62DPzA_qZAS39Yi"
)

_client: Client | None = None
_openai_client: OpenAI | None = None
EMBEDDING_MODEL = "text-embedding-3-small"

STOPWORDS = {
    "de", "het", "een", "van", "te", "en", "in", "op", "voor", "met", "om",
    "dat", "die", "dit", "is", "zijn", "wordt", "worden", "of", "bij", "aan",
    "ik", "heb", "jij", "je", "u", "hij", "zij", "wij", "we", "als", "niet",
    "geen", "welke", "wat", "hoe", "dien", "kan", "moet", "er", "ook", "dan",
}

# Small synonym map: officers phrase questions in everyday language, but
# regulation text often uses a different specific term for the same thing.
# Postgres full-text search stems words to a root but does NOT know
# synonyms, so "reinheid" and "netheid" are invisible to each other without
# this. Expand this list as you notice more misses during testing.
SYNONYMS = {
    "stoep": ["voetpad"],
    "voetpad": ["stoep"],
    "reinheid": ["netheid", "proper", "properheid", "schoon"],
    "netheid": ["reinheid", "proper", "schoon"],
    "afval": ["vuil", "vuilnis"],
    "kosten": ["retributie", "tarief", "tarieven", "prijs", "vergoeding"],
    "vergoeding": ["retributie", "kosten"],
    "prijs": ["retributie", "tarief"],
    "winkel": ["handelszaak", "zaak"],
    "aanvraag": ["aanvraagformulier"],
    "hond": ["honden", "leiband", "aangelijnd", "rondlopen", "hondenloopzone"],
    "honden": ["hond", "leiband", "aangelijnd", "rondlopen", "hondenloopzone"],
    "loslopen": ["rondlopen", "leiband", "aangelijnd", "hondenloopzone"],
    "straat": ["weg", "openbare"],
    "weg": ["straat", "openbare"],
}


def _keyword_score(text: str, words: list[str]) -> int:
    """Count how many of the given keywords appear (case-insensitive) in a
    chunk's text. Used to rank fallback matches by actual overlap instead of
    returning whatever arbitrary order the database happens to produce —
    otherwise the oldest-inserted document tends to win every time,
    regardless of relevance."""
    t = text.lower()
    return sum(1 for w in words if w.lower() in t)


def _expand_query_terms(words: list[str]) -> list[str]:
    """Add known synonyms for each word, without duplicates."""
    expanded = list(words)
    for w in words:
        for syn in SYNONYMS.get(w, []):
            if syn not in expanded:
                expanded.append(syn)
    return expanded


def get_client() -> Client:
    global _client
    if _client is None:
        _client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _client


def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return _openai_client


def embed_text(text: str) -> list[float] | None:
    """Convert text into an embedding vector for semantic search. Returns
    None on any failure (missing key, network issue, etc.) so callers can
    fall back to lexical search rather than crash the whole request."""
    try:
        response = _get_openai_client().embeddings.create(
            model=EMBEDDING_MODEL, input=text
        )
        return response.data[0].embedding
    except Exception:
        return None


def _semantic_search(question: str, limit: int) -> list[dict]:
    """Primary retrieval path: embed the question and find chunks closest
    to it in *meaning*, not literal wording. This is what actually solves
    the general synonym problem — "reinheid" and "netheid" end up close in
    embedding space even though they share no letters, and this holds for
    any wording variation, not just ones we've anticipated and hard-coded."""
    embedding = embed_text(question)
    if embedding is None:
        return []
    try:
        # Passed as a bracketed string, not a raw Python list — matches
        # match_chunks()'s `text` parameter, cast to vector inside the SQL
        # function. Keeps this robust across client library versions.
        embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"
        response = get_client().rpc(
            "match_chunks", {"query_embedding": embedding_str, "match_count": limit}
        ).execute()
        rows = response.data or []
        for row in rows:
            row["rank"] = row.get("similarity")  # unify field name with lexical path
        return rows
    except Exception:
        return []


def fetch_all_documents() -> list[dict]:
    """Master list of source documents, for the 'manage sources' panel."""
    client = get_client()
    response = client.table("documents").select("*").order("id").execute()
    return response.data or []


def _ilike_fallback(client, expanded_words: list[str], limit: int) -> list[dict]:
    """Keyword fallback used when full-text search finds nothing. Pulls a
    larger candidate pool, then ranks it by how many distinct keywords each
    chunk actually contains — so results are ordered by real overlap rather
    than by database insertion order (which otherwise makes whichever
    document was uploaded first win by accident, every time)."""
    or_filter = ",".join(f"text.ilike.%{w}%" for w in expanded_words[:8])
    try:
        response = (
            client.table("chunks")
            .select("id, document_id, page, article, text")
            .or_(or_filter)
            .limit(max(limit * 5, 30))
            .execute()
        )
        candidates = response.data or []
    except Exception:
        return []

    for row in candidates:
        row["rank"] = None  # not a true relevance score — flagged as such in the UI
        row["_keyword_score"] = _keyword_score(row.get("text", ""), expanded_words)

    candidates.sort(key=lambda r: r["_keyword_score"], reverse=True)
    return candidates[:limit]


def retrieve_chunks(question: str, top_k: int = 6) -> list[dict]:
    """Find the most relevant chunks for a question, each enriched with its
    parent document's metadata (title, level, status, warning, public_url).

    Retrieval chain, each step only runs if the one before it found nothing:
      1. Semantic (embedding) search — matches by meaning, handles any
         wording variation generically.
      2. Lexical full-text search (search_chunks RPC) — exact/stemmed word
         matching, in case the embedding call fails (e.g. API hiccup).
      3. Ranked ILIKE keyword fallback — last resort.
    """
    client = get_client()

    chunk_rows = _semantic_search(question, top_k)

    q_words = [w for w in re.findall(r"\w+", question.lower()) if w not in STOPWORDS]
    expanded_words = _expand_query_terms(q_words)

    if not chunk_rows:
        extra_terms = [w for w in expanded_words if w not in q_words]
        expanded_query = question if not extra_terms else question + " OR " + " OR ".join(extra_terms)
        try:
            response = get_client().rpc(
                "search_chunks", {"search_query": expanded_query, "result_limit": top_k}
            ).execute()
            chunk_rows = response.data or []
        except Exception:
            chunk_rows = []

    if not chunk_rows and expanded_words:
        chunk_rows = _ilike_fallback(client, expanded_words, top_k)

    if not chunk_rows:
        return []

    doc_ids = list({c["document_id"] for c in chunk_rows})
    docs_response = client.table("documents").select("*").in_("id", doc_ids).execute()
    docs_by_id = {d["id"]: d for d in (docs_response.data or [])}

    return [{**c, "document": docs_by_id.get(c["document_id"], {})} for c in chunk_rows]


def retrieve_related_documents(question: str, limit: int = 10) -> list[dict]:
    """Broader search for the full 'ranked supporting sources' list — same
    retrieval chain as retrieve_chunks(), but with a wider limit so more
    genuinely-related documents surface, not just the single tightest
    match used to ground the answer.
    """
    client = get_client()

    chunk_rows = _semantic_search(question, limit)

    q_words = [w for w in re.findall(r"\w+", question.lower()) if w not in STOPWORDS]
    expanded_words = _expand_query_terms(q_words)

    if not chunk_rows and expanded_words:
        broad_query = " OR ".join(expanded_words)
        try:
            response = client.rpc(
                "search_chunks", {"search_query": broad_query, "result_limit": limit}
            ).execute()
            chunk_rows = response.data or []
        except Exception:
            chunk_rows = []

    if not chunk_rows:
        chunk_rows = _ilike_fallback(client, expanded_words, limit)

    if not chunk_rows:
        return []

    doc_ids = list({c["document_id"] for c in chunk_rows})
    docs_response = client.table("documents").select("*").in_("id", doc_ids).execute()
    docs_by_id = {d["id"]: d for d in (docs_response.data or [])}

    return [{**c, "document": docs_by_id.get(c["document_id"], {})} for c in chunk_rows]


def fetch_reviewed_answers(limit: int = 200) -> list[dict]:
    """Every past answer an officer has actually reviewed (confirmed,
    corrected, or rejected) — the raw material for learning from feedback."""
    client = get_client()
    response = (
        client.table("answers")
        .select("id, question, finding, quote, decision, officer_note")
        .not_.is_("decision", "null")
        .order("id", desc=True)
        .limit(limit)
        .execute()
    )
    return response.data or []


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def find_similar_past_answer(question: str, threshold: float = 0.85) -> dict | None:
    """Check whether an officer has already reviewed a very similar past
    question. If so, that decision (confirmed/corrected/rejected) is
    returned so the new answer can be guided by it — this is what makes the
    knowledge base actually improve from officer feedback over time,
    rather than repeating the same mistake indefinitely.

    Demo-scale simplification: past questions are embedded on the fly here
    rather than cached in the database. Fine for the handful of reviewed
    answers a hackathon produces; a production version would store each
    answer's embedding at review time instead of recomputing it every call.
    """
    reviewed = fetch_reviewed_answers()
    if not reviewed:
        return None

    question_embedding = embed_text(question)
    if question_embedding is None:
        return None

    try:
        past_texts = [r["question"] for r in reviewed]
        response = _get_openai_client().embeddings.create(
            model=EMBEDDING_MODEL, input=past_texts
        )
        past_embeddings = [item.embedding for item in response.data]
    except Exception:
        return None

    best_score = -1.0
    best_entry = None
    for entry, emb in zip(reviewed, past_embeddings):
        score = _cosine_similarity(question_embedding, emb)
        if score > best_score:
            best_score = score
            best_entry = entry

    if best_entry is not None and best_score >= threshold:
        return {**best_entry, "similarity": best_score}
    return None


def add_document(
    filename: str,
    title: str,
    level: str,
    doc_date: str,
    status: str,
    public_url: str,
    warning: str = "",
    page_count: int = 1,
) -> dict:
    client = get_client()
    response = (
        client.table("documents")
        .insert(
            {
                "filename": filename,
                "title": title,
                "level": level,
                "doc_date": doc_date,
                "status": status,
                "public_url": public_url,
                "warning": warning,
                "page_count": page_count,
            }
        )
        .execute()
    )
    return response.data[0] if response.data else {}


def add_chunk(
    document_id: int, page: str, article: str, text: str, paragraphs: list | None = None
) -> dict:
    client = get_client()
    row = {
        "document_id": document_id,
        "page": page,
        "article": article or "",
        "paragraphs": paragraphs or [],
        "text": text,
    }
    # Embed immediately so new sources are semantically searchable right
    # away, without needing to rerun the backfill script.
    embedding = embed_text(text)
    if embedding is not None:
        row["embedding"] = embedding
    response = client.table("chunks").insert(row).execute()
    return response.data[0] if response.data else {}


def log_answer(
    question: str,
    finding: str,
    quote: str,
    chunk_id: int | None,
    decision: str | None = None,
    officer_note: str | None = None,
) -> dict:
    client = get_client()
    response = (
        client.table("answers")
        .insert(
            {
                "question": question,
                "finding": finding,
                "quote": quote,
                "chunk_id": chunk_id,
                "decision": decision,
                "officer_note": officer_note,
            }
        )
        .execute()
    )
    return response.data[0] if response.data else {}


def update_answer_review(answer_id: int, decision: str, officer_note: str | None = None) -> dict:
    client = get_client()
    response = (
        client.table("answers")
        .update({"decision": decision, "officer_note": officer_note})
        .eq("id", answer_id)
        .execute()
    )
    return response.data[0] if response.data else {}
