"""
PROV-AI Challenge 2 prototype: "Answer Like the Expert"

Retrieves chunks from Supabase (documents + chunks tables), grounds a
Dutch officer-briefing answer in them, and logs the result to the
answers table for traceability.
"""

import os
import re
import sys
from openai import OpenAI

from db import retrieve_chunks, log_answer, find_similar_past_answer

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

SYSTEM_PROMPT = """\
Je bent een assistent voor een medewerker lokale economie bij een Vlaamse
gemeente. Je krijgt een vraag en een set brontekstfragmenten. Elk fragment
heeft: titel, niveau, artikel, pagina, datum, status en eventueel een
waarschuwing van het documentbeheer.

Mogelijke statuswaarden en wat ze betekenen:
- "current": van kracht, actuele regelgeving.
- "undated": geen datum bekend — actualiteit niet bevestigd.
- "guidance": richtlijn/toelichting, GEEN bindende wetgeving.
- "outdated_copy": een verouderde kopie van een tekst die ondertussen is
  gewijzigd — de kopie zelf bevat geen latere wijzigingen.

STRIKTE REGELS:
1. Baseer je antwoord UITSLUITEND op de gegeven fragmenten. Verzin nooit
   informatie die niet letterlijk in de fragmenten staat.
2. Citeer minstens één exact fragment (kort, woordelijk) als bewijs.
3. Als een fragment een "Waarschuwing" heeft, of status "outdated_copy" of
   "undated" is, neem die waarschuwing (parafraseer kort) op bij
   Onzekerheid. Vermeld dit ALTIJD wanneer zo'n fragment gebruikt wordt.
4. Als status "guidance" is, vermeld expliciet dat dit een richtlijn is,
   geen bindende wetgeving.
5. NOOIT "geen bekende onzekerheid" schrijven tenzij je zeker weet dat de
   fragmenten de vraag volledig dekken, inclusief kosten, termijnen of
   vervolgstappen die de vraag kan impliceren. Bij twijfel: benoem wat
   ontbreekt.
6. Als GEEN van de fragmenten relevant is, zeg dat expliciet bij Bevinding
   in plaats van te gokken.
7. Als er een EERDERE BEOORDELING DOOR EEN MEDEWERKER wordt meegegeven:
   - Bij "confirmed": dit bevestigt dat een gelijkaardig eerder antwoord
     correct was. Stem je Bevinding hiermee af als de vraag echt
     vergelijkbaar is.
   - Bij "corrected": de medewerker heeft een eerder antwoord gecorrigeerd.
     Volg die correctie (officer_note) als leidraad voor je Bevinding — dit
     weegt zwaarder dan je eigen interpretatie, al gebruik je nog steeds een
     echt citaat en bron uit de gegeven fragmenten bij Bewijs/Bron.
   - Bij "rejected": vermijd de aanpak of formulering die eerder werd
     afgewezen.
   - Vermeld in Onzekerheid kort dat een eerdere, gelijkaardige vraag door
     een medewerker werd beoordeeld.
8. Antwoord ALTIJD in het Nederlands, in exact dit format:

Vraag: <herhaal de vraag kort>
Bevinding: <korte samenvatting op basis van de bronnen>
Bewijs: "<kort, woordelijk citaat uit het relevante fragment>"
Bron: <titel, artikel, pagina, datum> — <url of "geen link beschikbaar">
Toepasselijkheid: <voor welke situatie dit geldt, en de status van de bron>
Onzekerheid: <wat ontbreekt, waarschuwingen van het brondocument, of "geen bekende onzekerheid">
"""

EMAIL_PROMPT = """\
Je herschrijft een intern medewerker-antwoord naar een e-mailconcept voor de
entrepreneur die de vraag stelde. Gebruik ENKEL informatie uit het gegeven
antwoord. Verzin geen naam, bedragen of termijnen die niet in het antwoord
staan; gebruik [placeholder] voor ontbrekende gegevens.
Sluit af met: "Dit is een conceptantwoord. Controleer de brontekst voordat u
dit verstuurt." Antwoord in het Nederlands, in briefvorm (Beste ... /
Met vriendelijke groeten, [naam], dienst lokale economie).
"""

FIELDS = ["Vraag", "Bevinding", "Bewijs", "Bron", "Toepasselijkheid", "Onzekerheid"]


def parse_answer_fields(text: str) -> dict:
    """Extract the labeled fields from a formatted answer, for logging."""
    out = {}
    for i, label in enumerate(FIELDS):
        start_match = re.search(rf"{label}:\s*", text)
        if not start_match:
            continue
        start_idx = start_match.end()
        end_idx = len(text)
        if i + 1 < len(FIELDS):
            next_match = re.search(rf"{FIELDS[i + 1]}:", text[start_idx:])
            if next_match:
                end_idx = start_idx + next_match.start()
        out[label] = text[start_idx:end_idx].strip()
    return out


def format_sources_for_prompt(chunks: list[dict]) -> str:
    blocks = []
    for c in chunks:
        doc = c.get("document") or {}
        blocks.append(
            f"---\n"
            f"Titel: {doc.get('title', 'onbekend')}\n"
            f"Niveau: {doc.get('level', 'onbekend')}\n"
            f"Artikel: {c.get('article') or 'n.v.t.'}\n"
            f"Pagina: {c.get('page') or 'n.v.t.'}\n"
            f"Datum: {doc.get('doc_date') or 'onbekend'}\n"
            f"Status: {doc.get('status', 'onbekend')}\n"
            f"Waarschuwing: {doc.get('warning') or 'geen'}\n"
            f"URL: {doc.get('public_url') or 'geen link beschikbaar'}\n"
            f"Tekst: {c['text']}\n"
        )
    return "\n".join(blocks)


def _build_bron_line(chunk: dict) -> str:
    """Construct the citation from verified retrieval metadata, rather than
    trusting the model to correctly remember which document/article a quote
    came from when several fragments were given to it at once."""
    doc = chunk.get("document") or {}
    title = doc.get("title", "onbekend document")
    article = chunk.get("article") or "n.v.t."
    page = chunk.get("page") or "n.v.t."
    date = doc.get("doc_date") or "onbekend"
    url = doc.get("public_url") or "geen link beschikbaar"
    return f"{title}, {article}, pagina {page}, datum {date} — {url}"


def _quote_is_verifiable(quote: str, chunks: list[dict]) -> bool:
    """Check the quoted evidence actually appears (near-verbatim) in one of
    the retrieved chunks, catching cases where the model paraphrased rather
    than quoted."""
    q = re.sub(r"\s+", " ", quote.strip().strip('"').lower())
    if not q:
        return False
    for c in chunks:
        text = re.sub(r"\s+", " ", c.get("text", "").lower())
        if q in text:
            return True
    return False


def answer_question(question: str) -> tuple[str, list[dict], int | None]:
    """Returns (answer_text, chunks_used, answer_log_id)."""
    chunks = retrieve_chunks(question)
    if not chunks:
        return (
            "Bevinding: Dit systeem vond geen relevante bron voor deze vraag. "
            "Voeg eventueel een nieuwe bron toe of herformuleer de vraag.",
            [],
            None,
        )

    sources_block = format_sources_for_prompt(chunks)

    # Check whether an officer has already reviewed a very similar past
    # question — if so, that decision guides this new answer.
    past = find_similar_past_answer(question)
    feedback_block = ""
    if past:
        feedback_block = (
            f"\n\nEERDERE BEOORDELING DOOR EEN MEDEWERKER "
            f"(gelijkaardige vraag, similariteit {past['similarity']:.2f}):\n"
            f"Eerdere vraag: {past['question']}\n"
            f"Beslissing: {past['decision']}\n"
            f"Eerdere Bevinding: {past.get('finding', '')}\n"
            f"Notitie/correctie medewerker: {past.get('officer_note') or 'geen'}\n"
        )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"VRAAG:\n{question}\n\nBRONFRAGMENTEN:\n{sources_block}{feedback_block}",
            },
        ],
        temperature=0.1,
    )
    answer_text = response.choices[0].message.content
    fields = parse_answer_fields(answer_text)

    # Override the model's own Bron line with the verified citation for the
    # top-ranked chunk. The model sometimes cites the right passage but the
    # wrong document/article when several fragments are in context — this
    # closes that gap by trusting our own retrieval metadata instead.
    primary_chunk = chunks[0]
    correct_bron = _build_bron_line(primary_chunk)
    new_answer_text = re.sub(
        r"Bron:\s*.*?(?=\n\s*Toepasselijkheid:)",
        lambda m: f"Bron: {correct_bron}",
        answer_text,
        flags=re.DOTALL,
    )
    if new_answer_text == answer_text:
        # Fallback if the model's formatting didn't match the expected
        # multi-line pattern — try a single-line replacement instead.
        new_answer_text = re.sub(r"Bron:.*", f"Bron: {correct_bron}", answer_text)
    answer_text = new_answer_text

    # Flag (don't silently accept) a quote that can't be found verbatim in
    # any retrieved chunk — it may have been paraphrased rather than quoted.
    if not _quote_is_verifiable(fields.get("Bewijs", ""), chunks):
        answer_text += (
            "\n\n[Systeemwaarschuwing: het geciteerde bewijs kon niet woordelijk "
            "worden teruggevonden in de geraadpleegde fragmenten. Controleer "
            "de brontekst voordat u dit antwoord gebruikt.]"
        )

    answer_id = None
    try:
        log_row = log_answer(
            question=question,
            finding=fields.get("Bevinding", ""),
            quote=fields.get("Bewijs", ""),
            chunk_id=chunks[0]["id"],  # primary citation
        )
        answer_id = log_row.get("id")
    except Exception:
        pass  # never block the officer's view on a logging failure

    return answer_text, chunks, answer_id


def to_email_draft(briefing_answer: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": EMAIL_PROMPT},
            {"role": "user", "content": briefing_answer},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    if len(sys.argv) < 2:
        question = "Ik wil een vaste standplaats op de markt in Schoten. Hoe dien ik een aanvraag in?"
    else:
        question = " ".join(sys.argv[1:])

    print(f"\n=== VRAAG ===\n{question}\n")
    print("=== ANTWOORD (briefing) ===")
    answer, chunks, answer_id = answer_question(question)
    print(answer)
    print(f"\n[gelogd als answers.id = {answer_id}]")
    if chunks:
        print("\n=== E-MAILCONCEPT ===")
        print(to_email_draft(answer))
