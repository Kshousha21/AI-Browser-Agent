"""Validate output locally; provider metadata never establishes provenance."""
import json
import re
from typing import Literal, Union
from pydantic import ValidationError, create_model, Field
from local_retrieval.models import SearchHit
from .errors import RagError
from .models import Citation, INSUFFICIENT, ProviderAnswer, RagSettings, VisualEvidenceCitation
from .visual import VisualSource


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate key")
        value[key] = item
    return value


def validate_answer(raw: str, sources: dict[str, SearchHit], settings: RagSettings):
    if not isinstance(raw, str) or len(raw) > settings.max_output_chars:
        raise RagError("rag_invalid_answer")
    try:
        data = json.loads(raw, object_pairs_hook=_unique_object)
        answer = ProviderAnswer.model_validate(data)
    except (ValueError, ValidationError, RecursionError):
        raise RagError("rag_invalid_answer") from None
    if not answer.answer.strip() or len(answer.answer) > settings.max_answer_chars:
        raise RagError("rag_invalid_answer")
    if answer.status == "insufficient_evidence":
        if answer.citations:
            raise RagError("rag_invalid_answer")
        # Never promote factual prose disguised as an abstention.
        return answer.status, INSUFFICIENT, []
    if not answer.citations:
        raise RagError("rag_invalid_answer")
    citations, seen = [], set()
    for citation in answer.citations:
        hit = sources.get(citation.source_id)
        if hit is None:
            raise RagError("rag_unknown_source")
        if isinstance(citation, VisualEvidenceCitation):
            if not isinstance(hit, VisualSource):
                raise RagError("rag_invalid_evidence")
            if citation.source_id not in seen:
                seen.add(citation.source_id)
                citations.append(Citation(source_id=citation.source_id, document_id=hit.document_id,
                    page_start=hit.page, page_end=hit.page, evidence_type="visual",
                    rendered_page_id=hit.rendered_page_id))
            continue
        if isinstance(hit, VisualSource):
            raise RagError("rag_invalid_evidence")
        if not citation.evidence.strip() or citation.evidence not in hit.text:
            raise RagError("rag_invalid_evidence")
        if citation.source_id not in seen:
            seen.add(citation.source_id)
            citations.append(Citation(source_id=citation.source_id,
                **{key: getattr(hit, key) for key in (
                    "document_id", "chunk_id", "page_start", "page_end", "chunk_index",
                    "char_start", "char_end", "trust")}))
    visual = {(item.document_id, item.page_start): item for item in citations if item.evidence_type == "visual"}
    citations = [item.model_copy(update={"evidence_type": "both",
        "rendered_page_id": visual[(item.document_id, item.page_start)].rendered_page_id})
        if item.evidence_type == "text" and (item.document_id, item.page_start) in visual else item for item in citations]
    return answer.status, answer.answer, citations


def citation_copy_hints(raw: str, sources, settings: RagSettings) -> list[dict[str, str]]:
    """Suggest exact local substrings for whitespace-only quote mistakes.

    These are untrusted source excerpts for the existing single repair attempt,
    never corrected output or authority to skip validate_answer.
    """
    if not isinstance(raw, str) or len(raw) > settings.max_output_chars:
        return []
    try:
        answer = ProviderAnswer.model_validate(json.loads(raw, object_pairs_hook=_unique_object))
    except (ValueError, ValidationError, RecursionError):
        return []
    hints = []
    for citation in answer.citations:
        hit = sources.get(citation.source_id)
        if isinstance(citation, VisualEvidenceCitation) or not isinstance(hit, SearchHit):
            continue
        if citation.evidence in hit.text:
            continue
        words = citation.evidence.split()
        if not words:
            continue
        match = re.search(r"\s+".join(re.escape(word) for word in words), hit.text)
        if match is None:
            continue
        # A short single-line excerpt is easier to copy without PDF line joins.
        excerpt = max(match.group().splitlines(), key=len).strip()[:200]
        hint = {"source_id": citation.source_id, "evidence": excerpt}
        if excerpt and excerpt in hit.text and hint not in hints:
            hints.append(hint)
        if len(hints) == 5:
            break
    return hints


def citation_repair_schema(raw: str, sources, settings: RagSettings):
    """Constrain the one structured retry to locally verified citation pairs."""
    from .models import EvidenceCitation
    hints = citation_copy_hints(raw, sources, settings)
    if not hints:
        return None
    answer = ProviderAnswer.model_validate(json.loads(raw, object_pairs_hook=_unique_object))
    pairs = {}
    for hint in hints:
        pairs.setdefault(hint["source_id"], set()).add(hint["evidence"])
    for citation in answer.citations:
        hit = sources.get(citation.source_id)
        if (not isinstance(citation, VisualEvidenceCitation) and isinstance(hit, SearchHit)
                and citation.evidence.strip() and citation.evidence in hit.text):
            pairs.setdefault(citation.source_id, set()).add(citation.evidence)
    choices = []
    for source_id, quotes in sorted(pairs.items()):
        choices.append(create_model("ExactText" + source_id, __base__=EvidenceCitation,
            source_id=(Literal[source_id], ...), evidence=(Literal[tuple(sorted(quotes))], ...)))
    for source_id, source in sources.items():
        if isinstance(source, VisualSource):
            choices.append(create_model("ExactImage" + source_id, __base__=VisualEvidenceCitation,
                source_id=(Literal[source_id], ...)))
    citation_type = choices[0] if len(choices) == 1 else Union[tuple(choices)]
    return create_model("CitationRepairAnswer", __base__=ProviderAnswer,
        citations=(list[citation_type], Field(max_length=20)))
