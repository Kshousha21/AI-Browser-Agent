"""Faithful normalization, complete character coverage and page provenance."""
import pytest
from pydantic import ValidationError

from local_pdf.models import DocumentPage, IngestionSettings
from local_pdf.text import chunk_pages, normalize_text

DOCUMENT_ID = "a" * 64


def page(text, number=1):
    return DocumentPage(document_id=DOCUMENT_ID, page_number=number, text=text,
                        text_status="text" if text else "empty")


def test_normalization_preserves_words_paragraphs_numbers_and_units():
    raw = " \tSynthetic  12.50 mm\r\n\r\n\r\n  300\tK; 2 µm; −5 °C\x00\x07\rLast-line. "
    expected = "Synthetic 12.50 mm\n\n300 K; 2 µm; −5 °C\nLast-line."
    assert normalize_text(raw) == expected
    assert normalize_text(expected) == expected
    assert normalize_text("   \r\n\x00\t") == ""


@pytest.mark.parametrize("overlap", [0, 16, 64])
@pytest.mark.parametrize("text", ["X" * 641, ("Invented paragraph of 12.5 mm. " * 5 + "\n\n") * 7])
def test_deterministic_bounded_chunks_reconstruct_each_page_with_exact_overlap(overlap, text):
    settings = IngestionSettings(max_chunk_chars=128, overlap_chars=overlap)
    pages = [page(text), page("", 2), page(text + " final", 3)]
    chunks = chunk_pages(pages, settings)
    assert chunks == chunk_pages(pages, settings)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert all(0 < len(c.text) <= 128 and c.document_id == DOCUMENT_ID for c in chunks)
    assert all(c.page_start == c.page_end and c.page_start != 2 for c in chunks)
    for source in (pages[0], pages[2]):
        local = [c for c in chunks if c.page_start == source.page_number]
        assert local[0].char_start == 0 and local[-1].char_end == len(source.text)
        rebuilt = local[0].text
        for left, right in zip(local, local[1:]):
            assert right.char_start == left.char_end - overlap
            if overlap:
                assert left.text[-overlap:] == right.text[:overlap]
            rebuilt += right.text[overlap:]
        assert rebuilt == source.text
        assert all(c.text == source.text[c.char_start:c.char_end] for c in local)


def test_paragraph_boundary_preferred_and_tiny_page_unsplit():
    settings = IngestionSettings(max_chunk_chars=128, overlap_chars=16)
    text = "A" * 90 + "\n\n" + "B" * 90
    assert chunk_pages([page(text)], settings)[0].text == "A" * 90 + "\n\n"
    assert len(chunk_pages([page("Tiny paragraph.\n\nAnother.")], settings)) == 1
    assert chunk_pages([page("")], settings) == []


def test_chunk_ids_change_with_configuration_or_page_position():
    text = "invented " * 100
    first = chunk_pages([page(text)], IngestionSettings(max_chunk_chars=128, overlap_chars=16))
    changed = chunk_pages([page(text)], IngestionSettings(max_chunk_chars=128, overlap_chars=32))
    moved = chunk_pages([page(text, 2)], IngestionSettings(max_chunk_chars=128, overlap_chars=16))
    assert first[0].chunk_id != changed[0].chunk_id != moved[0].chunk_id


@pytest.mark.parametrize("values", [
    {"max_chunk_chars": 127}, {"overlap_chars": -1}, {"overlap_chars": 2001},
    {"max_chunk_chars": 128, "overlap_chars": 65}, {"max_chunk_chars": 16001},
    {"max_file_bytes": 0}, {"max_file_bytes": 101 * 1024 * 1024},
    {"max_pages": True}, {"unknown_setting": 1},
])
def test_settings_are_strict_and_overlap_cannot_prevent_progress(values):
    with pytest.raises(ValidationError):
        IngestionSettings(**values)
