"""Real local rasterization with generated public-safe fixtures; no providers."""
import asyncio
from hashlib import sha256
from pathlib import Path
import struct
from threading import Event
import zlib
import pytest
from local_pdf import DocumentStore, PdfIngestionService
from local_pdf.rendering import LocalPageRenderer, RenderError, RenderSettings, PNG_SIGNATURE
from .visual_pdf_fixtures import research_pdf


def prepared(tmp_path, kind="chart"):
    path = tmp_path / (kind + ".pdf")
    path.write_bytes(research_pdf(kind))
    store = DocumentStore(tmp_path / "documents.sqlite3")
    metadata = PdfIngestionService(store).ingest_pdf(path).document
    renderer = LocalPageRenderer(store, lambda identifier: path if identifier == metadata.document_id else None)
    return renderer, metadata, path


@pytest.mark.parametrize("kind,page", [("chart", 4), ("diagram", 2), ("table", 3), ("image_only", 1),
    ("conflict", 4), ("illegible", 4)])
async def test_render_generated_selected_page_bounded_immutable_ephemeral(tmp_path, kind, page):
    renderer, metadata, path = prepared(tmp_path, kind)
    before = sha256(path.read_bytes()).hexdigest()
    files = set(tmp_path.iterdir())
    result = await renderer.render(metadata.document_id, page, [metadata.document_id])
    assert result.page == page and result.document_id == metadata.document_id
    assert result.png.startswith(PNG_SIGNATURE)
    assert struct.unpack(">II", result.png[16:24]) == (result.width, result.height)
    assert result.width * result.height <= 2000000 and max(result.width, result.height) <= 1600
    assert len(result.png) <= 4 * 1024 * 1024 and len(result.rendered_page_id) == 64
    assert sha256(path.read_bytes()).hexdigest() == before and set(tmp_path.iterdir()) == files
    assert not hasattr(result, "path") and str(path) not in repr(result)
    if kind == "image_only":
        assert metadata.extracted_chars == metadata.chunk_count == 0


@pytest.mark.parametrize("page", [0, 5, -1, "4", True, None])
async def test_page_bounds_before_native_render(tmp_path, page):
    renderer, metadata, _ = prepared(tmp_path)
    with pytest.raises(RenderError, match="render_invalid_page"):
        await renderer.render(metadata.document_id, page, [metadata.document_id])


async def test_selected_document_and_changed_bytes_fail_closed(tmp_path):
    renderer, metadata, path = prepared(tmp_path)
    with pytest.raises(RenderError, match="render_invalid_page"):
        await renderer.render(metadata.document_id, 4, ["a" * 64])
    path.write_bytes(research_pdf("conflict"))
    with pytest.raises(RenderError, match="render_identity_mismatch"):
        await renderer.render(metadata.document_id, 4, [metadata.document_id])


async def test_cancelled_native_work_does_not_publish_or_create_artifacts(tmp_path, monkeypatch):
    renderer, metadata, _ = prepared(tmp_path)
    started, finished = asyncio.Event(), asyncio.Event()
    release = Event()
    loop = asyncio.get_running_loop()
    original = renderer._render
    def operation(document, page, cancelled):
        loop.call_soon_threadsafe(started.set)
        release.wait()
        try:
            return original(document, page, cancelled)
        finally:
            loop.call_soon_threadsafe(finished.set)
    monkeypatch.setattr(renderer, "_render", operation)
    task = asyncio.create_task(renderer.render(metadata.document_id, 4, [metadata.document_id]))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    release.set()
    await finished.wait()
    assert not list(tmp_path.glob("*.png"))


def test_renderer_fresh_install_pin_and_hard_settings():
    requirements = (Path(__file__).parents[1] / "requirements.txt").read_text()
    assert "pypdfium2==5.13.0" in requirements
    import importlib.metadata
    assert importlib.metadata.version("pypdfium2") == "5.13.0"
    for kwargs in ({"dpi": 151}, {"max_dimension": 2001}, {"max_pixels": 4000001}, {"max_image_bytes": 5*1024*1024}):
        with pytest.raises(ValueError):
            RenderSettings(**kwargs)
