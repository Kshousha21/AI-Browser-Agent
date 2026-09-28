"""Bounded ephemeral page images. PDFium supplements, never replaces, pypdf.

Only application-owned prepared paths enter this class. PDF identity is rehashed
before rendering; no URL fetching, file mutation or image persistence occurs.
"""
import asyncio
from dataclasses import dataclass, field
from hashlib import sha256
import math
from pathlib import Path
import struct
from threading import Event, Lock
import zlib

from pydantic import Field
from .models import LocalModel
from .store import DocumentStore

_PDFIUM_LOCK = Lock()  # PDFium is not thread-safe, even across documents.
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class RenderError(RuntimeError):
    def __init__(self, code="render_failed"):
        self.code = code if code in {"render_failed", "render_unavailable", "render_invalid_page",
            "render_document_unavailable", "render_identity_mismatch", "render_limit", "render_cancelled"} else "render_failed"
        super().__init__(self.code)


class RenderSettings(LocalModel):
    dpi: int = Field(default=120, ge=72, le=150)
    max_dimension: int = Field(default=1600, ge=256, le=2000)
    max_pixels: int = Field(default=2000000, ge=65536, le=4000000)
    max_image_bytes: int = Field(default=4 * 1024 * 1024, ge=1024, le=4 * 1024 * 1024)
    max_file_bytes: int = Field(default=25 * 1024 * 1024, ge=1024, le=25 * 1024 * 1024)


@dataclass(frozen=True, repr=False)
class RenderedPage:
    document_id: str
    page: int
    rendered_page_id: str
    width: int
    height: int
    png: bytes = field(repr=False)
    media_type: str = "image/png"


def _chunk(kind, data):
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def encode_png(bitmap, limits):
    """Encode our known RGB bitmap, not arbitrary input images; stream rows."""
    width, height = bitmap.width, bitmap.height
    if (bitmap.n_channels != 3 or bitmap.mode != "RGB" or width < 1 or height < 1
            or max(width, height) > limits.max_dimension or width * height > limits.max_pixels):
        raise RenderError("render_limit")
    compressor, pieces = zlib.compressobj(), []
    size = 0
    for row in range(height):
        encoded = compressor.compress(b"\0" + bytes(bitmap.buffer[row * bitmap.stride:row * bitmap.stride + width * 3]))
        size += len(encoded)
        if size > limits.max_image_bytes:
            raise RenderError("render_limit")
        pieces.append(encoded)
    pieces.append(compressor.flush())
    payload = PNG_SIGNATURE + _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    payload += _chunk(b"IDAT", b"".join(pieces)) + _chunk(b"IEND", b"")
    if len(payload) > limits.max_image_bytes:
        raise RenderError("render_limit")
    return payload


class LocalPageRenderer:
    def __init__(self, store: DocumentStore, path_for_document, settings=None):
        self.store, self.path_for_document = store, path_for_document
        self.settings = settings or RenderSettings()

    def info(self, document_id):
        return self.store.get(document_id)

    def page_text(self, document_id, page):
        pages = self.store.pages(document_id, limit=1, offset=page - 1)
        return pages[0].text if pages else ""

    async def render(self, document_id, page, selected_document_ids):
        if (document_id not in selected_document_ids or type(page) is not int
                or page < 1):
            raise RenderError("render_invalid_page")
        document = self.info(document_id)
        if document is None or page > document.page_count:
            raise RenderError("render_invalid_page")
        cancelled = Event()
        task = asyncio.create_task(asyncio.to_thread(self._render, document, page, cancelled))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled.set()
            # Consume a late failure; the worker owns all handles and always
            # closes them, including when cancellation occurs inside PDFium.
            task.add_done_callback(lambda done: done.exception() if not done.cancelled() else None)
            raise

    def _render(self, document, page_number, cancelled):
        pdf = page = bitmap = None
        try:
            path = self.path_for_document(document.document_id)
            if path is None:
                raise RenderError("render_document_unavailable")
            path = Path(path)
            if path.is_symlink() or not path.is_file() or path.stat().st_size > self.settings.max_file_bytes:
                raise RenderError("render_document_unavailable")
            with path.open("rb") as source:
                data = source.read(self.settings.max_file_bytes + 1)
            if len(data) > self.settings.max_file_bytes:
                raise RenderError("render_limit")
            if sha256(data).hexdigest() != document.document_id:
                raise RenderError("render_identity_mismatch")
            with _PDFIUM_LOCK:
                try:
                    if cancelled.is_set():
                        raise RenderError("render_cancelled")
                    import pypdfium2 as pdfium
                    import pypdfium2.raw as raw
                    pdf = pdfium.PdfDocument(data)
                    if len(pdf) != document.page_count:
                        raise RenderError("render_identity_mismatch")
                    page = pdf[page_number - 1]
                    width, height = page.get_size()
                    if not all(math.isfinite(value) and value > 0 for value in (width, height)):
                        raise RenderError("render_limit")
                    scale = min(self.settings.dpi / 72, (self.settings.max_dimension - 1) / max(width, height),
                        math.sqrt((self.settings.max_pixels - 4000) / (width * height)))
                    if scale <= 0:
                        raise RenderError("render_limit")
                    bitmap = page.render(scale=scale, force_bitmap_format=raw.FPDFBitmap_BGR, rev_byteorder=True)
                    if cancelled.is_set():
                        raise RenderError("render_cancelled")
                    png = encode_png(bitmap, self.settings)
                    identifier = sha256(document.document_id.encode() + str(page_number).encode() + png).hexdigest()
                    return RenderedPage(document.document_id, page_number, identifier, bitmap.width, bitmap.height, png)
                finally:
                    for handle in (bitmap, page, pdf):
                        if handle is not None:
                            handle.close()
        except ImportError:
            raise RenderError("render_unavailable") from None
        except RenderError:
            raise
        except Exception:
            raise RenderError() from None
