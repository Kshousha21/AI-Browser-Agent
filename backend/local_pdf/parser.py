"""Bounded, strict local extraction; parser diagnostics never leave this module."""
from contextlib import contextmanager
from io import BytesIO
import logging
import os
from pathlib import Path
import re
import stat
from threading import RLock
import warnings

from pypdf import PdfReader, __version__ as PARSER_VERSION
from pypdf.constants import UserAccessPermissions
from pypdf.errors import DependencyError, WrongPasswordError

from .errors import PdfIngestError
from .models import DocumentPage, IngestionSettings
from .text import normalize_text

_PARSER_LOCK = RLock()


class _RejectDiagnostics(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        # The pinned strict reader accepts redundant header whitespace, then
        # checks both object number and generation against the xref. Throwing
        # here turns that harmless diagnostic into a repair fallback. Admit only
        # this exact structured warning; actual repairs/other warnings still fail.
        if (record.levelno == logging.WARNING and record.name == "pypdf._reader"
                and record.msg == "Superfluous whitespace found in object header %(idnum)r %(generation)r"
                and isinstance(record.args, dict) and set(record.args) == {"idnum", "generation"}
                and all(isinstance(value, bytes) and re.fullmatch(rb"[0-9]{1,20}", value)
                        for value in record.args.values())):
            return
        if record.levelno >= logging.WARNING:
            raise PdfIngestError("pdf_parse_failed")


@contextmanager
def _private_parser():
    """Fail on repair warnings without emitting their potentially private text.

    Ingestion serializes its parser calls. Save/restore the parser logger tree;
    do not change browser, audit or root handlers. No other subsystem uses pypdf.
    """
    with _PARSER_LOCK:
        parent = logging.getLogger("pypdf")
        loggers = [parent] + [item for name, item in logging.Logger.manager.loggerDict.copy().items()
                              if name.startswith("pypdf.") and isinstance(item, logging.Logger)]
        saved = [(log, log.handlers[:], log.level, log.propagate, log.disabled) for log in loggers]
        try:
            for log in loggers:
                log.handlers = []
                log.setLevel(logging.NOTSET)
                log.propagate = True
                log.disabled = False
            parent.handlers = [_RejectDiagnostics()]
            parent.setLevel(logging.WARNING)
            parent.propagate = False
            with warnings.catch_warnings():
                warnings.filterwarnings("error", module=r"^pypdf(?:\.|$)")
                yield
        finally:
            for log, handlers, level, propagate, disabled in saved:
                log.handlers = handlers
                log.setLevel(level)
                log.propagate = propagate
                log.disabled = disabled


def read_pdf(path: Path, settings: IngestionSettings) -> bytes:
    """Read once from a regular file; hash and parse the same bounded bytes."""
    try:
        if not path.exists():
            raise PdfIngestError("pdf_not_found")
        if path.suffix.lower() != ".pdf" or not path.is_file():
            raise PdfIngestError("invalid_pdf")
        with path.open("rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode):
                raise PdfIngestError("invalid_pdf")
            if info.st_size > settings.max_file_bytes:
                raise PdfIngestError("pdf_too_large")
            data = stream.read(settings.max_file_bytes + 1)
        if len(data) > settings.max_file_bytes:
            raise PdfIngestError("pdf_too_large")
        if not re.match(rb"%PDF-(?:1\.[0-7]|2\.0)[\r\n]", data):
            raise PdfIngestError("invalid_pdf")
        if not data.rstrip().endswith(b"%%EOF"):
            raise PdfIngestError("pdf_parse_failed")
        return data
    except OSError:
        raise PdfIngestError("pdf_read_failed") from None


def extract_pages(data: bytes, document_id: str, settings: IngestionSettings) -> list[DocumentPage]:
    """Extract all pages before storage, with explicit page/text/stream limits."""
    try:
        with _private_parser():
            # Published PDFs can use encryption for permissions while requiring
            # no password to open. Try only the empty opening password; never
            # collect, guess or persist a user/owner password. is_encrypted stays
            # True even after successful opening, so it is not a failure check.
            reader = PdfReader(BytesIO(data), strict=True)
            if reader.is_encrypted:
                # The default constructor attempts the empty password once.
                # None means that this attempt did not unlock the document.
                if reader.are_permissions_valid is None:
                    raise PdfIngestError("pdf_encrypted")
                if reader.are_permissions_valid is not True:
                    raise PdfIngestError("pdf_parse_failed")
                permissions = reader.user_access_permissions
                if permissions is None or not permissions & UserAccessPermissions.EXTRACT:
                    raise PdfIngestError("pdf_extraction_not_allowed")
            count = len(reader.pages)
            if count < 1:
                raise PdfIngestError("invalid_pdf")
            if count > settings.max_pages:
                raise PdfIngestError("pdf_limit_exceeded")
            pages: list[DocumentPage] = []
            total = 0
            for number, page in enumerate(reader.pages, start=1):
                content = page.get_contents()
                if content is not None and len(content.get_data()) > settings.max_page_stream_bytes:
                    raise PdfIngestError("pdf_limit_exceeded")
                raw = page.extract_text(extraction_mode="plain")
                if len(raw) > settings.max_page_chars:
                    raise PdfIngestError("pdf_limit_exceeded")
                total += len(raw)
                if total > settings.max_total_chars:
                    raise PdfIngestError("pdf_limit_exceeded")
                text = normalize_text(raw)
                pages.append(DocumentPage(document_id=document_id, page_number=number,
                                          text=text, text_status="text" if text else "empty"))
            return pages
    except PdfIngestError:
        raise
    except WrongPasswordError:
        raise PdfIngestError("pdf_encrypted") from None
    except DependencyError:
        raise PdfIngestError("pdf_crypto_unavailable") from None
    except Exception:
        # Never include parser exception text, metadata, paths, or a traceback.
        raise PdfIngestError("pdf_parse_failed") from None
