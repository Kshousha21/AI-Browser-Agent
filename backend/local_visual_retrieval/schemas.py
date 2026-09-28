"""Closed local configuration and representation contracts."""
from typing import Literal
from pydantic import Field, model_validator
from local_pdf.models import LocalModel
from local_retrieval.models import Digest, fingerprint

CODES = {"visual_embedding_model_unavailable", "visual_index_unavailable", "visual_index_incompatible",
    "visual_index_corrupt", "visual_index_write_failed", "visual_invalid_request", "visual_source_changed",
    "visual_render_failed", "visual_embedding_failed", "visual_capacity_exceeded", "visual_disabled"}

class VisualError(RuntimeError):
    def __init__(self, code):
        self.code = code if code in CODES else "visual_embedding_failed"
        super().__init__(self.code)

class RetrievalConfig(LocalModel):
    @model_validator(mode="before")
    @classmethod
    def environment_scalars(cls, value):
        # Pydantic Settings supplies nested environment leaves as strings.
        import json
        if not isinstance(value, dict): return value
        result = dict(value)
        scalar_fields = {"enabled", "local_only", "timeout_seconds", "max_index_pages", "always_use_text",
                         "always_use_visual", "text_top_k", "visual_top_k", "fused_top_k", "max_visual_pages"}
        for key in scalar_fields & result.keys():
            if isinstance(result[key], str):
                try: result[key] = json.loads(result[key])
                except ValueError: pass
        return result

class VisualSettings(RetrievalConfig):
    enabled: bool = True
    model: Literal["nomic-ai/nomic-embed-multimodal-3b"] = "nomic-ai/nomic-embed-multimodal-3b"
    local_only: Literal[True] = True
    model_directory: str | None = Field(default=None, repr=False)
    base_model_directory: str | None = Field(default=None, repr=False)
    revision: str = "main"
    python_executable: str | None = Field(default=None, repr=False)
    device: Literal["auto", "cpu", "cuda"] = "auto"
    timeout_seconds: float = Field(default=180.0, ge=1, le=600)
    max_index_pages: int = Field(default=10000, ge=1, le=50000)

class MultimodalSettings(RetrievalConfig):
    enabled: bool = True
    always_use_text: Literal[True] = True
    always_use_visual: Literal[True] = True
    text_top_k: int = Field(default=5, ge=1, le=10)
    visual_top_k: int = Field(default=5, ge=1, le=20)
    fused_top_k: int = Field(default=5, ge=1, le=10)
    max_visual_pages: int = Field(default=3, ge=1, le=5)
    fusion: Literal["rrf"] = "rrf"

class VisualIdentity(LocalModel):
    model: str
    revision: str
    artifact_fingerprint: Digest
    processor: str
    runtime: str
    dimension: int = Field(ge=1, le=8192)
    representation: Literal["biqwen-last-l2-f32-v1"] = "biqwen-last-l2-f32-v1"
    renderer: str
    device_policy: str = "auto"
    @property
    def key(self):
        return fingerprint(self.model_dump())

class VisualHit(LocalModel):
    document_id: Digest
    source_hash: Digest
    prepared_identity: Digest
    page: int = Field(ge=1)
    rendered_page_id: Digest
    rank: int = Field(ge=1)
    score: float = Field(allow_inf_nan=False)
