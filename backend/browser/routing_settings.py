"""Opt-in capability routing; provider names remain configuration-owned."""
from pydantic import BaseModel, Field


class RoutingSettings(BaseModel):
    model_config = {"extra": "forbid"}
    enabled: bool = False
    deterministic_first: bool = False  # Deprecated compatibility input; initial routing is always model-first.
    fallback_provider: str = "local_router"
    fallback_timeout_seconds: float = Field(default=90, ge=1, le=3600)
    max_reroutes: int = Field(default=4, ge=0, le=10)
