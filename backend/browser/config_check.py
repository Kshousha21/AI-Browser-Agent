"""Offline configuration checks with fixed, value-free diagnostic output.

Strict mode reuses the production loader and validates only the active provider.
--structure validates YAML without requiring workstation environment credentials.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from .config import (DEFAULT_CONFIG_PATH, _read_yaml, load_settings, ServerSettings,
                     AuthSettings, PageSettings, AgentSettings, ResearchSettings, RoutingSettings, VisualSettings, MultimodalSettings, SiteAdaptersSettings,
                     SecuritySettings, LoggingSettings)
from .provider_settings import ProviderProfile, ProviderSettings, SUPPORTED_TYPES

ROOT = Path(__file__).resolve().parents[2]


def scope_checks(root: Path = ROOT) -> bool:
    try:
        from .product_scope import product_url_allowed
        manifest = json.loads((root / "extension/public/manifest.json").read_text(encoding="utf-8"))
        samples = json.loads((root / "shared/scope-diagnostics.json").read_text(encoding="utf-8"))
        return ("activeTab" in manifest["permissions"] and "scripting" in manifest["permissions"]
                and "<all_urls>" not in manifest.get("host_permissions", [])
                and all(product_url_allowed(x["url"]) is x["allowed"] for x in samples))
    except (OSError, ValueError, KeyError, TypeError, ImportError):
        return False


def validate_structure(path: Path) -> None:
    raw = _read_yaml(path)
    if set(raw) - {"server", "auth", "provider", "providers", "page", "agent", "research", "routing", "visual_retrieval", "multimodal_retrieval", "site_adapters", "security", "logging"}:
        raise ValueError("Unknown configuration section")
    for name, model in [("server", ServerSettings), ("auth", AuthSettings), ("page", PageSettings),
                        ("visual_retrieval", VisualSettings), ("multimodal_retrieval", MultimodalSettings),
                        ("agent", AgentSettings), ("research", ResearchSettings), ("routing", RoutingSettings), ("site_adapters", SiteAdaptersSettings),
                        ("security", SecuritySettings), ("logging", LoggingSettings)]:
        model.model_validate(raw.get(name, {}))
    routing = RoutingSettings.model_validate(raw.get("routing", {}))
    if routing.enabled:
        router = ProviderProfile.model_validate(raw.get("providers", {})[routing.fallback_provider])
        if router.type != "ollama" or not router.local_only or router.auth.mode not in {"none", "native"}:
            raise ValueError("Routing requires a local-only Ollama profile")
    selection = raw.get("provider", {})
    if "active" in selection:
        if set(selection) - {"active"}:
            raise ValueError("Mixed canonical provider selection")
        profile = ProviderProfile.model_validate(raw.get("providers", {})[selection["active"]])
        if not profile.model and not profile.model_env and not (profile.type == "azure_openai" and profile.deployment_env):
            raise ValueError("Active model reference is missing")
    else:
        ProviderSettings.model_validate(selection)


def token_present(path: Path) -> bool:
    # Presence only. No settings or environment mapping is returned or logged.
    from dotenv import dotenv_values
    from contextlib import redirect_stderr
    import io
    with redirect_stderr(io.StringIO()):
        env = dotenv_values(path.parent / ".env")
    value = os.environ.get("BROWSER_AUTH__TOKEN") or env.get("BROWSER_AUTH__TOKEN")
    if not value:
        try:
            value = _read_yaml(path).get("auth", {}).get("token")
        except Exception:
            return False
    return bool(value)


def config_report(path: Path | None = None, *, structure: bool = False) -> dict:
    path = Path(path or os.environ.get("BROWSER_CONFIG", DEFAULT_CONFIG_PATH))
    report = {"valid": False, "mode": "structure" if structure else "runtime",
              "config_exists": path.is_file(), "product_scope": scope_checks(),
              "backend_token": "NOT CHECKED" if structure else "MISSING"}
    try:
        if not structure:
            report["backend_token"] = "PRESENT" if token_present(path) else "MISSING"
        if not path.is_file():
            raise ValueError("Missing configuration")
        if structure:
            validate_structure(path)
        else:
            # Production loader performs active-only environment, TLS-path, host,
            # port, timeout, and budget validation. It does not contact a provider.
            settings = load_settings(path)
            report.update(server={"host": settings.server.host, "port": settings.server.port},
                          active_provider=settings.provider.name if settings.provider.name in SUPPORTED_TYPES else "unknown",
                          backend_token="PRESENT" if settings.auth.token else "MISSING")
        report["valid"] = report["product_scope"]
        report["message"] = ("YAML structure valid; run doctor for environment readiness."
                             if structure else "Configuration valid; active provider configured offline.")
        if not report["product_scope"]:
            report["message"] = "URL validation or Chrome manifest is inconsistent."
    except Exception:
        # Never stringify parser/validation exceptions: YAML and custom validators
        # may embed source lines or rejected values, even with Pydantic input hidden.
        report["message"] = ("Configuration invalid or active-provider settings missing. Check YAML syntax, "
                             "active profile/required environment references, loopback host/port, limits and TLS file paths. "
                             "See README Windows quick start; values are never printed.")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--structure", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    report = config_report(args.config, structure=args.structure)
    if args.json:
        print(json.dumps(report))
    else:
        print(report["message"])
        print("Product scope: " + ("OK" if report["product_scope"] else "FAILED"))
        if not args.structure:
            print("Backend connection token: " + report["backend_token"])
            print("Active provider type: " + report.get("active_provider", "UNAVAILABLE"))
        print("No live services contacted.")
    raise SystemExit(0 if report["valid"] else 2)


if __name__ == "__main__":
    main()
