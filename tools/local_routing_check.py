"""Opt-in synthetic local routing check; never downloads or executes browser actions."""
import argparse
import asyncio
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))


async def check(args):
    from browser.config import load_provider_settings, SiteAdaptersSettings
    from browser.providers import build_provider
    from browser.tool_routing import choose_route, RouteState, RouteController
    from browser.tool_profiles import Route
    from browser.routing_settings import RoutingSettings
    from browser.agent_runtime import available_tools
    from browser.prompting import build_messages
    from browser.protocol import PageState, PageElement
    from app.site_adapters import SiteAdapterRegistry

    router_settings = load_provider_settings(args.config, args.router_profile)
    main_settings = load_provider_settings(args.config, args.main_profile)
    if any(s.name != "ollama" or not s.local_only for s in [router_settings, main_settings]):
        raise ValueError("Both smoke-test profiles must be local-only Ollama")
    router, main = build_provider(router_settings), build_provider(main_settings)
    try:
        started = time.monotonic()
        choice, _, source = await choose_route("show me more", RouteState(
            page_kind="search_results", search_available=True, has_search_results=True,
            previous_route=Route.SEARCH), RoutingSettings(enabled=True), router)
        classification_ok = choice.route == Route.SEARCH and source == "local_router"
        print(json.dumps({"check":"local_classifier", "passed":classification_ok,
            "route":choice.route.value, "seconds":round(time.monotonic()-started, 2)}), flush=True)
        page = PageState(url="https://example.org/research",
            snapshot_id="synthetic-snapshot", document_id="synthetic-document",
            title="Synthetic fixture", visible_text="Synthetic test page. No private content.",
            elements=[PageElement(agent_id="f0:e-001",tag="input",role="searchbox",name="Search")])
        registry = SiteAdapterRegistry(SiteAdaptersSettings())
        controller = RouteController(Route.SEARCH)
        schemas = controller.tools(available_tools(page.url, registry),
                                   adapter=registry.adapter_for(page.url), page_kind="page_view")
        messages = build_messages("Search for AI research", page, registry.system_prompt_addendum(page.url))
        messages.insert(1, controller.instruction())
        started = time.monotonic()
        reply = await main.generate(messages, tools=schemas)
        dispatch_ok = (reply.decision.type == "tool" and reply.decision.tool == "type_text"
                       and reply.decision.snapshot_id == page.snapshot_id
                       and {"click", "type_text", "press_key"} <= {tool.name for tool in schemas})
        print(json.dumps({"check":"local_main_filtered_dispatch", "passed":dispatch_ok,
            "tools_exposed":len(schemas), "decision_kind":reply.decision.type,
            "selected_tool":(reply.decision.tool if reply.decision.type == "tool" and reply.decision.tool in {t.name for t in schemas} else None),
            "snapshot_matches":reply.decision.type == "tool" and reply.decision.snapshot_id == page.snapshot_id,
            "seconds":round(time.monotonic()-started, 2)}), flush=True)
        return classification_ok and dispatch_ok
    finally:
        await router.aclose()
        await main.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--config", type=Path, default=ROOT / "backend/config.yaml")
    parser.add_argument("--main-profile", default="local_agent")
    parser.add_argument("--router-profile", default="local_router")
    args = parser.parse_args()
    if not args.run:
        parser.error("Pass --run to contact only the configured local models using synthetic data")
    try:
        passed = asyncio.run(check(args))
    except Exception:
        print("Local routing smoke failed; no provider response or private configuration is printed.")
        passed = False
    raise SystemExit(0 if passed else 1)
