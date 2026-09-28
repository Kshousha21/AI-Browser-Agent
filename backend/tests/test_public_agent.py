"""Public-site contracts: DOM tools, deletion, local routing and file provenance."""
import json
import pytest
from app.site_adapters import SiteAdapterRegistry
from app.site_adapters.generic import GenericSiteAdapter
from app.site_adapters.page_files import retrieve
from browser.actions import ActionResult
from browser.agent import run_agent
from browser.agent_runtime import available_tools
from browser.config import AgentSettings, SecuritySettings, SiteAdaptersSettings
from browser.protocol import PageState, PageElement
from browser.providers import MockProvider, ToolCall
from browser.routing_settings import RoutingSettings
from browser.tool_routing import RoutingMemory, RouteController, choose_route
from browser.tool_profiles import Route
from browser.product_scope import product_url_allowed
from browser.security import ActionClassifier, PermissionManager
from document_workflow.models import PrepareRequest, DocumentError
from document_workflow.selection import CurrentSelection, resolve_attachment
from document_workflow.session import DocumentSession
from .test_agent import FakeBridge
from .public_document_support import services, settings, offer, request, Bridge

def page(url="https://example.org/docs"):
    return PageState(url=url, tab_id=1, document_id="d", snapshot_id="s", title="Public handbook",
        visible_text="The public library opens at nine.", elements=[
            PageElement(agent_id="f0:e-001", tag="input", role="searchbox", name="Search"),
            PageElement(agent_id="f0:e-002", tag="button", role="button", name="Search"),
            PageElement(agent_id="f0:e-003", tag="a", role="link", name="Guide", href="https://other.example/guide"),
            PageElement(agent_id="f0:e-004", tag="button", role="button", name="Delete file")])

def call(tool, args=None, snapshot="s"):
    return ToolCall(tool=tool, arguments=args or {}, snapshot_id=snapshot, reason="User request")

def observed(p, action="get_page_state"):
    return ActionResult(success=True, action=action, snapshot_id=p.snapshot_id, data={"page_state":p.model_dump()})

@pytest.mark.parametrize("url", ["https://example.org/docs", "http://news.example:8080/article", "https://shop.example/product"])
async def test_generic_summary_and_no_site_tools(url):
    current=page(url); registry=SiteAdapterRegistry(SiteAdaptersSettings())
    assert isinstance(registry.adapter_for(url), GenericSiteAdapter)
    assert {t.name for t in registry.tool_schemas(url)} == {"get_page_files"}
    provider=MockProvider(scripted=["The public library opens at nine."])
    bridge=FakeBridge([observed(current)])
    result=await run_agent("Summarize this page",current,[],provider,AgentSettings(),bridge,
        site_adapters=registry,security_settings=SecuritySettings())
    assert result.text=="The public library opens at nine."
    assert "public library" in repr(provider.calls)
    assert not bridge.confirmation_scopes

@pytest.mark.parametrize("tool,args", [("click",{"element_id":"f0:e-003"}),
    ("type_text",{"element_id":"f0:e-001","text":"libraries","submit":True}),
    ("scroll",{"direction":"down"}), ("navigate",{"url":"https://other.example/guide"}),
    ("go_back",{}),("go_forward",{}),("refresh",{})])
async def test_normal_dom_actions_do_not_confirm(tool,args):
    current=page(); operation=call(tool,args)
    bridge=FakeBridge([observed(current),observed(current,tool),observed(current),observed(current)])
    result=await run_agent("Find libraries and open https://other.example/guide",current,[],
        MockProvider(scripted=[operation,"Done"]),AgentSettings(),bridge,security_settings=SecuritySettings())
    assert operation.tool in [c.tool for c in bridge.calls]
    assert not bridge.confirmation_scopes
    assert result.text=="Done"

async def test_search_uses_observed_input_and_button():
    current=page(); found=page();found.visible_text="Library search found Central Library."
    bridge=FakeBridge([observed(current),observed(current,"type_text"),observed(found,"click"),observed(found),observed(found)])
    provider=MockProvider(scripted=[call("type_text",{"element_id":"f0:e-001","text":"library"}),
        call("click",{"element_id":"f0:e-002"}),"Central Library"])
    result=await run_agent("Search for library",current,[],provider,AgentSettings(),bridge,
        security_settings=SecuritySettings(),site_adapters=SiteAdapterRegistry(SiteAdaptersSettings()))
    assert [c.tool for c in bridge.calls][:3]==["get_page_state","type_text","click"]
    assert result.text=="Central Library" and not bridge.confirmation_scopes

async def test_delete_has_exactly_one_confirmation_and_revalidates():
    current=page(); fresh=current.model_copy(update={"snapshot_id":"fresh"})
    bridge=FakeBridge([observed(current),observed(fresh),observed(fresh,"click"),observed(fresh),observed(fresh)],confirmation_choices=["allow_once"])
    result=await run_agent("Delete file",current,[],MockProvider(scripted=[call("click",{"element_id":"f0:e-004"}),"Deleted"]),
        AgentSettings(),bridge,security_settings=SecuritySettings())
    assert len(bridge.confirmation_scopes)==1
    assert [c.tool for c in bridge.calls][:3]==["get_page_state","get_page_state","click"]
    assert bridge.calls[2].snapshot_id=="fresh" and result.text=="Deleted"

async def test_stale_dom_recovers_automatically_without_replay():
    current=page(); fresh=current.model_copy(update={"snapshot_id":"fresh"})
    bridge=FakeBridge([observed(current),ActionResult(success=False,action="click",snapshot_id="s",error="stale_snapshot"),
        observed(fresh),observed(fresh,"click"),observed(fresh),observed(fresh)])
    result=await run_agent("Open Guide",current,[],MockProvider(scripted=[call("click",{"element_id":"f0:e-003"}),
        call("click",{"element_id":"f0:e-003"},"fresh"),"Done"]),AgentSettings(),bridge,security_settings=SecuritySettings())
    assert result.text=="Done" and not bridge.confirmation_scopes
    assert [c.tool for c in bridge.calls][:4]==["get_page_state","click","get_page_state","click"]
    assert bridge.calls[3].snapshot_id=="fresh"

@pytest.mark.parametrize("url", ["chrome://settings", "chrome-extension://abc/page", "file:///private.pdf", "about:blank", "javascript:alert(1)"])
async def test_internal_pages_are_unsupported_without_model_call(url):
    assert not product_url_allowed(url)
    registry=SiteAdapterRegistry(SiteAdaptersSettings());current=page(url)
    state=RoutingMemory().state(current,available_tools(url,registry),adapters=registry)
    router=MockProvider(scripted=['{"route":"READ"}'])
    result,_,_=await choose_route("Summarize",state,RoutingSettings(),router)
    assert result.route==Route.UNSUPPORTED and not router.calls

async def test_local_router_schema_and_main_capabilities_are_separate():
    registry=SiteAdapterRegistry(SiteAdaptersSettings());current=page()
    tools=available_tools(current.url,registry)
    state=RoutingMemory().state(current,tools,adapters=registry)
    router=MockProvider(scripted=['{"route":"SEARCH"}'])
    decision,_,_=await choose_route("Search for library",state,RoutingSettings(),router)
    assert decision.route==Route.SEARCH
    names={t.name for t in RouteController(decision.route).tools(tools,registry.generic)}
    assert {"click","type_text","press_key"} <= names
    assert "acquire_attachment" not in names

async def test_2b_router_and_main_ollama_provider_wire_contract():
    """Real provider serialization/validation with deterministic loopback transport."""
    import httpx
    from browser.providers.ollama_provider import OllamaProvider
    received=[]
    def wire(request):
        body=json.loads(request.content); received.append(body)
        assert request.url.host=="127.0.0.1"
        content='{"route":"SEARCH"}' if body['model']=='qwen3.5:2b' else json.dumps({
            'kind':'tool','tool':'type_text','arguments':{'element_id':'f0:e-001','text':'library',
                'snapshot_id':'s','reason':'Search the observed input'}})
        return httpx.Response(200,json={'done':True,'done_reason':'stop','message':{'role':'assistant','content':content}})
    client=httpx.AsyncClient(transport=httpx.MockTransport(wire))
    router=OllamaProvider(model='qwen3.5:2b',http_client=client)
    main=OllamaProvider(model='qwen3.5:9b',tool_mode='structured',http_client=client)
    registry=SiteAdapterRegistry(SiteAdaptersSettings());current=page();tools=available_tools(current.url,registry)
    try:
        choice,_,_=await choose_route('Search for library',RoutingMemory().state(current,tools,adapters=registry),RoutingSettings(),router)
        from browser.prompting import build_messages
        controller=RouteController(choice.route)
        response=await main.generate(build_messages('Search for library',current,registry.system_prompt_addendum(current.url)),
            tools=controller.tools(tools,registry.generic))
        assert response.decision.tool=='type_text'
        assert response.decision.arguments['element_id']=='f0:e-001'
        assert [r['model'] for r in received]==['qwen3.5:2b','qwen3.5:9b']
        assert not received[0].get('tools')
    finally: await client.aclose()

async def test_generic_pdf_prepare_ask_and_citation(tmp_path):
    service,provider=services(tmp_path/'local')
    session=DocumentSession(settings(),lambda:service)
    _,current,payload=offer(session.selection)
    bridge=Bridge(tmp_path/'downloads')
    await session.run('document_prepare',request(payload,question="How much power does the synthetic photonics package use?"),bridge)
    assert bridge.done['status'] in {'ready','answered'}, bridge.done
    assert session.ready
    # Exercise the normal prepared-document follow-up path and citations.
    identifier=next(iter(session.ready))
    followup=Bridge(tmp_path/'downloads');followup.target=current
    await session.run('document_question',{'document_ids':[identifier], 'question':'How much power does the synthetic photonics package use?',
        'allow_remote_context':False},followup)
    assert followup.done.get('answer'), followup.done
    assert followup.done['answer']['citations']

def test_current_file_ref_rejects_changed_owner_or_name():
    selection,current,payload=offer()
    ref=payload['results'][0]['result_ref']
    assert resolve_attachment(current,ref,selection.refs,[]).url.endswith('.pdf')
    changed=current.model_copy(update={'tab_id':2})
    with pytest.raises(DocumentError): resolve_attachment(changed,ref,selection.refs,[])

def test_manifest_is_minimal_and_persistent_access_optional():
    from pathlib import Path
    root=Path(__file__).resolve().parents[2]
    manifest=json.loads((root/'extension/public/manifest.json').read_text())
    assert set(manifest['permissions'])=={'activeTab','scripting','sidePanel','storage','downloads','declarativeNetRequest'}
    assert '<all_urls>' not in manifest.get('host_permissions',[])
    assert set(manifest['optional_host_permissions'])=={'http://*/*','https://*/*'}
    assert not manifest.get('content_scripts')


def test_file_selection_resolves_second_window_and_binds_exact_filename():
    current = page()
    current.elements = [PageElement(agent_id=f"f0:e-{i:03}", tag="a", role="link", name=f"Manual {i}",
        href=f"https://example.org/files/manual-{i}.pdf") for i in range(12)]
    refined = GenericSiteAdapter().refine_page_state(current)
    result = retrieve(call("get_page_files", {"offset": 10, "limit": 2}), refined)
    selection = CurrentSelection()
    payload = selection.offer(refined, result)
    assert payload is not None
    chosen = resolve_attachment(refined, payload["results"][1]["result_ref"], selection.refs, [])
    assert chosen.index == 11
    assert chosen.url == "https://example.org/files/manual-11.pdf"
