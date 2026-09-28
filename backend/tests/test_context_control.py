"""PageState full/diff behavior and context-size fallback tests."""

from __future__ import annotations

from browser.context_control import (
    diff_page_payload,
    full_page_payload,
    page_observation,
    serialized_size,
)
from browser.protocol import PageElement, PageHeading, PageState


def state(**updates: object) -> PageState:
    base = PageState(
        snapshot_id="s-1",
        document_id="d-1",
        navigation_id=0,
        url="https://example.test/jobs",
        title="Jobs",
        visible_text="First region\nSecond region",
        headings=[PageHeading(level=1, text="Jobs")],
        elements=[
            PageElement(
                agent_id="f0:e-001",
                tag="button",
                role="button",
                name="Search",
                semantic_region="main",
            ),
            PageElement(
                agent_id="f0:e-002",
                tag="a",
                role="link",
                name="Open",
                href="https://example.test/job/1",
                semantic_region="results",
            ),
        ],
    )
    return base.model_copy(update=updates)


def test_first_observation_is_explicit_full_snapshot() -> None:
    payload = page_observation(None, state())
    assert payload["type"] == "page_state_full"
    assert payload["full_snapshot_reason"] == "initial"


def test_full_payload_omits_empty_defaults_and_tracking_heavy_urls() -> None:
    page = state(
        elements=[
            PageElement(
                agent_id="f0:e-009",
                tag="a",
                role="link",
                name="Open job",
                href="https://example.test/job/9?" + ("tracking=x&" * 80),
            )
        ]
    )
    element = full_page_payload(page)["elements"][0]
    assert element == {
        "agent_id": "f0:e-009",
        "tag": "a",
        "role": "link",
        "name": "Open job",
        "href": "https://example.test/job/9",
    }


def test_hidden_element_payload_withholds_content_without_default_field_bloat() -> None:
    page = state(
        elements=[
            PageElement(
                agent_id="f0:e-010",
                tag="button",
                role="button",
                name="hidden canary",
                text="hidden canary",
                visible=False,
            )
        ]
    )
    element = full_page_payload(page)["elements"][0]
    assert "hidden canary" not in str(element)
    assert element["flags"] == ["hidden; content withheld"]
    assert "enabled" not in element


def test_same_page_dom_mutation_becomes_explicit_diff() -> None:
    stable = [f"Stable region {index}" for index in range(40)]
    current = state(
        snapshot_id="s-2",
        visible_text="\n".join([*stable, "Changed second region"]),
    )
    previous = state(visible_text="\n".join([*stable, "Original second region"]))
    payload = page_observation(previous, current)
    assert payload["type"] == "page_state_diff"
    assert payload["base_snapshot_id"] == "s-1"
    assert payload["snapshot_id"] == "s-2"
    assert payload["url_changed"] is False
    assert payload["text"]["changed_regions"]


def test_document_navigation_forces_a_full_snapshot() -> None:
    current = state(
        snapshot_id="s-2",
        document_id="d-2",
        url="https://example.test/results",
    )
    payload = page_observation(state(), current)
    assert payload["type"] == "page_state_full"
    assert payload["full_snapshot_reason"] == "document_changed"


def test_missing_lineage_forces_a_full_snapshot() -> None:
    payload = page_observation(
        state(document_id=""), state(snapshot_id="s-2", document_id="")
    )
    assert payload["type"] == "page_state_full"
    assert payload["full_snapshot_reason"] == "missing_page_identity"


def test_explicit_full_request_forces_a_full_snapshot() -> None:
    payload = page_observation(state(), state(snapshot_id="s-2"), force_full=True)
    assert payload["type"] == "page_state_full"
    assert payload["full_snapshot_reason"] == "navigation_or_explicit_request"


def test_diff_larger_than_half_the_full_payload_falls_back_to_full() -> None:
    many = [
        PageElement(
            agent_id=f"f0:e-{index:03d}",
            tag="a",
            role="link",
            name=f"Job {index}",
            href=f"https://example.test/job/{index}",
            semantic_region=f"job-{index}",
        )
        for index in range(10, 110)
    ]
    payload = page_observation(
        state(),
        state(snapshot_id="s-2", elements=many, visible_text="Completely new"),
    )
    assert payload["type"] == "page_state_full"
    assert payload["full_snapshot_reason"] == "diff_too_large"


def test_added_removed_and_small_changed_elements_are_structured() -> None:
    current = state(
        snapshot_id="s-2",
        elements=[
            state().elements[0].model_copy(update={"enabled": False}),
            PageElement(
                agent_id="f0:e-003",
                tag="a",
                role="link",
                name="New",
                href="https://example.test/job/3",
                semantic_region="results",
            ),
        ],
    )
    delta = diff_page_payload(state(), current)["elements"]
    assert [item["agent_id"] for item in delta["added"]] == ["f0:e-003"]
    assert [item["agent_id"] for item in delta["removed"]] == ["f0:e-002"]
    assert delta["changed"][0]["changes"]["enabled"] == {
        "before": True,
        "after": False,
    }


def test_heading_warning_and_metadata_changes_are_reported() -> None:
    current = state(
        snapshot_id="s-2",
        headings=[PageHeading(level=1, text="Updated jobs")],
        warnings=["security:prompt_injection_pattern"],
        looks_like_login=True,
    )
    delta = diff_page_payload(state(), current)
    assert delta["headings"]["added"][0]["text"] == "Updated jobs"
    assert delta["warnings"]["added"] == ["security:prompt_injection_pattern"]
    assert delta["metadata"]["looks_like_login"]["after"] is True


def test_budgeted_full_payload_is_smaller_than_raw_page_json() -> None:
    page = state(
        visible_text="\n".join(f"region {index}" for index in range(4_000))
    )
    payload = full_page_payload(page)
    assert serialized_size(payload) < len(page.model_dump_json())
    assert payload["truncated"]["text"] is True


def test_adapter_budget_keeps_priority_controls_and_caps_generic_fallback() -> None:
    controls = [
        PageElement(
            agent_id=f"f0:e-{index:03d}",
            tag="button",
            role="button",
            name="Search" if index == 0 else f"Control {index}",
        )
        for index in range(60)
    ]
    page = state(
        elements=controls,
        site_context={
            "site": "local_document_fixture",
            "page_kind": "search_results",
            "results": [],
        },
    )
    sent = full_page_payload(page)["elements"]
    assert len(sent) == 32
    assert sent[0]["name"] == "Search"


def test_nested_adapter_result_append_has_a_structured_diff() -> None:
    previous = state(
        site_context={
            "site": "generic_portal",
            "page_kind": "search_drawer",
            "search": {"results": [{"title": "Result A", "url": "/a"}]},
        }
    )
    current = previous.model_copy(
        update={
            "snapshot_id": "s-2",
            "site_context": {
                "site": "generic_portal",
                "page_kind": "search_drawer",
                "search": {
                    "results": [
                        {"title": "Result A", "url": "/a"},
                        {"title": "Result B", "url": "/b"},
                    ]
                },
            },
        }
    )

    delta = diff_page_payload(previous, current)["site_context"]

    assert delta["search"]["results"]["added"][0]["title"] == "Result B"


def test_generic_file_inventory_is_summary_first():
    page=state(site_context={"adapter":"GenericSiteAdapter","files":[{"url":"https://example.org/report.pdf","title":"x"*500} for _ in range(200)]})
    context=full_page_payload(page)["site_context"]
    assert context["file_count"]==200
    assert context["file_metadata_tool"]=="get_page_files"
    assert "files" not in context and "report.pdf" not in str(context)
    assert len(str(context))<500
