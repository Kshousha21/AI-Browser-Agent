"""Generic element priority, suppression, and semantic deduplication."""

from browser.element_priority import generic_element_count, prioritize_elements
from browser.protocol import PageElement


def element(agent_id: str, **values: object) -> PageElement:
    return PageElement(
        agent_id=agent_id,
        tag=str(values.pop("tag", "div")),
        role=str(values.pop("role", "generic")),
        document_order=int(values.pop("document_order", 0)),
        **values,
    )


def test_named_visible_controls_win_a_small_budget() -> None:
    items = [
        element("f0:e-001", name="", visible=False),
        element("f0:e-002", tag="a", role="link", name="Result", href="/job"),
        element("f0:e-003", tag="button", role="button", name="Search"),
        element("f0:e-004", tag="input", role="searchbox", name="What"),
    ]
    selected, overflow = prioritize_elements(items, 2)
    assert [item.agent_id for item in selected] == ["f0:e-004", "f0:e-003"]
    assert overflow is True


def test_decorative_path_and_unnamed_generic_are_suppressed() -> None:
    selected, overflow = prioritize_elements(
        [
            element("f0:e-001", tag="path"),
            element("f0:e-002"),
            element("f0:e-003", tag="button", role="button", name="Keep"),
        ],
        10,
    )
    assert [item.agent_id for item in selected] == ["f0:e-003"]
    assert overflow is False


def test_hidden_responsive_duplicate_does_not_replace_visible_action() -> None:
    visible = element(
        "f0:e-001",
        tag="a",
        role="link",
        name="Jobs",
        href="https://example.test/jobs",
        visible=True,
        semantic_region="desktop",
    )
    hidden = visible.model_copy(
        update={
            "agent_id": "f0:e-002",
            "visible": False,
            "semantic_region": "mobile",
        }
    )
    selected, _ = prioritize_elements([hidden, visible], 10)
    assert selected == [visible]


def test_same_text_with_different_destinations_is_not_deduplicated() -> None:
    first = element(
        "f0:e-001", tag="a", role="link", name="Open", href="/one"
    )
    second = first.model_copy(
        update={"agent_id": "f0:e-002", "href": "/two"}
    )
    selected, _ = prioritize_elements([first, second], 10)
    assert selected == [first, second]


def test_generic_count_is_measurable() -> None:
    assert generic_element_count(
        [
            element("f0:e-001"),
            element("f0:e-002", tag="button", role="button", name="Go"),
        ]
    ) == 1
