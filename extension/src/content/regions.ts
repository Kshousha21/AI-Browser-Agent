import type { PageRegion } from "@shared/protocol";

import { accessibleName, computeRole, normalizeText } from "./names";
import { isRendered } from "./visibility";

const REGION_SELECTOR = [
  "[data-testid]",
  "main",
  "[role='main']",
  "article",
  "[role='article']",
  "form",
  "table",
  "[role='dialog']",
  "[aria-modal='true']",
  "h1",
  "h2",
  "h3",
  "h4",
  "h5",
  "h6",
].join(",");
const MAX_REGIONS = 600;
const MAX_REGION_TEXT = 12_000;

/** Capture generic semantic structure for host-scoped backend adapters. */
export function collectSemanticRegions(): PageRegion[] {
  const nodes = Array.from(document.querySelectorAll<HTMLElement>(REGION_SELECTOR))
    .filter((node, index, all) => all.indexOf(node) === index)
    .sort(documentOrder)
    .slice(0, MAX_REGIONS);
  const ids = new Map<HTMLElement, string>();
  nodes.forEach((node, index) => {
    ids.set(node, `f0:r-${String(index + 1).padStart(4, "0")}`);
  });
  return nodes.map((node) => {
    const visible = isRendered(node);
    const link =
      node instanceof HTMLAnchorElement
        ? node
        : node.querySelector<HTMLAnchorElement>("a[href]");
    const keyed =
      node.hasAttribute("data-key")
        ? node
        : node.querySelector<HTMLElement>("[data-key]");
    const ancestorRegionIds: string[] = [];
    let ancestor = node.parentElement;
    while (ancestor) {
      const regionId = ids.get(ancestor);
      if (regionId) ancestorRegionIds.push(regionId);
      ancestor = ancestor.parentElement;
    }
    const agentElementIds = [
      ...(node.hasAttribute("data-agent-id") ? [node] : []),
      ...Array.from(node.querySelectorAll<HTMLElement>("[data-agent-id]")),
    ]
      .map((item) => item.getAttribute("data-agent-id") ?? "")
      .filter(Boolean)
      .slice(0, 120);
    return {
      region_id: ids.get(node) ?? "",
      tag: node.tagName.toLowerCase(),
      role: computeRole(node),
      name: visible ? accessibleName(node) : "",
      text: visible
        ? normalizeText(node.innerText || node.textContent || "").slice(0, MAX_REGION_TEXT)
        : "",
      dom_id: node.id,
      test_id: node.getAttribute("data-testid") ?? "",
      data_key: keyed?.getAttribute("data-key") ?? "",
      class_tokens: Array.from(node.classList)
        .filter((item) => !item.startsWith("css-") && item.length <= 80)
        .slice(0, 12),
      href: link?.href ?? "",
      visible,
      ancestor_region_ids: ancestorRegionIds,
      agent_element_ids: [...new Set(agentElementIds)],
    };
  });
}

function documentOrder(left: HTMLElement, right: HTMLElement): number {
  const relation = left.compareDocumentPosition(right);
  if (relation & Node.DOCUMENT_POSITION_FOLLOWING) return -1;
  if (relation & Node.DOCUMENT_POSITION_PRECEDING) return 1;
  return 0;
}
