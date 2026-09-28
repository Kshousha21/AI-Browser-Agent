import { accessibleName, computeRole, normalizeText } from "./names";
import type { ElementDescriptor, PageElement, ResolvedElement } from "./types";
import { visibilityState } from "./visibility";

const INTERACTIVE_TAGS = new Set([
  "A", "AREA", "BUTTON", "INPUT", "SELECT", "TEXTAREA", "SUMMARY",
]);
const STRUCTURAL_TAGS = new Set(["FORM", "TABLE"]);
const FORM_TAGS = new Set(["INPUT", "SELECT", "TEXTAREA"]);
const ARIA_CONTROLS = new Set([
  "checkbox", "combobox", "listbox", "menuitem", "option", "radio",
  "searchbox", "slider", "spinbutton", "switch", "tab", "textbox", "treeitem",
]);
const AGENT_ID = /^f\d+:e-\d{3,}$/;

interface Candidate {
  element: HTMLElement;
  descriptor: ElementDescriptor;
  page: PageElement;
}

function enabled(element: HTMLElement): boolean {
  if (element.getAttribute("aria-disabled") === "true") return false;
  if ("disabled" in element && (element as HTMLInputElement).disabled) return false;
  return element.closest("fieldset[disabled]") === null;
}

function interactive(element: HTMLElement): boolean {
  if (INTERACTIVE_TAGS.has(element.tagName)) {
    return element.tagName !== "A" || element.hasAttribute("href");
  }
  if (element.isContentEditable) return true;
  if (element.hasAttribute("role") || element.hasAttribute("tabindex")) return true;
  if (element.hasAttribute("onclick")) return true;
  return getComputedStyle(element).cursor === "pointer";
}

function isCandidate(element: HTMLElement): boolean {
  if (element instanceof HTMLInputElement && element.type.toLowerCase() === "password") {
    return false;
  }
  if (!interactive(element) && !STRUCTURAL_TAGS.has(element.tagName)) return false;
  const role = computeRole(element);
  if (role === "none" || role === "presentation") return false;
  const ancestor = element.parentElement?.closest<HTMLElement>(
    "a[href],button,input,select,textarea,summary,[role='button'],[role='link']",
  );
  if (ancestor && !INTERACTIVE_TAGS.has(element.tagName) && !element.hasAttribute("role")) {
    return false;
  }
  return true;
}

function semanticRegion(element: HTMLElement): string {
  const region = element.closest<HTMLElement>(
    "[data-key],[data-testid],[role='dialog'],[role='main'],main,form,nav,article,section",
  );
  if (!region) return "";
  return [
    region.tagName.toLowerCase(),
    region.getAttribute("role") ?? "",
    region.getAttribute("data-testid") ?? "",
    region.getAttribute("data-key") ?? "",
    region.id,
  ].filter(Boolean).join(":").slice(0, 300);
}

function descriptor(
  element: HTMLElement,
  agentId: string,
  documentOrder: number,
): ElementDescriptor {
  const link = element instanceof HTMLAnchorElement ||
    element instanceof HTMLAreaElement ? element : null;
  const form = element.closest("form");
  const submitter =
    element instanceof HTMLButtonElement || element instanceof HTMLInputElement
      ? element
      : null;
  const ariaCurrent = element.getAttribute("aria-current");
  return {
    agentId,
    tag: element.tagName.toLowerCase(),
    role: computeRole(element),
    name: accessibleName(element),
    documentOrder,
    href: link?.href ?? "",
    formAction: submitter?.formAction || form?.action || "",
    formMethod: (submitter?.formMethod || form?.method || "").toLowerCase(),
    formTarget: submitter?.formTarget || form?.target || "",
    domId: element.id,
    testId: element.getAttribute("data-testid") ?? "",
    dataKey: element.getAttribute("data-key") ?? "",
    semanticRegion: semanticRegion(element),
    selected:
      element.getAttribute("aria-selected") === "true" ||
      (ariaCurrent !== null && ariaCurrent !== "false"),
  };
}

export function pageElement(
  element: HTMLElement,
  item: ElementDescriptor,
): PageElement {
  const state = visibilityState(element);
  const input = element instanceof HTMLInputElement ? element : null;
  const button = element instanceof HTMLButtonElement ? element : null;
  const valued =
    element instanceof HTMLInputElement ||
    element instanceof HTMLTextAreaElement ||
    element instanceof HTMLSelectElement;
  return {
    agent_id: item.agentId,
    tag: item.tag,
    role: item.role,
    name: item.name,
    text: normalizeText(element.innerText || element.textContent || ""),
    placeholder: element.getAttribute("placeholder") ?? "",
    value: input?.type === "password" ? "" : valued ? element.value : "",
    input_type: input?.type ?? button?.type ?? "",
    href: item.href,
    form_action: item.formAction,
    form_method: item.formMethod,
    form_target: item.formTarget,
    dom_id: item.domId,
    test_id: item.testId,
    data_key: item.dataKey,
    semantic_region: item.semanticRegion,
    selected: item.selected,
    enabled: enabled(element),
    visible: state.visible,
    occluded: state.occluded,
    in_viewport: state.inViewport,
    bbox: state.bbox,
    document_order: item.documentOrder,
  };
}

function priority(item: PageElement): number {
  const named = item.name.trim().length > 0;
  if (item.visible && named && FORM_TAGS.has(item.tag.toUpperCase())) return 0;
  if (item.visible && named && item.role === "button") return 1;
  if (item.visible && named && item.role === "link" && item.href) return 2;
  if (item.visible && named && ARIA_CONTROLS.has(item.role)) return 3;
  const structural = [item.test_id, item.semantic_region, item.data_key]
    .join(" ").toLocaleLowerCase();
  if (item.visible && named && /result|card|item|job/.test(structural)) return 4;
  if (item.visible && STRUCTURAL_TAGS.has(item.tag.toUpperCase())) return 5;
  if (item.visible && actionable(item)) return 6;
  if (!item.visible && actionable(item)) return 7;
  if (!named || item.role === "generic") return 8;
  return 9;
}

function actionable(item: PageElement): boolean {
  return (
    FORM_TAGS.has(item.tag.toUpperCase()) ||
    item.role === "button" ||
    item.role === "link" ||
    ARIA_CONTROLS.has(item.role) ||
    Boolean(item.href || item.form_action)
  );
}

function lowValue(item: PageElement): boolean {
  if (item.tag === "path") return true;
  if (item.tag === "svg" && !item.name && ["", "generic", "img"].includes(item.role)) {
    return true;
  }
  return (
    ["", "generic", "none", "presentation"].includes(item.role) &&
    !item.name &&
    !item.href &&
    !item.form_action &&
    !STRUCTURAL_TAGS.has(item.tag.toUpperCase()) &&
    !FORM_TAGS.has(item.tag.toUpperCase())
  );
}

function normalized(value: string): string {
  return value.trim().replace(/\s+/g, " ").toLocaleLowerCase();
}

function semanticKey(item: PageElement): string {
  return JSON.stringify([
    item.role.toLocaleLowerCase(),
    normalized(item.name),
    item.tag,
    item.href,
    item.form_action,
    item.form_method,
    item.form_target,
    item.data_key,
    item.semantic_region,
  ]);
}

function destinationKey(item: PageElement): string | null {
  const destination = item.href || item.form_action;
  if (!destination || !item.name) return null;
  return JSON.stringify([
    item.role.toLocaleLowerCase(),
    normalized(item.name),
    destination,
    item.form_method,
    item.form_target,
  ]);
}

function prioritized(candidates: Candidate[]): Candidate[] {
  const sorted = candidates
    .filter((item) => !lowValue(item.page))
    .sort((left, right) =>
      priority(left.page) - priority(right.page) ||
      left.page.document_order - right.page.document_order
    );
  const output: Candidate[] = [];
  const exact = new Set<string>();
  const visibleDestinations = new Set<string>();
  for (const candidate of sorted) {
    const key = semanticKey(candidate.page);
    if (exact.has(key)) continue;
    const destination = destinationKey(candidate.page);
    if (!candidate.page.visible && destination && visibleDestinations.has(destination)) {
      continue;
    }
    exact.add(key);
    if (candidate.page.visible && destination) visibleDestinations.add(destination);
    output.push(candidate);
  }
  return output;
}

export class ElementRegistry {
  private forward = new WeakMap<Element, string>();
  private readonly reverse = new Map<string, WeakRef<Element>>();
  private readonly snapshots = new Map<string, Map<string, ElementDescriptor>>();
  private currentDescriptors = new Map<string, ElementDescriptor>();
  private counter = 0;
  private namespace = "";

  bindTab(tabId: number): void {
    if (!this.namespace) {
      const nonce = BigInt("0x" + crypto.randomUUID().replaceAll("-", "")).toString().padStart(39, "0");
      this.namespace = `${tabId}${nonce}`;
      this.invalidate();
    }
  }

  invalidate(): void {
    this.forward = new WeakMap();
    this.reverse.clear();
    this.snapshots.clear();
    this.currentDescriptors.clear();
    this.currentSnapshotId = "";
  }
  currentSnapshotId = "";

  rebuild(maxElements: number): {
    snapshotId: string;
    elements: PageElement[];
    truncated: boolean;
  } {
    this.sweep();
    const raw: Candidate[] = [];
    const nodes = Array.from(document.querySelectorAll("*"));
    for (let order = 0; order < nodes.length; order += 1) {
      const node = nodes[order];
      if (!(node instanceof HTMLElement) || !isCandidate(node)) continue;
      const item = descriptor(node, "", order);
      raw.push({ element: node, descriptor: item, page: pageElement(node, item) });
    }

    const candidates = prioritized(raw);
    const output: PageElement[] = [];
    const descriptors = new Map<string, ElementDescriptor>();
    for (const candidate of candidates.slice(0, maxElements)) {
      let id = this.forward.get(candidate.element);
      if (!id) {
        id = `f0:e-${this.namespace + String(++this.counter).padStart(this.namespace ? 9 : 3, "0")}`;
        this.forward.set(candidate.element, id);
      }
      const item = { ...candidate.descriptor, agentId: id };
      this.reverse.set(id, new WeakRef(candidate.element));
      // Even a same-value attribute write notifies page observers. Select2
      // responds by resynchronizing widget classes, making our own observation
      // immediately stale. Keep existing decoration untouched; actual identity
      // changes still write the correct ID and all DOM freshness guards remain.
      if (candidate.element.getAttribute("data-agent-id") !== id) {
        candidate.element.setAttribute("data-agent-id", id);
      }
      descriptors.set(id, item);
      output.push({ ...candidate.page, agent_id: id });
    }

    const snapshotId = `s-${crypto.randomUUID().slice(0, 8)}`;
    this.currentSnapshotId = snapshotId;
    this.currentDescriptors = descriptors;
    this.snapshots.set(snapshotId, descriptors);
    while (this.snapshots.size > 8) {
      const oldest = this.snapshots.keys().next().value as string | undefined;
      if (oldest) this.snapshots.delete(oldest);
    }
    return {
      snapshotId,
      elements: output,
      truncated: candidates.length > maxElements,
    };
  }

  resolve(elementId: string, snapshotId: string): ResolvedElement | null {
    if (!AGENT_ID.test(elementId)) return null;
    const original = this.snapshots.get(snapshotId)?.get(elementId);
    if (!original) return null;
    if (snapshotId === this.currentSnapshotId) {
      const direct = this.reverse.get(elementId)?.deref();
      if (direct instanceof HTMLElement && direct.isConnected) {
        return { element: direct, descriptor: original, reresolved: false };
      }
      return null;
    }

    const candidates: Array<{ element: HTMLElement; item: ElementDescriptor }> = [];
    for (const item of this.currentDescriptors.values()) {
      if (
        item.role === original.role &&
        item.name === original.name &&
        item.tag === original.tag
      ) {
        const element = this.reverse.get(item.agentId)?.deref();
        if (element instanceof HTMLElement && element.isConnected) {
          candidates.push({ element, item });
        }
      }
    }
    if (candidates.length === 1) {
      const match = candidates[0];
      return match
        ? { element: match.element, descriptor: match.item, reresolved: true }
        : null;
    }
    return null;
  }

  describe(resolved: ResolvedElement): PageElement {
    return pageElement(resolved.element, resolved.descriptor);
  }

  stats(): { reverse: number; snapshots: number } {
    return { reverse: this.reverse.size, snapshots: this.snapshots.size };
  }

  private sweep(): void {
    for (const [id, ref] of this.reverse) {
      const element = ref.deref();
      if (!element || !element.isConnected) this.reverse.delete(id);
    }
  }
}

