import { DOCUMENT_ID, INSTRUMENT_ATTRIBUTE, extractSnapshot, instrumentState } from "./page";
import { ElementRegistry } from "./registry";
import type {
  ActionResult,
  BrowserToolCall,
  ExtractOptions,
  PageState,
  ResolvedElement,
} from "./types";
import { visibilityState } from "./visibility";

const ELEMENT_ID = /^f\d+:e-\d{3,}$/;
const QUIET_MS = 500;

// Count-only fetch/XHR bookkeeping is not a DOM/attachment identity change.
// Compare every recorded transition, not just the final attribute: navigation,
// dialogs, malformed state and change-then-restore must still invalidate.
function instrumentEvidence(raw: string | null): string | null {
  if (raw === null) return null;
  try {
    const value = JSON.parse(raw) as Record<string, unknown>;
    if (!value || typeof value !== "object" || Array.isArray(value) ||
        Object.keys(value).sort().join(",") !== "dialogs,inflight,navigation" ||
        !Number.isSafeInteger(value["inflight"]) || (value["inflight"] as number) < 0 ||
        !Number.isSafeInteger(value["navigation"]) || (value["navigation"] as number) < 0 ||
        !Array.isArray(value["dialogs"]) || value["dialogs"].some(dialog =>
          !dialog || typeof dialog !== "object" || Array.isArray(dialog) ||
          Object.keys(dialog).sort().join(",") !== "dismissed,message,type" ||
          !["alert", "confirm", "prompt"].includes(dialog.type) ||
          typeof dialog.message !== "string" || dialog.dismissed !== true)) return null;
    return JSON.stringify([value["navigation"], value["dialogs"].map(dialog =>
      [dialog.type, dialog.message, dialog.dismissed])]);
  } catch { return null; }
}

export class BrowserRuntime {
  private readonly registry = new ElementRegistry();
  private readonly observer: MutationObserver;
  private dirty = true;
  private occludedTarget: HTMLElement | null = null;
  private domVersion = 0;
  private readonly leases = new Set<string>();
  private lastMutationAt = performance.now();
  private options: ExtractOptions = {
    maxVisibleTextChars: 20_000,
    maxElements: 300,
  };

  constructor() {
    if (typeof chrome !== "undefined" && chrome.runtime?.onConnect) {
      chrome.runtime.onConnect.addListener(port => {
        if (!port.name.startsWith("agent-control:") || port.sender?.id !== chrome.runtime.id) return;
        const lease = port.name.slice("agent-control:".length);
        this.leases.add(lease);
        port.onDisconnect.addListener(() => this.leases.delete(lease));
        port.postMessage({ ready: true });
      });
    }
    this.observer = new MutationObserver((mutations) => this.recordMutations(mutations));
    this.observer.observe(document, {
      subtree: true,
      childList: true,
      characterData: true,
      attributes: true,
      attributeOldValue: true,
    });
  }

  private recordMutations(mutations: MutationRecord[]): void {
    if (
      mutations.every(
        (mutation) =>
          mutation.type === "attributes" &&
          (mutation.attributeName === "data-agent-id" ||
            mutation.attributeName === "data-agent-highlight" ||
            (mutation.target === document.documentElement && mutation.attributeName === INSTRUMENT_ATTRIBUTE &&
              instrumentEvidence(mutation.oldValue) !== null &&
              instrumentEvidence(mutation.oldValue) === instrumentEvidence(document.documentElement.getAttribute(INSTRUMENT_ATTRIBUTE)))),
      )
    ) return;
    this.dirty = true;
    this.domVersion += 1;
    this.lastMutationAt = performance.now();
  }

  extract(options: ExtractOptions): PageState {
    this.options = options;
    // Pending mutations precede and are represented by this fresh observation.
    // Drain before extraction only: changes during/after it must still invalidate it.
    this.recordMutations(this.observer.takeRecords());
    const snapshot = extractSnapshot(this.registry, options);
    this.dirty = false;
    return snapshot;
  }

  bindTab(tabId: number): void { this.registry.bindTab(tabId); }

  invalidate(): void { this.registry.invalidate(); this.dirty = true; }

  identity(): { document_id: string; snapshot_id: string; unchanged: boolean } {
    this.recordMutations(this.observer.takeRecords());
    return { document_id: DOCUMENT_ID, snapshot_id: this.registry.currentSnapshotId, unchanged: !this.dirty };
  }

  stats(): { reverse: number; snapshots: number } {
    return this.registry.stats();
  }

  async execute(call: BrowserToolCall, options: ExtractOptions): Promise<ActionResult> {
    this.options = options;
    const started = performance.now();
    if ((call.documentId && call.documentId !== DOCUMENT_ID) ||
        (call.expectedUrl && call.expectedUrl !== location.href)) {
      return this.result(call, started, { success: false, error: "target_document_changed", message: "The target document changed before execution." });
    }
    if (call.leaseId && !this.leases.has(call.leaseId)) {
      return this.result(call, started, { success: false, error: "run_paused", message: "The panel execution lease is closed." });
    }
    const beforeUrl = location.href;
    const beforeNav = instrumentState().navigation;
    const beforeDialogs = instrumentState().dialogs.length;

    if (call.tool === "get_page_state") {
      const page = this.extract(options);
      return this.result(call, started, {
        success: true,
        snapshotId: page.snapshot_id,
        data: { page_state: page },
      });
    }
    if (call.tool === "read_page") {
      const page = this.extract(options);
      const maximum =
        typeof call.arguments["max_chars"] === "number"
          ? call.arguments["max_chars"]
          : page.visible_text.length;
      return this.result(call, started, {
        success: true,
        snapshotId: page.snapshot_id,
        data: { text: page.visible_text.slice(0, maximum), page_state: page },
      });
    }
    if (call.tool === "find_text") {
      const page = this.extract(options);
      const needle = String(call.arguments["text"] ?? "").toLocaleLowerCase();
      const haystack = page.visible_text.toLocaleLowerCase();
      const matches: string[] = [];
      let offset = 0;
      while (needle && matches.length < 20) {
        const index = haystack.indexOf(needle, offset);
        if (index < 0) break;
        matches.push(page.visible_text.slice(Math.max(0, index - 80), index + needle.length + 80));
        offset = index + needle.length;
      }
      return this.result(call, started, {
        success: true,
        snapshotId: page.snapshot_id,
        data: { count: matches.length, matches, page_state: page },
      });
    }
    if (call.tool === "wait_for_stable") {
      const timeout =
        typeof call.arguments["timeout_ms"] === "number"
          ? call.arguments["timeout_ms"]
          : 10_000;
      const stable = await this.waitForStable(timeout, call.leaseId);
      return this.result(call, started, {
        success: stable,
        error: stable ? null : "timeout",
        message: stable ? "Page is stable." : "Page did not become stable before timeout.",
      });
    }
    if (call.tool === "press_key") {
      const key = String(call.arguments["key"] ?? "");
      const target =
        document.activeElement instanceof HTMLElement
          ? document.activeElement
          : document.body;
      this.pressKey(target, key);
      return this.afterAction(call, started, beforeUrl, beforeNav, beforeDialogs);
    }

    if (this.dirty) this.extract(this.options);
    const elementId =
      typeof call.arguments["element_id"] === "string"
        ? call.arguments["element_id"]
        : null;

    if (call.tool === "scroll" && !elementId) {
      const direction = String(call.arguments["direction"] ?? "down");
      const amount =
        typeof call.arguments["amount"] === "number" ? call.arguments["amount"] : 600;
      const horizontal = direction === "left" || direction === "right";
      const sign = direction === "up" || direction === "left" ? -1 : 1;
      scrollBy({ left: horizontal ? amount * sign : 0, top: horizontal ? 0 : amount * sign });
      return this.afterAction(call, started, beforeUrl, beforeNav, beforeDialogs);
    }

    if (!elementId || !ELEMENT_ID.test(elementId)) {
      return this.result(call, started, {
        success: false,
        error: "invalid_arguments",
        message: "A frame-qualified agent_id is required; CSS selectors are forbidden.",
      });
    }
    const resolved = this.registry.resolve(elementId, call.snapshotId);
    if (!resolved) {
      return this.result(call, started, {
        success: false,
        elementId,
        error:
          call.snapshotId === this.registry.currentSnapshotId
            ? "element_not_found"
            : "stale_snapshot",
        message: "The element could not be uniquely resolved against the current DOM.",
      });
    }
    const common = {
      elementId,
      elementName: resolved.descriptor.name,
      reresolved: resolved.reresolved,
    };
    if (call.tool === "get_element_details") {
      const element = this.registry.describe(resolved);
      const safeElement = element.visible
        ? element
        : { ...element, name: "", text: "", placeholder: "", value: "" };
      return this.result(call, started, {
        success: true,
        snapshotId: this.registry.currentSnapshotId,
        message: "Element details read.",
        data: { element: safeElement },
        ...common,
      });
    }
    if (call.tool === "scroll") {
      resolved.element.scrollIntoView({ block: "center", inline: "nearest" });
      return this.afterAction(
        call, started, beforeUrl, beforeNav, beforeDialogs, common,
      );
    }
    if (call.tool === "click") {
      const error = await this.click(resolved, call);
      if (error) {
        return this.result(call, started, { success: false, error, ...common });
      }
    } else if (call.tool === "type_text" || call.tool === "clear_input") {
      const text =
        call.tool === "clear_input" ? "" : String(call.arguments["text"] ?? "");
      const error = this.setText(resolved.element, text);
      if (error) {
        return this.result(call, started, { success: false, error, ...common });
      }
      if (call.tool === "type_text" && call.arguments["submit"] === true) {
        this.pressKey(resolved.element, "Enter");
      }
    } else {
      return this.result(call, started, {
        success: false,
        error: "invalid_arguments",
        message: `Unsupported content tool: ${call.tool}`,
        ...common,
      });
    }
    return this.afterAction(
      call, started, beforeUrl, beforeNav, beforeDialogs, common,
    );
  }

  private async click(resolved: ResolvedElement, call: BrowserToolCall): Promise<"element_occluded" | "element_disabled" | "run_paused" | null> {
    const element = resolved.element;
    if (
      element.getAttribute("aria-disabled") === "true" ||
      ("disabled" in element && (element as HTMLInputElement).disabled)
    ) return "element_disabled";
    element.scrollIntoView({ block: "center", inline: "center" });
    await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
    if (call.leaseId && !this.leases.has(call.leaseId)) return "run_paused";
    const state = visibilityState(element);
    if (!state.visible || state.occluded) {
      this.occludedTarget = element;
      return "element_occluded";
    }
    this.occludedTarget = null;
    await this.highlight(element);
    if ((call.leaseId && !this.leases.has(call.leaseId)) || (call.expectedUrl && call.expectedUrl !== location.href)) return "run_paused";
    const rect = element.getBoundingClientRect();
    const init = {
      bubbles: true,
      cancelable: true,
      composed: true,
      clientX: rect.left + rect.width / 2,
      clientY: rect.top + rect.height / 2,
      button: 0,
      buttons: 1,
    };
    for (const type of ["pointerover", "pointerenter", "pointerdown"]) {
      element.dispatchEvent(new PointerEvent(type, init));
    }
    element.dispatchEvent(new MouseEvent("mousedown", init));
    element.focus({ preventScroll: true });
    element.dispatchEvent(new PointerEvent("pointerup", { ...init, buttons: 0 }));
    element.dispatchEvent(new MouseEvent("mouseup", { ...init, buttons: 0 }));
    element.dispatchEvent(new MouseEvent("click", { ...init, buttons: 0 }));
    return null;
  }

  private setText(element: HTMLElement, text: string): "password_field" | "invalid_arguments" | null {
    if (element instanceof HTMLInputElement) {
      if (element.type.toLowerCase() === "password") return "password_field";
      const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")?.set;
      setter?.call(element, text);
    } else if (element instanceof HTMLTextAreaElement) {
      const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")?.set;
      setter?.call(element, text);
    } else if (element.isContentEditable) {
      element.focus();
      element.dispatchEvent(
        new InputEvent("beforeinput", {
          bubbles: true,
          cancelable: true,
          inputType: "insertText",
          data: text,
        }),
      );
      element.textContent = text;
    } else {
      return "invalid_arguments";
    }
    element.dispatchEvent(new InputEvent("input", { bubbles: true, inputType: "insertText", data: text }));
    element.dispatchEvent(new Event("change", { bubbles: true }));
    return null;
  }

  private pressKey(element: HTMLElement, key: string): void {
    element.focus();
    const init = { key, bubbles: true, cancelable: true, composed: true };
    const proceed = element.dispatchEvent(new KeyboardEvent("keydown", init));
    element.dispatchEvent(new KeyboardEvent("keyup", init));
    if (proceed && key === "Enter") {
      const form = element.closest("form");
      if (form instanceof HTMLFormElement) form.requestSubmit();
    }
  }

  private async highlight(element: HTMLElement): Promise<void> {
    element.setAttribute("data-agent-highlight", "true");
    element.animate(
      [
        { outline: "3px solid #ff9f1c", outlineOffset: "3px" },
        { outline: "3px solid transparent", outlineOffset: "6px" },
      ],
      { duration: 650, easing: "ease-out" },
    );
    await new Promise((resolve) => setTimeout(resolve, 120));
    element.removeAttribute("data-agent-highlight");
  }

  private async waitForStable(timeoutMs: number, leaseId?: string): Promise<boolean> {
    const deadline = performance.now() + timeoutMs;
    while (performance.now() < deadline) {
      if (leaseId && !this.leases.has(leaseId)) return false;
      const quiet = performance.now() - this.lastMutationAt >= QUIET_MS;
      const occlusionCleared =
        !this.occludedTarget ||
        !this.occludedTarget.isConnected ||
        !visibilityState(this.occludedTarget).occluded;
      if (
        document.readyState === "complete" &&
        quiet &&
        instrumentState().inflight === 0 &&
        occlusionCleared
      ) {
        this.occludedTarget = null;
        return true;
      }
      await new Promise((resolve) => setTimeout(resolve, 50));
    }
    return false;
  }

  private async afterAction(
    call: BrowserToolCall,
    started: number,
    beforeUrl: string,
    beforeNav: number,
    beforeDialogs: number,
    extra: {
      elementId?: string;
      elementName?: string;
      reresolved?: boolean;
    } = {},
  ): Promise<ActionResult> {
    await Promise.resolve();
    const state = instrumentState();
    const dialog = state.dialogs.length > beforeDialogs;
    return this.result(call, started, {
      success: !dialog,
      error: dialog ? "blocked_by_dialog" : null,
      message: dialog
        ? state.dialogs.at(-1)?.message ?? "A native dialog was auto-dismissed."
        : "Action completed.",
      pageChanged:
        location.href !== beforeUrl ||
        state.navigation !== beforeNav,
      ...extra,
    });
  }

  private result(
    call: BrowserToolCall,
    started: number,
    values: {
      success: boolean;
      snapshotId?: string;
      elementId?: string;
      elementName?: string;
      pageChanged?: boolean;
      reresolved?: boolean;
      error?: ActionResult["error"];
      message?: string;
      data?: Record<string, unknown>;
    },
  ): ActionResult {
    return {
      success: values.success,
      action: call.tool,
      element_id: values.elementId ?? null,
      element_name: values.elementName ?? null,
      snapshot_id:
        values.snapshotId ?? (this.registry.currentSnapshotId || call.snapshotId),
      page_changed: values.pageChanged ?? false,
      reresolved: values.reresolved ?? false,
      duration_ms: Math.round(performance.now() - started),
      error: values.error ?? null,
      message: values.message ?? "",
      data: values.data ?? {},
    };
  }
}
