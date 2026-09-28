import type { ActionResult, ControlStatePayload, PageState, TabMetadata, ToolError, ToolRequestPayload } from "@shared/protocol";
import { isAllowed } from "../lib/allowlist";
import { sameNavigationDestination } from "../lib/navigation-target";
import type { ExtensionSettings } from "../lib/settings";
import { acquire, AcquisitionFailure } from "./acquisition";
import { PreviewFailure, previewFail, previewIdentity, validatePreview, validatePreviewDownload } from "../lib/attachment-preview";
import type { PreviewIdentity } from "../lib/attachment-preview";
import { ResearchGrant, UserFileGrant } from "./user-file-action";
import type { FileSelection, ResearchSelection, UserFileAction } from "./user-file-action";

const CONTENT = "__aiBrowserAgentContent";
interface ContentApi {
  inspectAttachmentPreview: typeof import("../content/attachment-preview").inspectAttachmentPreview;
  bindTab(tabId: number): void;
  invalidate(): void;
  identity(): { document_id: string; snapshot_id: string; unchanged: boolean };
  extractPageState(options: object): PageState;
  executeTool(call: object, options: object): Promise<ActionResult>;
}
class BrowserFailure extends Error {
  constructor(readonly code: ToolError, message: string, readonly details: Record<string, unknown> = {}) { super(message); }
}
interface PreviewOperation {
  id: string; url: string; attachment: PreviewIdentity; version: number; epoch: number; runId: string;
  tabId: number; transitioning: boolean; destination: string; chromeId: string | null;
  href: string | null; closed: boolean; unexpected: boolean; lease: chrome.runtime.Port; heartbeat: number;
  controller: AbortController;
  userActionId: string | null;
}
function fail(code: ToolError, message: string): never { throw new BrowserFailure(code, message); }

export async function activeTab(): Promise<chrome.tabs.Tab | null> {
  if (browserSession.ownerTabId !== null) {
    const tab = await browserSession.ownerTab();
    return tab?.active ? tab : null;
  }
  const [tab] = await chrome.tabs.query({ active: true, lastFocusedWindow: true });
  return tab ?? null;
}

/** Side-panel-owned target authority; no service-worker session or unrelated DOM access. */
export class BrowserSession {
  private owner: number | null = null;
  ownerWindowId: number | null = null;
  get ownerTabId(): number | null { return this.owner; }
  bindOwner(tabId: number, windowId: number): void {
    if (!Number.isSafeInteger(tabId) || tabId < 0 || (this.owner !== null && this.owner !== tabId)) {
      fail("target_tab_changed", "A tab-specific panel cannot change its owning tab.");
    }
    this.owner = tabId; this.ownerWindowId = windowId;
  }
  async ownerTab(): Promise<chrome.tabs.Tab | null> {
    if (this.owner === null) return null;
    try {
      const tab = await chrome.tabs.get(this.owner);
      this.ownerWindowId = tab.windowId; return tab;
    } catch { return null; }
  }
  requireOwner(tabId: number | null | undefined): void {
    if (this.owner !== null && this.owner !== tabId) {
      fail("target_tab_changed", "Use the assistant belonging to that tab.");
    }
  }
  mode: ControlStatePayload["state"] = "running";
  epoch = 0;
  version = 0;
  runId: string | null = null;
  target: PageState | null = null;
  snapshots = new Set<string>();
  opened = new Set<number>();
  ports = new Set<chrome.runtime.Port>();
  revisions = new Map<number, { key: string; value: string }>();
  expectedTab: number | null = null;
  opening = false;
  navigating: number | null = null;
  linkNavigation: { tabId: number; source: string; destination: string; started?: boolean; sourceDocumentId?: string } | null = null;
  closing: number | null = null;
  preview: PreviewOperation | null = null;
  userFileGrant: UserFileGrant | null = null;
  researchGrant: ResearchGrant | null = null;
  private documentController = new AbortController();
  get documentSignal(): AbortSignal { return this.documentController.signal; }
  onInvalidation: ((event: "active_target_changed" | "target_tab_closed" | "target_document_changed") => void) | null = null;

  begin(): void {
    this.cancelLeases();
    this.runId = null; this.mode = "running"; this.epoch = 0;
    this.version++; this.target = null; this.snapshots.clear(); this.opened.clear();
  }
  beginUserFileAction(id: string, action: UserFileAction, expiresAt: number, selection: FileSelection): void {
    if (this.userFileGrant || this.owner === null || this.mode !== "running") return previewFail("operation_interrupted", "human_grant");
    this.userFileGrant = new UserFileGrant(id, action, expiresAt, selection, this.owner, this.version, this.epoch);
  }
  endUserFileAction(): void { this.userFileGrant = null; }
  beginResearch(id: string, maximum = 3, specific?: { selection: FileSelection; sourceUrl: string; expiresAt: number }): void {
    if (this.researchGrant || this.userFileGrant || this.owner === null || this.mode !== "running") return previewFail("operation_interrupted", "research_grant");
    const grant = new ResearchGrant(id, maximum, this.owner, this.version, this.epoch);
    if (specific) grant.admitSpecific(specific.selection, specific.sourceUrl, this.target?.url, specific.expiresAt);
    this.researchGrant = grant;
  }
  approveResearchPlan(id: string, sourceUrl: string, files: ResearchSelection[], expiresAt: number): void {
    if (!this.researchGrant) return previewFail("operation_interrupted", "research_grant");
    this.researchGrant.check(id, this.owner, this.version, this.epoch);
    this.researchGrant.plan(sourceUrl, this.target?.url, files, expiresAt);
  }
  activateResearchFile(id: string, sourceKey: string, selection: FileSelection): void {
    if (!this.researchGrant || this.mode !== "running" || this.owner === null) return previewFail("operation_interrupted", "research_grant");
    this.researchGrant.check(id, this.owner, this.version, this.epoch);
    // The whole bounded cohort was admitted while fresh. This activation checks
    // exact newly observed lineage without creating a new timed approval.
    this.userFileGrant = this.researchGrant.file(sourceKey, selection, this.target?.url);
  }
  endResearch(): void { this.researchGrant = null; this.endUserFileAction(); }
  cancelPreview(): void {
    const preview = this.preview; this.preview = null;
    if (preview) { clearInterval(preview.heartbeat); preview.closed = true; preview.controller.abort(); this.ports.delete(preview.lease); preview.lease.disconnect(); }
  }
  cancelLeases(): void { this.documentController.abort(); this.documentController = new AbortController(); this.endResearch(); this.cancelPreview(); for (const port of this.ports) port.disconnect(); this.ports.clear(); }
  pause(): void { this.cancelLeases(); this.mode = "paused"; this.version++; this.snapshots.clear(); }
  resume(): void { this.mode = "resuming"; this.version++; this.snapshots.clear(); }
  stop(): void { this.cancelLeases(); this.mode = "stopped"; this.version++; this.snapshots.clear(); this.runId = null; }
  control(state: ControlStatePayload): void {
    if (state.epoch < this.epoch) return;
    this.epoch = state.epoch;
    // A queued running status cannot undo an immediate local Pause.
    if (this.mode === "paused" && state.state === "running") return;
    this.mode = state.state;
    this.runId = state.run_id;
    if (state.state === "paused" || state.state === "waiting" || state.state === "stopped") {
      this.cancelLeases(); this.version++; this.snapshots.clear();
    }
  }
  invalidate(event: "active_target_changed" | "target_tab_closed" | "target_document_changed"): void {
    if (this.mode === "stopped") return;
    this.pause(); if (this.runId) this.onInvalidation?.(event);
  }
  activated(tabId: number, windowId?: number): void {
    if (this.owner !== null) {
      if (windowId !== undefined && windowId !== this.ownerWindowId) return;
      if (tabId !== this.owner) this.invalidate("active_target_changed");
      return;
    }
    if (tabId === this.expectedTab || this.opening || this.closing !== null) return;
    if (this.target?.tab_id !== tabId) this.invalidate("active_target_changed");
  }
  updated(tabId: number, info: chrome.tabs.TabChangeInfo): void {
    if (info.url || info.status === "loading" || info.title) this.revisions.delete(tabId);
    const preview = this.preview;
    if (preview && tabId === preview.tabId && (info.url || info.status === "loading")) {
      if ((info.url && info.url !== preview.destination) || !preview.transitioning) this.invalidate("target_document_changed");
      return; // Only this exact, one-operation transition is exempted, never ordinary navigation.
    }
    const link = this.linkNavigation;
    if (link && tabId === link.tabId && (info.url || info.status === "loading")) {
      if (info.url && !sameNavigationDestination(info.url, link.destination)) this.invalidate("target_document_changed");
      else link.started = true;
      return; // Only this destination, including the same web page ID, belongs to this operation.
    }
    if (tabId === this.target?.tab_id && (info.url || info.status === "loading") && this.navigating !== tabId) {
      this.invalidate("target_document_changed");
    }
  }
  removed(tabId: number): void {
    this.opened.delete(tabId); this.revisions.delete(tabId);
    if (tabId === this.owner) { this.stop(); this.target = null; return; }
    if (tabId === this.target?.tab_id && this.closing !== tabId) this.invalidate("target_tab_closed");
  }
  adopt(page: PageState, reset = false): void {
    this.requireOwner(page.tab_id);
    if (reset || page.tab_id !== this.target?.tab_id || page.document_id !== this.target?.document_id) this.snapshots.clear();
    this.target = page; this.snapshots.add(page.snapshot_id);
    while (this.snapshots.size > 8) this.snapshots.delete(this.snapshots.values().next().value!);
  }
  guard(request: ToolRequestPayload): () => void {
    const version = this.version;
    return () => {
      const refresh = request.tool === "get_page_state" && request.reset_target === true && this.mode === "resuming";
      if (version !== this.version || (this.mode !== "running" && !refresh) || (request.control_epoch ?? 0) !== this.epoch) {
        fail("run_paused", "Action belongs to a paused or superseded run.");
      }
      const id = request.arguments?.["user_action_id"];
      if (request.tool === "return_research_results") {
        if (!this.researchGrant) return previewFail("operation_interrupted", "research_grant");
        this.researchGrant.check(request.arguments["research_id"], this.owner, this.version, this.epoch);
        this.researchGrant.returnTarget(request.arguments["url"]);
      }
      if (id !== undefined) {
        if (!this.userFileGrant) return previewFail("operation_interrupted", "human_grant");
        this.userFileGrant.check(request, this.owner, this.version, this.epoch);
      } else if (this.userFileGrant && ["open_attachment_preview", "inspect_attachment_preview", "acquire_attachment", "download_attachment"].includes(request.tool)) {
        return previewFail("operation_interrupted", "human_grant");
      }
    };
  }
  revision(tab: chrome.tabs.Tab): string {
    const id = tab.id!;
    const key = JSON.stringify([tab.url, tab.pendingUrl, tab.title, tab.windowId, tab.status]);
    let saved = this.revisions.get(id);
    if (!saved || saved.key !== key) {
      saved = { key, value: crypto.randomUUID() }; this.revisions.set(id, saved);
    }
    return saved.value;
  }
  async tabs(_settings: ExtensionSettings): Promise<TabMetadata[]> {
    const current = await activeTab();
    const tabs = await chrome.tabs.query({});
    tabs.sort((a, b) => Number(b.id === current?.id) - Number(a.id === current?.id) || Number(this.opened.has(b.id!)) - Number(this.opened.has(a.id!)));
    return tabs.filter(tab => tab.id !== undefined && tab.id >= 0).slice(0, 50).map(tab => {
      const base = { tab_id: tab.id!, active: tab.id === current?.id, allowed: false };
      if ((this.owner !== null && tab.id !== this.owner) || !tab.url || !isAllowed(tab.url) || (tab.pendingUrl && !isAllowed(tab.pendingUrl))) {
        return { ...base, details: "redacted" };
      }
      const url = new URL(tab.url);
      return { ...base, allowed: true, opened_by_agent: this.opened.has(tab.id!),
        title: (tab.title ?? "").replace(/[\x00-\x1f]/g, " ").slice(0, 160),
        location: (url.origin + url.pathname).slice(0, 240), revision: this.revision(tab) };
    });
  }
}
export const browserSession = new BrowserSession();

/** Port readiness is an event, not a short synthetic timeout. */
export function waitForLease(port: chrome.runtime.Port, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const cleanup = () => { port.onMessage.removeListener(ready); port.onDisconnect.removeListener(cancel); signal.removeEventListener("abort", cancel); };
    const cancel = () => { cleanup(); reject(new PreviewFailure("operation_interrupted", "observer_lease")); };
    const ready = (message: { ready?: boolean; stage?: string }) => {
      if (message.ready === undefined) return;
      const stage = ["target", "regex", "rules"].includes(message.stage ?? "") ? "guard_" + message.stage : "observer_lease";
      cleanup(); message.ready ? resolve() : reject(new PreviewFailure("operation_interrupted", stage));
    };
    port.onMessage.addListener(ready); port.onDisconnect.addListener(cancel); signal.addEventListener("abort", cancel, { once: true });
    if (signal.aborted) cancel();
  });
}

/** Only document-workflow navigation uses this lifecycle wait. Agent timeouts remain separate. */
export function waitForDocumentTab(tabId: number, destination: string, check: () => void, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    let settled = false;
    const cleanup = () => { chrome.tabs.onUpdated.removeListener(updated); chrome.tabs.onRemoved.removeListener(removed); signal.removeEventListener("abort", cancel); };
    const finish = (error?: unknown) => { if (settled) return; settled = true; cleanup(); error ? reject(error) : resolve(); };
    const cancel = () => finish(new PreviewFailure("operation_interrupted", "navigation_wait"));
    const inspect = async (tab: chrome.tabs.Tab) => {
      if (settled) return;
      try {
        check();
        if (tab.pendingUrl && tab.pendingUrl !== destination) return finish(new PreviewFailure("preview_identity_mismatch", "navigation_wait"));
        if (tab.url !== destination) {
          if (tab.pendingUrl === destination) return;
          return finish(new PreviewFailure("preview_identity_mismatch", "navigation_wait"));
        }
        // Probe the committed destination immediately, without waiting for
        // unrelated images/iframes to finish. Missing state waits for browser
        // lifecycle events; it is never retried on a timer.
        const [probe] = await chrome.scripting.executeScript({ target: { tabId }, injectImmediately: true,
          func: (expected: string) => location.href === expected && document.documentElement !== null, args: [destination] });
        if (settled) return;
        check(); if (probe?.result && probe.documentId) finish();
      } catch (error) { finish(error); }
    };
    const updated = (id: number, _info: chrome.tabs.TabChangeInfo, tab: chrome.tabs.Tab) => { if (id === tabId) inspect(tab); };
    const removed = (id: number) => { if (id === tabId) finish(new BrowserFailure("target_tab_closed", "Target tab closed while waiting.")); };
    chrome.tabs.onUpdated.addListener(updated); chrome.tabs.onRemoved.addListener(removed); signal.addEventListener("abort", cancel, { once: true });
    if (signal.aborted) { cancel(); return; }
    void chrome.tabs.get(tabId).then(inspect, () => finish(new BrowserFailure("target_tab_closed", "Target tab closed while waiting.")));
  });
}

async function allowedTab(tabId: number, _settings: ExtensionSettings): Promise<chrome.tabs.Tab> {
  let tab: chrome.tabs.Tab;
  try { tab = await chrome.tabs.get(tabId); } catch { return fail("target_tab_closed", "Target tab no longer exists."); }
  if (!tab.url || !isAllowed(tab.url) || (tab.pendingUrl && !isAllowed(tab.pendingUrl))) {
    return fail("permission_denied", "Open a normal HTTP/HTTPS website and click the extension to grant tab access.");
  }
  return tab;
}
function options(settings: ExtensionSettings): object { return { maxVisibleTextChars: settings.maxVisibleTextChars, maxElements: 300 }; }

/** A small URL gate obtains Chrome's document ID before installing the DOM runtime. */
async function ensureContent(tab: chrome.tabs.Tab, check: () => void, immediately = false): Promise<string> {
  check();
  const [probe] = await chrome.scripting.executeScript({ target: { tabId: tab.id! }, injectImmediately: immediately,
    func: (name: string, expectedUrl: string) => ({ same: location.href === expectedUrl,
      loaded: typeof (globalThis as Record<string, unknown>)[name] === "object" }), args: [CONTENT, tab.url!] }).catch(() => fail("permission_denied", "Chrome has not granted access to this page. Click the extension on this tab, or grant optional site access in Settings. Internal Chrome pages cannot be controlled."));
  check();
  if (!probe?.result?.same || !probe.documentId) return fail("target_document_changed", "Document changed before injection.");
  const target = { tabId: tab.id!, documentIds: [probe.documentId] };
  if (!probe.result.loaded) {
    await chrome.scripting.executeScript({ target, files: ["content/instrument.js"], world: "MAIN", injectImmediately: immediately });
    check();
    await chrome.scripting.executeScript({ target, files: ["content/extract.js"], injectImmediately: immediately });
    check();
  }
  return probe.documentId;
}

export async function extractPageState(tabId: number, settings: ExtensionSettings, reset = false, check = () => {}): Promise<PageState> {
  browserSession.requireOwner(tabId);
  const tab = await allowedTab(tabId, settings); check();
  const documentId = await ensureContent(tab, check); check();
  const [result] = await chrome.scripting.executeScript({ target: { tabId, documentIds: [documentId] },
    func: (name: string, id: number, expectedUrl: string, resetRegistry: boolean, limits: object) => {
      if (location.href !== expectedUrl) throw new Error("Target document changed");
      const api = (globalThis as Record<string, unknown>)[name] as ContentApi;
      api.bindTab(id); if (resetRegistry) api.invalidate();
      return { ...api.extractPageState(limits), tab_id: id };
    }, args: [CONTENT, tabId, tab.url!, reset, options(settings)] });
  check();
  if (!result?.result || !isAllowed(result.result.url)) return fail("permission_denied", "Fresh page unavailable or outside scope.");
  if ((await activeTab())?.id !== tabId) return fail("target_tab_changed", "The active tab changed during observation.");
  check(); browserSession.adopt(result.result, reset); return result.result;
}

function result(request: ToolRequestPayload, started: number, data: Record<string, unknown> = {}, changed = false): ActionResult {
  const page = data["page_state"] as PageState | undefined;
  return { success: true, action: request.tool, snapshot_id: page?.snapshot_id ?? request.snapshot_id,
    page_changed: changed, reresolved: false, duration_ms: Math.round(performance.now() - started), message: "Completed.", data };
}

async function requireTarget(request: ToolRequestPayload, settings: ExtensionSettings, check: () => void): Promise<chrome.tabs.Tab> {
  if (request.tab_id !== undefined && request.tab_id !== null) browserSession.requireOwner(request.tab_id);
  const current = await activeTab(); check();
  if (request.tab_id === undefined || request.tab_id === null || current?.id !== request.tab_id) {
    return fail("target_tab_changed", "Request is not bound to the current target tab.");
  }
  const tab = await allowedTab(request.tab_id, settings); check();
  if (request.tool !== "get_page_state" && request.tool !== "get_tabs" && request.tool !== "wait_for_stable") {
    if (!browserSession.target || browserSession.target.tab_id !== request.tab_id || browserSession.target.document_id !== request.document_id) {
      return fail("target_document_changed", "Request is bound to a different document.");
    }
    if (!browserSession.snapshots.has(request.snapshot_id)) return fail("stale_snapshot", "Snapshot is no longer actionable.");
  }
  return tab;
}

async function observePreview(request: ToolRequestPayload, settings: ExtensionSettings, check: () => void, wait: boolean) {
  const preview = browserSession.preview;
  if (!preview || preview.id !== request.arguments["preview_id"] || preview.tabId !== request.tab_id ||
      preview.runId !== request.run_id || preview.version !== browserSession.version || preview.epoch !== browserSession.epoch ||
      preview.userActionId !== (request.arguments["user_action_id"] ?? null)) return previewFail("preview_identity_mismatch", "authority");
  const verify = (): void => { check(); if (preview.unexpected) previewFail("preview_unexpected_download"); if (preview.closed) previewFail("operation_interrupted"); };
  verify();
  const tab = await requireTarget(request, settings, verify); verify();
  if (tab.url !== preview.url || (tab.pendingUrl && tab.pendingUrl !== preview.url)) return previewFail("preview_identity_mismatch", "owner_destination");
  const [html] = await chrome.scripting.executeScript({ target: { tabId: tab.id! }, injectImmediately: true,
    func: () => document.contentType === "text/html" }); verify();
  if (!html?.result || (preview.chromeId && html.documentId !== preview.chromeId)) return previewFail("preview_identity_mismatch", "html_document");
  const chromeId = await ensureContent(tab, verify, true); verify();
  if (chromeId !== html.documentId) return previewFail("preview_identity_mismatch", "chrome_document");
  const observerId = wait ? crypto.randomUUID() : undefined;
  let observerPort: chrome.runtime.Port | undefined;
  const cancelObserver = () => observerPort?.disconnect();
  let response;
  try {
    if (observerId) {
      observerPort = chrome.tabs.connect(tab.id!, { name: `preview-observer:${observerId}`, documentId: chromeId });
      browserSession.ports.add(observerPort);
      preview.controller.signal.addEventListener("abort", cancelObserver, { once: true });
      await waitForLease(observerPort, preview.controller.signal); verify();
    }
    [response] = await chrome.scripting.executeScript({ target: { tabId: tab.id!, documentIds: [chromeId] }, injectImmediately: true,
    // Chrome's injected-value conversion can omit null-valued object fields.
    // Preserve strict nullable identity fields in both directions, never default
    // a missing field into authority or weaken previewIdentity validation.
    func: async (name: string, url: string, identityJson: string, waitForEvents: boolean, allowNonPdf: boolean, observerLease: string | null) => JSON.stringify(await
      ((globalThis as Record<string, unknown>)[name] as ContentApi).inspectAttachmentPreview(url, JSON.parse(identityJson), waitForEvents, allowNonPdf, observerLease ?? undefined)),
    args: [CONTENT, preview.url, JSON.stringify(preview.attachment), wait, preview.userActionId !== null, observerId ?? null] });
  } finally {
    preview.controller.signal.removeEventListener("abort", cancelObserver);
    if (observerPort) { browserSession.ports.delete(observerPort); observerPort.disconnect(); }
  }
  verify();
  const data = response?.result ? JSON.parse(response.result) as Awaited<ReturnType<ContentApi["inspectAttachmentPreview"]>> : null;
  if (!data) return previewFail("preview_control_missing");
  if ("error" in data) return previewFail(data.error, "content_" + data.stage);
  const now = await allowedTab(tab.id!, settings); verify();
  if (now.url !== preview.url || (now.pendingUrl && now.pendingUrl !== preview.url) || (await activeTab())?.id !== tab.id) return previewFail("preview_identity_mismatch", "owner_after_observation");
  verify();
  validatePreviewDownload(data.preview.url, preview.url, preview.attachment, previewIdentity(data.preview.attachment, preview.userActionId !== null));
  if (preview.href !== null && preview.href !== data.preview.url) return previewFail("preview_identity_mismatch", "changed_href");
  preview.href = data.preview.url; preview.chromeId = chromeId;
  const page: PageState = { ...data.page_state, tab_id: tab.id! };
  browserSession.adopt(page, wait);
  return { ...data, page_state: page };
}

async function openPreview(request: ToolRequestPayload, settings: ExtensionSettings, tab: chrome.tabs.Tab, check: () => void) {
  const id = request.arguments["preview_id"], raw = request.arguments["url"];
  if (browserSession.preview || typeof id !== "string" || !/^[a-f0-9]{32}$/.test(id) || typeof raw !== "string") return previewFail();
  const grant = request.arguments["user_action_id"] === undefined ? null : browserSession.userFileGrant;
  const attachment = previewIdentity(request.arguments["attachment"], grant !== null);
  grant?.preview(request);
  validatePreview(raw, tab.url!, attachment);
  if (!isAllowed(raw)) return previewFail();
  const chromeId = await ensureContent(tab, check); check();
  const validate = async (stage: "before_lease" | "before_navigation") => {
    const [probe] = await chrome.scripting.executeScript({ target: { tabId: tab.id!, documentIds: [chromeId] },
      func: (name: string) => ({ ...((globalThis as Record<string, unknown>)[name] as ContentApi).identity(), url: location.href }),
      args: [CONTENT] }); check();
    const evidence = { probe_present: !!probe?.result, url_matches: probe?.result?.url === tab.url,
      snapshot_matches: probe?.result?.snapshot_id === request.snapshot_id,
      document_matches: probe?.result?.document_id === request.document_id,
      dom_unchanged: probe?.result?.unchanged === true };
    if (Object.values(evidence).some(value => !value)) throw new BrowserFailure("stale_snapshot",
      "The selected preview changed after observation. Refresh results.",
      { preview_stage: stage, snapshot_check: evidence });
  };
  await validate("before_lease"); check();
  const lease = chrome.runtime.connect({ name: "document-acquisition-guard" });
  const preview: PreviewOperation = { id, url: raw, attachment, version: browserSession.version, epoch: browserSession.epoch,
    runId: request.run_id, tabId: tab.id!, transitioning: true, destination: raw, chromeId: null, href: null,
    closed: false, unexpected: false, lease, heartbeat: 0, controller: new AbortController(), userActionId: grant?.id ?? null };
  browserSession.preview = preview; browserSession.ports.add(lease);
  lease.onDisconnect.addListener(() => { preview.closed = true; preview.controller.abort(); });
  lease.onMessage.addListener((message: { unexpected?: boolean }) => { if (message.unexpected) { preview.unexpected = true; preview.controller.abort(); } });
  preview.heartbeat = window.setInterval(() => { if (!preview.closed) lease.postMessage({ ping: true }); }, 5000);
  const leaseReady = waitForLease(lease, preview.controller.signal);
  lease.postMessage({ tabId: tab.id, url: raw, preview: true, attachment,
    ...(grant ? { userActionId: grant.id, filename: grant.selection.filename } : {}) });
  await leaseReady;
  check(); await validate("before_navigation"); check();
  if (preview.closed || preview.unexpected) return previewFail("operation_interrupted");
  // A second explicit action may use the already observed preview. Reloading
  // that URL lets URL-only readiness accept the OLD document before load begins.
  if (tab.pendingUrl) return previewFail("operation_interrupted", "navigation_dispatch");
  if (tab.url !== raw) {
    try { await chrome.tabs.update(tab.id!, { url: raw }); }
    catch { return previewFail("operation_interrupted", "navigation_dispatch"); }
    check();
    await waitForDocumentTab(tab.id!, raw, check, preview.controller.signal);
  }
  if (preview.unexpected) return previewFail("preview_unexpected_download");
  // Owner and original snapshot checks remain active until this exact destination
  // returns a new minimal observation. No old result_ref is carried forward.
  const data = await observePreview(request, settings, check, true);
  check(); preview.transitioning = false;
  return data;
}

export async function executeBrowserTool(request: ToolRequestPayload, settings: ExtensionSettings): Promise<ActionResult> {
  const started = performance.now();
  const check = browserSession.guard(request);
  let lease: chrome.runtime.Port | null = null;
  try {
    check();
    if (browserSession.ownerTabId !== null) browserSession.requireOwner(request.tab_id);
    if (request.tool === "get_tabs") { const tabs = await browserSession.tabs(settings); check(); return result(request, started, { tabs }); }
    if (request.tool === "get_page_state" && request.reset_target) {
      const current = await activeTab(); check();
      if (current?.id === undefined) return fail("permission_denied", "Return to this assistant's owning tab before resuming.");
      const page = await extractPageState(current.id, settings, true, check);
      return result(request, started, { page_state: page }, true);
    }
    const tab = await requireTarget(request, settings, check);
    if (request.tool === "return_research_results") {
      const grant = browserSession.researchGrant!;
      const url = grant.returnTarget(request.arguments["url"]);
      if (!isAllowed(url) || new URL(url).origin !== new URL(tab.url!).origin) return previewFail("preview_identity_mismatch", "research_return");
      if (tab.pendingUrl) return previewFail("preview_identity_mismatch", "research_return");
      browserSession.endUserFileAction(); browserSession.cancelPreview();
      // A direct attachment download leaves the source document in place.
      // Reloading that same URL can let a URL-only readiness probe accept the
      // old document before the reload starts, then invalidate our own next
      // action. Observe the current source afresh instead; no navigation grant
      // is created when navigation is not needed.
      if (tab.url !== url) {
        browserSession.navigating = tab.id!;
        check(); await chrome.tabs.update(tab.id!, { url }); check();
        await waitForDocumentTab(tab.id!, url, check, browserSession.documentSignal);
      }
      const page = await extractPageState(tab.id!, settings, true, check);
      check(); if (page.url !== url) return previewFail("preview_identity_mismatch", "research_return");
      grant.returnedToResults();
      return result(request, started, { page_state: page }, true);
    }
    if (request.tool === "open_attachment_preview") return result(request, started, await openPreview(request, settings, tab, check), true);
    if (request.tool === "inspect_attachment_preview") {
      if (request.arguments["user_action_id"] !== undefined) browserSession.userFileGrant!.inspect(request);
      return result(request, started, await observePreview(request, settings, check, false));
    }
    if (browserSession.ownerTabId !== null && (request.tool === "open_tab" ||
        ((request.tool === "switch_tab" || request.tool === "close_tab") && request.arguments["tab_id"] !== browserSession.ownerTabId))) {
      return fail("permission_denied", "This assistant is tab-bound. Open the other tab's own assistant.");
    }
    if (request.tool === "get_page_state") {
      if (typeof request.arguments["selection_snapshot"] === "string") {
        const chromeId = await ensureContent(tab, check); check();
        const [probe] = await chrome.scripting.executeScript({ target: { tabId: tab.id!, documentIds: [chromeId] },
          func: (name: string) => ((globalThis as Record<string, unknown>)[name] as ContentApi).identity(), args: [CONTENT] });
        check();
        if (probe?.result?.snapshot_id !== request.arguments["selection_snapshot"] || probe.result.document_id !== request.document_id) {
          return fail("stale_snapshot", "The selected search snapshot has expired.");
        }
      }
      return result(request, started, { page_state: await extractPageState(tab.id!, settings, false, check) });
    }
    if (request.tool === "acquire_attachment" || request.tool === "download_attachment") {
      const userDownload = request.tool === "download_attachment";
      const grant = request.arguments["user_action_id"] === undefined ? null : browserSession.userFileGrant;
      if (userDownload && !grant) return previewFail("operation_interrupted", "human_grant");
      if (grant) {
        const raw = request.arguments["url"];
        if (typeof raw !== "string") return previewFail();
        const filename = decodeURIComponent(new URL(raw).pathname.split("/").pop() ?? "");
        if (userDownload && request.arguments["filename"] !== filename) return previewFail();
        grant.download(request, filename);
      }
      const preview = browserSession.preview;
      if (request.arguments["preview_id"] !== undefined || preview !== null) {
        if (!preview || request.arguments["preview_id"] !== preview.id || request.arguments["url"] !== preview.href) return previewFail();
        const validate = async (): Promise<void> => {
          const data = await observePreview(request, settings, check, false);
          check(); if (data.preview.url !== request.arguments["url"]) previewFail();
        };
        await validate(); check();
        // Promote the same lease after legacy approval or a bound human click. Keep
        // the preview transition guard until the final recheck inside acquire.
        const finalValidate = async () => { await validate(); check(); preview.destination = preview.href!; preview.transitioning = true; };
        try { return result(request, started, await acquire(request, settings, tab.url!, check, finalValidate, preview.lease, userDownload, browserSession.documentSignal)); }
        finally { browserSession.cancelPreview(); }
      }
      const chromeId = await ensureContent(tab, check); check();
      const validate = async (): Promise<void> => {
        const [probe] = await chrome.scripting.executeScript({ target: { tabId: tab.id!, documentIds: [chromeId] },
          func: (name: string, exactUrl: string, requireLink: boolean) => ({
            ...((globalThis as Record<string, unknown>)[name] as ContentApi).identity(),
            observed: !requireLink || [...document.querySelectorAll<HTMLAnchorElement>("a[href]")].some(a => a.href === exactUrl),
          }), args: [CONTENT, request.arguments["url"] as string, userDownload] });
        check();
        if (probe?.result?.snapshot_id !== request.snapshot_id || probe.result.document_id !== request.document_id || !probe.result.unchanged || (userDownload && !probe.result.observed)) {
          fail("stale_snapshot", "The approved attachment snapshot has expired.");
        }
      };
      await validate();
      browserSession.navigating = tab.id!;
      return result(request, started, await acquire(request, settings, tab.url!, check, validate, undefined, userDownload, browserSession.documentSignal));
    }
    if (request.tool === "open_tab" || request.tool === "switch_tab") {
      let destination: chrome.tabs.Tab;
      if (request.tool === "open_tab") {
        const url = request.arguments["url"];
        if (typeof url !== "string" || !isAllowed(url)) return fail("navigation_blocked", "Destination is outside supported scope.");
        browserSession.opening = true; check();
        destination = await chrome.tabs.create({ url, active: true });
        if (destination.id !== undefined) browserSession.opened.add(destination.id);
      } else {
        const id = request.arguments["tab_id"];
        if (!Number.isInteger(id) || typeof id !== "number") return fail("invalid_arguments", "A tab ID is required.");
        destination = await allowedTab(id, settings); check();
        browserSession.expectedTab = id;
        await chrome.tabs.update(id, { active: true });
        await chrome.windows.update(destination.windowId, { focused: true });
      }
      check();
      if (destination.id === undefined) return fail("target_tab_closed", "New target tab is unavailable.");
      browserSession.expectedTab = destination.id;
      await waitForTab(destination.id, 15000, check);
      const page = await extractPageState(destination.id, settings, true, check);
      return result(request, started, { tab_id: destination.id, opened_by_agent: browserSession.opened.has(destination.id), page_state: page }, true);
    }
    if (request.tool === "close_tab") {
      const id = request.arguments["tab_id"];
      if (typeof id !== "number" || !Number.isInteger(id)) return fail("invalid_arguments", "A tab ID is required.");
      const target = await allowedTab(id, settings); check();
      const siblings = await chrome.tabs.query({ windowId: target.windowId }); check();
      if (siblings.length <= 1) return fail("permission_denied", "The last tab cannot be closed because it would close the window.");
      if (!request.tab_revision || browserSession.revision(target) !== request.tab_revision) return fail("stale_snapshot", "The approved tab changed before closing.");
      browserSession.closing = id; check();
      await chrome.tabs.remove(id); check();
      browserSession.opened.delete(id);
      if (id !== tab.id) return result(request, started, { closed_tab_id: id });
      if (browserSession.ownerTabId !== null) return result(request, started, { closed_tab_id: id, page_access: false }, true);
      const current = await activeTab(); check();
      if (current?.id !== undefined && current.url && isAllowed(current.url)) {
        const page = await extractPageState(current.id, settings, true, check);
        return result(request, started, { closed_tab_id: id, page_state: page }, true);
      }
      return result(request, started, { closed_tab_id: id, page_access: false }, true);
    }
    if (["go_back", "go_forward", "refresh"].includes(request.tool)) {
      let loading = false;
      const changed = (id: number, info: chrome.tabs.TabChangeInfo): void => {
        if (id === tab.id && info.status === "loading") loading = true;
      };
      chrome.tabs.onUpdated.addListener(changed);
      browserSession.navigating = tab.id!;
      try {
        check();
        if (request.tool === "go_back") await chrome.tabs.goBack(tab.id!);
        else if (request.tool === "go_forward") await chrome.tabs.goForward(tab.id!);
        else await chrome.tabs.reload(tab.id!);
        const deadline = performance.now() + 15000;
        while (performance.now() < deadline) {
          check();
          const current = await allowedTab(tab.id!, settings);
          if ((loading || current.url !== tab.url) && current.status === "complete") {
            const page = await extractPageState(tab.id!, settings, true, check);
            return result(request, started, { page_state: page }, true);
          }
          await new Promise(resolve => setTimeout(resolve, 50));
        }
        return fail("timeout", "Browser history or reload did not produce a fresh destination.");
      } finally { chrome.tabs.onUpdated.removeListener(changed); }
    }
    if (request.tool === "navigate") {
      const url = request.arguments["url"];
      if (typeof url !== "string" || !isAllowed(url)) return fail("navigation_blocked", "Destination is outside supported scope.");
      browserSession.navigating = tab.id!;
      browserSession.linkNavigation = { tabId: tab.id!, source: tab.url!, destination: url,
        sourceDocumentId: browserSession.target?.document_id ?? "", started: false };
      check(); await chrome.tabs.update(tab.id!, { url }); check();
      const page = await observeNavigationDestination(settings, check);
      return result(request, started, { page_state: page }, true);
    }
    const chromeDocumentId = await ensureContent(tab, check); check();
    const leaseId = crypto.randomUUID();
    lease = chrome.tabs.connect(tab.id!, { name: `agent-control:${leaseId}`, documentId: chromeDocumentId });
    browserSession.ports.add(lease);
    await new Promise<void>((resolve, reject) => {
      const timer = setTimeout(() => reject(new BrowserFailure("timeout", "Content execution lease did not open.")), 2000);
      lease!.onMessage.addListener(() => { clearTimeout(timer); resolve(); });
      lease!.onDisconnect.addListener(() => { clearTimeout(timer); reject(new BrowserFailure("run_paused", "Content execution lease closed.")); });
    });
    check();
    // Bind Chrome execution to the exact browser document, then recheck runtime identity inside it.
    browserSession.navigating = request.tool === "click" || request.tool === "type_text" || request.tool === "press_key" ? tab.id! : null;
    const link = request.tool === "click"
      ? browserSession.target?.elements.find(item => item.agent_id === request.arguments["element_id"]
          && item.tag === "a" && item.visible && item.enabled && !item.occluded)
      : undefined;
    if (link?.href) {
      const destination = new URL(link.href, tab.url!).href;
      if (destination !== tab.url && isAllowed(destination)
          && !new URL(destination).pathname.includes("/download/")) {
        browserSession.linkNavigation = { tabId: tab.id!, source: tab.url!, destination,
          sourceDocumentId: browserSession.target?.document_id ?? "", started: false };
      }
    }
    let execution: chrome.scripting.InjectionResult<ActionResult> | undefined;
    let executionError: unknown;
    try {
      [execution] = await chrome.scripting.executeScript({ target: { tabId: tab.id!, documentIds: [chromeDocumentId] },
      func: (name: string, tool: ToolRequestPayload, url: string, limits: object, lease: string) => {
        const api = (globalThis as Record<string, unknown>)[name] as ContentApi;
        return api.executeTool({ tool: tool.tool, arguments: tool.arguments, snapshotId: tool.snapshot_id,
          leaseId: lease, documentId: tool.tool === "wait_for_stable" ? undefined : tool.document_id, expectedUrl: url }, limits);
      }, args: [CONTENT, request, tab.url!, options(settings), leaseId] });
    } catch (error) { executionError = error; }
    check();
    if (browserSession.linkNavigation && (!execution?.result || execution.result.success)) {
      // The click may destroy its own JavaScript context. Settle in the side panel,
      // then bind a fresh snapshot to the exact destination before continuing.
      const page = await observeNavigationDestination(settings, check);
      return result(request, started, { page_state: page }, true);
    }
    if (executionError) throw executionError;
    if (!execution?.result) return fail("injection_failed", "Browser action returned no result.");
    const response = execution.result;
    const raw = response.data["page_state"] as PageState | undefined;
    if (raw) { raw.tab_id = tab.id!; browserSession.adopt(raw); }
    if (request.tool === "click") {
      await new Promise(resolve => setTimeout(resolve, 150)); check();
      const current = await allowedTab(tab.id!, settings);
      if (current.url !== tab.url) { response.page_changed = true; await waitForTab(tab.id!, 15000, check); }
    }
    check(); return response;
  } catch (error) {
    if (browserSession.preview?.unexpected) error = new PreviewFailure("preview_unexpected_download", "preview_wait");
    if (browserSession.preview !== null) browserSession.cancelPreview();
    if (error instanceof PreviewFailure) return { ...result(request, started), success: false, error: "permission_denied", data: { document_error: error.code, preview_stage: error.stage }, message: "Selected preview authorization failed." };
    const failure = error instanceof BrowserFailure ? error : new BrowserFailure("injection_failed", error instanceof AcquisitionFailure ? error.message : "Target became unavailable during browser execution.");
    return { ...result(request, started), success: false, error: failure.code, message: failure.message, data: failure.details };
  } finally {
    if (lease) { browserSession.ports.delete(lease); lease.disconnect(); }
    browserSession.expectedTab = null; browserSession.opening = false;
    browserSession.linkNavigation = null;
    browserSession.navigating = null; browserSession.closing = null;
  }
}

async function waitForTab(tabId: number, timeoutMs: number, check: () => void): Promise<void> {
  const deadline = performance.now() + timeoutMs;
  while (performance.now() < deadline) {
    check();
    let tab: chrome.tabs.Tab;
    try { tab = await chrome.tabs.get(tabId); } catch { return fail("target_tab_closed", "Target tab closed while waiting."); }
    check(); if (tab.status === "complete") return;
    await new Promise(resolve => setTimeout(resolve, 50));
  }
  fail("timeout", "Target tab did not finish loading.");
}


async function observeNavigationDestination(settings: ExtensionSettings, check: () => void): Promise<PageState> {
  const link = browserSession.linkNavigation!;
  const deadline = performance.now() + 15000;
  while (performance.now() < deadline) {
    check();
    const current = await allowedTab(link.tabId, settings); check();
    if ((current.pendingUrl && !sameNavigationDestination(current.pendingUrl, link.destination))
        || (current.url !== link.source && !sameNavigationDestination(current.url ?? "", link.destination))) {
      browserSession.invalidate("target_document_changed"); check();
      return fail("target_document_changed", "Navigation reached a different destination.");
    }
    if ((current.url !== link.source || link.started) && sameNavigationDestination(current.url ?? "", link.destination)
        && !current.pendingUrl && current.status === "complete") {
      const page = await extractPageState(link.tabId, settings, true, check); check();
      if (!sameNavigationDestination(page.url, link.destination)) return fail("target_document_changed", "Navigation destination changed during observation.");
      if (page.url !== link.source || page.document_id !== link.sourceDocumentId) return page;
    }
    await new Promise(resolve => setTimeout(resolve, 50));
  }
  return fail("timeout", "Navigation did not reach its expected destination within 15 seconds.");
}
