/** Native tab-specific panel availability. No backend/session/browser-action routing. */
import { panelPath, supportedPanelTab } from "../lib/panel-scope";
import { isProductUrl } from "../lib/product-scope";

export class TabPanels {
  private pending = new Map<number, Promise<void>>();
  private versions = new Map<number, number>();
  private serial = 0;

  constructor() {}

  async initialize(): Promise<void> {
    // Also retires the enabled global default saved by older extension builds.
    await chrome.sidePanel.setOptions({ enabled: false });
    await chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: false });
    await this.refreshAll();
  }

  async refreshAll(): Promise<void> {
    const tabs = await chrome.tabs.query({});
    await Promise.all(tabs.filter(tab => tab.id !== undefined).map(tab => this.refresh(tab.id!)));
  }

  refresh(tabId: number): Promise<void> {
    const version = ++this.serial;
    this.versions.set(tabId, version);
    const update = (this.pending.get(tabId) ?? Promise.resolve()).then(async () => {
      let tab: chrome.tabs.Tab;
      try { tab = await chrome.tabs.get(tabId); } catch { return; } // The tab closed.
      if (this.versions.get(tabId) !== version) return;
      const enabled = !tab.url || supportedPanelTab(tab);
      const options = await chrome.sidePanel.getOptions({ tabId });
      if (this.versions.get(tabId) !== version) return;
      const path = panelPath(tabId);
      // Reapplying the same path must not replace an existing panel document.
      if (options.enabled !== enabled || (enabled && options.path !== path)) {
        await chrome.sidePanel.setOptions({ tabId, ...(enabled ? { path } : {}), enabled });
      }
    }).catch(() => { console.warn("[ai-browser-agent] tab panel configuration unavailable"); }).finally(() => {
      if (this.versions.get(tabId) === version) {
        this.pending.delete(tabId); this.versions.delete(tabId);
      }
    });
    this.pending.set(tabId, update);
    return update;
  }

  /** Called only from Chrome's toolbar user gesture, never from tab activation. */
  open(tab: chrome.tabs.Tab): Promise<boolean> {
    if (!Number.isSafeInteger(tab.id) || tab.id! < 0 || !tab.url || !isProductUrl(tab.url)
        || (tab.pendingUrl && !isProductUrl(tab.pendingUrl))) return Promise.resolve(false);
    return chrome.sidePanel.open({ tabId: tab.id! }).then(() => true);
  }
}

export function installTabPanels(): void {
  const panels = new TabPanels();
  const initialize = (): void => { void panels.initialize().catch(() => console.warn("[ai-browser-agent] panel setup unavailable")); };
  chrome.runtime.onInstalled.addListener(initialize);
  chrome.runtime.onStartup.addListener(initialize);
  chrome.tabs.onUpdated.addListener((id, info) => {
    if (info.url || info.status) void panels.refresh(id);
  });
  chrome.tabs.onCreated.addListener(tab => { if (tab.id !== undefined) void panels.refresh(tab.id); });
  chrome.action.onClicked.addListener(tab => {
    // Open synchronously while Chrome preserves the toolbar user gesture.
    void panels.open(tab).then(opened => {
      if (!opened) { console.info("[ai-browser-agent] unsupported_page"); return; }
      if (tab.id === undefined) return;
    }).catch(() => console.warn("[ai-browser-agent] panel_open_unavailable"));
  });
  initialize();
}
