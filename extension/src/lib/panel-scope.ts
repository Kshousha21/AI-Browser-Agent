/** Local panel identity only; never a URL or token from the target page. */
import { isAllowed } from "./allowlist";

export const PANEL_RESOURCE = "sidepanel/index.html";

export function panelPath(tabId: number): string {
  if (!Number.isSafeInteger(tabId) || tabId < 0) throw new Error("Invalid panel tab");
  return `${PANEL_RESOURCE}?tabId=${tabId}`;
}

export function panelTabId(url: string | undefined): number | null {
  if (!url) return null;
  try {
    const parsed = new URL(url);
    const id = parsed.searchParams.get("tabId");
    if (id === null || !/^(0|[1-9][0-9]*)$/.test(id)) return null;
    const tabId = Number(id);
    return Number.isSafeInteger(tabId) && url === chrome.runtime.getURL(panelPath(tabId)) ? tabId : null;
  } catch { return null; }
}

export function supportedPanelTab(tab: chrome.tabs.Tab): boolean {
  return tab.id !== undefined && tab.id >= 0 && !!tab.url && isAllowed(tab.url)
    && (!tab.pendingUrl || isAllowed(tab.pendingUrl));
}

/** Toolbar/extension-page user gesture only; never an activation callback. */
export function openPanelForTab(tab: chrome.tabs.Tab): Promise<void> {
  if (!supportedPanelTab(tab)) return Promise.resolve();
  // Lifecycle prepares the configured tab-specific path before the gesture.
  // Awaiting setOptions here loses the service worker's toolbar user gesture.
  return chrome.sidePanel.open({ tabId: tab.id! });
}

/** Reject an unbound/global panel or this resource opened as an ordinary tab. */
export async function resolvePanelOwner(url: string): Promise<number> {
  const tabId = panelTabId(url);
  if (tabId === null || await chrome.tabs.getCurrent()) throw new Error("Open the assistant from its owning tab.");
  const contexts = await chrome.runtime.getContexts({
    contextTypes: [chrome.runtime.ContextType.SIDE_PANEL], documentUrls: [url],
  });
  const options = await chrome.sidePanel.getOptions({ tabId });
  if (contexts.length !== 1 || !options.enabled || options.path !== panelPath(tabId)) {
    throw new Error("No matching tab-specific panel context.");
  }
  await chrome.tabs.get(tabId);
  return tabId;
}
