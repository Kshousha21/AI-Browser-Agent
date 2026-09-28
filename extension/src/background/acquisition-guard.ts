/** Events-only cleanup lease. No socket, model decisions, cookies or HTTP client. */
import { isProductUrl } from "../lib/product-scope";
import { panelTabId } from "../lib/panel-scope";
import { previewIdentity, safeAttachmentFilename, safePreviewUrl, validatePreview, validatePreviewDownload } from "../lib/attachment-preview";
import type { PreviewIdentity } from "../lib/attachment-preview";

// Downloads events do not identify the initiating tab. Never let two panel
// documents attribute/name/cancel the same browser download concurrently.
let activeAcquisition: chrome.runtime.Port | null = null;

chrome.runtime.onConnect.addListener(port => {
  const ownerTabId = panelTabId(port.sender?.url);
  if (port.name !== "document-acquisition-guard" || ownerTabId === null ||
      port.sender?.id !== chrome.runtime.id || port.sender.tab) return;
  let closed = false;
  let installed: Promise<void> | null = null;
  let ids: number[] = [];
  let downloadId: number | null = null;
  let stage = "target";
  let previewUrl: string | null = null;
  let attachment: PreviewIdentity | null = null;
  let previewMode = false;
  let unexpected = false;
  let promotionStarted = false;
  let humanId: string | null = null;
  let humanFilename: string | null = null;
  const started = Date.now();
  const created = (item: chrome.downloads.DownloadItem): void => {
    // Download events have no tab ID: cancel only the exact preview response or
    // a download explicitly attributed to its exact referrer, never other Chrome activity.
    if (!previewMode || closed || Date.parse(item.startTime) < started - 1000 ||
        (item.url !== previewUrl && item.referrer !== previewUrl)) return;
    unexpected = true;
    void chrome.downloads.cancel(item.id).catch(() => undefined);
    port.postMessage({ unexpected: true });
  };
  const cleanup = async (): Promise<void> => {
    closed = true;
    if (previewUrl !== null) chrome.downloads.onCreated.removeListener(created);
    await installed?.catch(() => undefined);
    if (downloadId !== null) await chrome.downloads.cancel(downloadId).catch(() => undefined);
    const removed = !ids.length || await chrome.declarativeNetRequest.updateSessionRules({ removeRuleIds: ids })
      .then(() => true, () => false);
    // Failed rule cleanup stays fail-closed; it cannot admit another owner.
    if (removed && activeAcquisition === port) activeAcquisition = null;
  };
  port.onDisconnect.addListener(() => { void cleanup(); });
  port.onMessage.addListener((data: Record<string, unknown>) => {
    if (data["downloadId"] !== undefined && Number.isInteger(data["downloadId"])) {
      downloadId = data["downloadId"] as number; return;
    }
    if (data["ping"] || closed) return;
    const promoting = data["promote"] === true;
    if (installed && !promoting) return;
    if (promoting && (!installed || !previewMode || unexpected || promotionStarted)) { port.postMessage({ ready: false, stage }); return; }
    if (promoting) promotionStarted = true;
    const previous = installed;
    installed = (async () => {
      if (promoting) await previous;
      const tabId = data["tabId"], rawUrl = data["url"];
      if (tabId !== ownerTabId) throw new Error("invalid guard owner");
      if (activeAcquisition !== null && !(promoting && activeAcquisition === port)) throw new Error("another acquisition is active");
      activeAcquisition = port;
      if (typeof tabId !== "number" || !Number.isInteger(tabId) || typeof rawUrl !== "string" || !isProductUrl(rawUrl)) throw new Error("invalid guard");
      const tab = await chrome.tabs.get(tabId);
      const url = new URL(rawUrl);
      if (!tab.url || new URL(tab.url).origin !== url.origin) throw new Error("invalid guard");
      const suppliedHuman = data["userActionId"];
      if (!promoting && suppliedHuman !== undefined) {
        if (typeof suppliedHuman !== "string" || !/^[a-f0-9-]{32,36}$/i.test(suppliedHuman) ||
            typeof data["filename"] !== "string" || !safeAttachmentFilename(data["filename"])) throw new Error("invalid human target");
        humanId = suppliedHuman; humanFilename = data["filename"];
      }
      if (promoting && suppliedHuman !== undefined && (suppliedHuman !== humanId || data["filename"] !== humanFilename)) throw new Error("changed human target");
      if (!promoting && data["preview"] === true) {
        attachment = previewIdentity(data["attachment"], humanId !== null);
        if (humanFilename !== null && attachment.filename !== humanFilename) throw new Error("changed human target");
        previewUrl = validatePreview(rawUrl, tab.url, attachment);
        previewMode = true;
        chrome.downloads.onCreated.addListener(created);
      } else {
        if ((humanId === null && !url.pathname.toLowerCase().endsWith(".pdf"))) throw new Error("invalid guard");
        if (humanId !== null) {
          const parsed = safePreviewUrl(rawUrl, tab.url);
          const path = decodeURIComponent(parsed.pathname).match(/\/([^/]+)$/);
                if (!path || path[1] !== humanFilename) throw new Error("invalid human target");
        }
        if (promoting) {
          if (tab.url !== previewUrl || !attachment || unexpected) throw new Error("preview changed");
          validatePreviewDownload(rawUrl, previewUrl!, attachment, attachment);
        }
      }
      const regex = "^" + url.href.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + "$";
      stage = "regex";
      if (!(await chrome.declarativeNetRequest.isRegexSupported({ regex })).isSupported) throw new Error("invalid guard");
      const id = ids[0] ?? 1000000 + crypto.getRandomValues(new Uint32Array(1))[0]! % 100000000;
      const removeRuleIds = [...ids];
      ids = [id, id + 1];
      stage = "rules";
      if (closed || unexpected) throw new Error("guard closed");
      await chrome.declarativeNetRequest.updateSessionRules({ removeRuleIds, addRules: [
        { id, priority: 1, action: { type: chrome.declarativeNetRequest.RuleActionType.BLOCK },
          condition: { tabIds: [tabId], resourceTypes: [chrome.declarativeNetRequest.ResourceType.MAIN_FRAME] } },
        { id: id + 1, priority: 2, action: { type: chrome.declarativeNetRequest.RuleActionType.ALLOW },
          condition: { tabIds: [tabId], regexFilter: regex, isUrlFilterCaseSensitive: true,
            resourceTypes: [chrome.declarativeNetRequest.ResourceType.MAIN_FRAME] } },
      ] });
      if (closed || unexpected) throw new Error("guard closed");
      if (promoting) previewMode = false;
      if (!closed) port.postMessage({ ready: true });
    })();
    void installed.catch(() => { if (!closed) port.postMessage({ ready: false, stage }); void cleanup(); });
  });
});
