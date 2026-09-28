/** Normal browser navigation/download; never fetch, replay headers, or read cookies. */
import type { ToolRequestPayload } from "@shared/protocol";
import { isAllowed } from "../lib/allowlist";
import type { ExtensionSettings } from "../lib/settings";
import { safeAttachmentFilename, safePreviewUrl } from "../lib/attachment-preview";

const MAX_BYTES = 25 * 1024 * 1024;
export class AcquisitionFailure extends Error {}

export async function acquire(request: ToolRequestPayload, _settings: ExtensionSettings,
                              currentUrl: string, check: () => void, validate: () => Promise<void>,
                              previewLease?: chrome.runtime.Port, userDownload = false, signal?: AbortSignal): Promise<Record<string, unknown>> {
  const rawUrl = request.arguments["url"], nonce = request.arguments[userDownload ? "download_action_id" : "acquisition_id"];
  if (typeof rawUrl !== "string" || typeof nonce !== "string" || !/^[a-f0-9]{32}$/.test(nonce) ||
      !isAllowed(rawUrl) || new URL(rawUrl).origin !== new URL(currentUrl).origin ||
      (!userDownload && !new URL(rawUrl).pathname.toLowerCase().endsWith(".pdf")) || request.tab_id == null) throw new Error("invalid acquisition");
  const url = new URL(rawUrl).href;
  const selectedFilename = request.arguments["filename"];
  if (userDownload) {
    const parsed = safePreviewUrl(rawUrl, currentUrl);
    const path = decodeURIComponent(parsed.pathname).match(/\/([^/]+)$/);
    if (typeof selectedFilename !== "string" || !safeAttachmentFilename(selectedFilename) || !path || path[1] !== selectedFilename) throw new Error("invalid download");
  }
  const filename = userDownload ? selectedFilename as string : `AIBrowserAgent/acq-${nonce}.pdf`;
  const lease = previewLease ?? chrome.runtime.connect({ name: "document-acquisition-guard" });
  let downloadId: number | null = null;
  let leaseClosed = false;
  let completed = false;
  const started = Date.now();
  let changedCheck = (): void => {};
  let rejectDownload = (_error: unknown): void => {};
  const disconnected = () => { leaseClosed = true; rejectDownload(new AcquisitionFailure("Acquisition guard disconnected.")); };
  const cancelled = () => { rejectDownload(new AcquisitionFailure("Acquisition cancelled.")); lease.disconnect(); };
  lease.onDisconnect.addListener(disconnected);
  signal?.addEventListener("abort", cancelled, { once: true });
  const created = (item: chrome.downloads.DownloadItem): void => {
    if (downloadId === null && item.url === url && Date.parse(item.startTime) >= started - 1000) {
      downloadId = item.id;
      if (!leaseClosed) lease.postMessage({ downloadId });
      changedCheck();
    }
  };
  const naming = (item: chrome.downloads.DownloadItem, suggest: (suggestion?: chrome.downloads.DownloadFilenameSuggestion) => void): void => {
    if (item.url === url && Date.parse(item.startTime) >= started - 1000) {
      created(item); suggest({ filename, conflictAction: "uniquify" });
    } else suggest();
  };
  const changed = (delta: chrome.downloads.DownloadDelta): void => { if (delta.id === downloadId) changedCheck(); };
  // An idle port heartbeat is Chrome worker resource retention, not readiness or
  // retry. Only actual browser events advance the document operation.
  const heartbeat = window.setInterval(() => { if (!leaseClosed) lease.postMessage({ ping: true }); }, 5000);
  try {
    if (signal?.aborted) throw new AcquisitionFailure("Acquisition cancelled.");
    await new Promise<void>((resolve, reject) => {
      const cleanup = () => { lease.onMessage.removeListener(ready); lease.onDisconnect.removeListener(closed); signal?.removeEventListener("abort", aborted); };
      const closed = () => { cleanup(); reject(new AcquisitionFailure("Acquisition guard disconnected.")); };
      const aborted = () => { cleanup(); reject(new AcquisitionFailure("Acquisition cancelled.")); };
      const ready = (message: { ready?: boolean; stage?: string }) => {
        if (message.ready === undefined) return;
        const stage = ["target", "regex", "rules"].includes(message.stage || "") ? message.stage : "setup";
        cleanup(); message.ready ? resolve() : reject(new AcquisitionFailure(`Acquisition guard unavailable (${stage}).`));
      };
      lease.onMessage.addListener(ready); lease.onDisconnect.addListener(closed); signal?.addEventListener("abort", aborted, { once: true });
      lease.postMessage({ tabId: request.tab_id, url, ...(previewLease ? { promote: true } : {}),
        ...(userDownload ? { userActionId: request.arguments["user_action_id"], filename } : {}) });
    });
    check();
    // A denied/busy panel must never observe or rename another owner's download.
    chrome.downloads.onCreated.addListener(created);
    chrome.downloads.onDeterminingFilename.addListener(naming);
    chrome.downloads.onChanged.addListener(changed);
    await validate(); // Guard setup cannot extend the authority of a changed DOM.
    check();
    const receipt = new Promise<Record<string, unknown>>((resolve, reject) => {
      let settled = false, inspecting = false, pending = false;
      rejectDownload = error => { if (!settled) { settled = true; reject(error); } };
      changedCheck = () => {
        if (settled || downloadId === null) return;
        if (inspecting) { pending = true; return; }
        inspecting = true;
        void (async () => {
          try {
            check(); if (leaseClosed || signal?.aborted) throw new AcquisitionFailure("Acquisition cancelled.");
            const [item] = await chrome.downloads.search({ id: downloadId! });
            if (settled) return;
            check(); if (leaseClosed || signal?.aborted) throw new AcquisitionFailure("Acquisition cancelled.");
            if (!item || item.url !== url || (item.finalUrl && item.finalUrl !== url) || item.state === "interrupted" ||
                item.bytesReceived > MAX_BYTES || item.totalBytes > MAX_BYTES || item.fileSize > MAX_BYTES) throw new AcquisitionFailure("Download failed validation.");
            if (item.state !== "complete") return;
            if (item.finalUrl !== url || !item.exists || !["safe", "deepScannedSafe", "allowlistedByPolicy"].includes(item.danger) ||
                item.fileSize <= 0 || item.bytesReceived !== item.fileSize ||
                (!userDownload && !item.filename.replace(/\\/g, "/").endsWith("/" + filename))) throw new AcquisitionFailure("Download failed validation.");
            settled = true; completed = true;
            resolve(userDownload ? { download_id: item.id, state: "complete" } :
              { acquisition_id: nonce, download_id: item.id, state: "complete", filename: item.filename, size: item.fileSize });
          } catch (error) { rejectDownload(error); }
          finally { inspecting = false; if (pending) { pending = false; changedCheck(); } }
        })();
      };
    });
    // Consume rejection immediately if cancellation happens while navigation is
    // still resolving. No unhandled rejection or late completion can revive it.
    void receipt.catch(() => undefined);
    if (signal?.aborted || leaseClosed) throw new AcquisitionFailure("Acquisition cancelled.");
    await chrome.tabs.update(request.tab_id, { url });
    check();
    return await receipt;
  } finally {
    clearInterval(heartbeat);
    chrome.downloads.onCreated.removeListener(created);
    chrome.downloads.onDeterminingFilename.removeListener(naming);
    chrome.downloads.onChanged.removeListener(changed);
    signal?.removeEventListener("abort", cancelled); lease.onDisconnect.removeListener(disconnected);
    if (downloadId !== null && !completed) await chrome.downloads.cancel(downloadId).catch(() => undefined);
    lease.disconnect(); // Worker removes redirect guard even if the panel closes.
  }
}
