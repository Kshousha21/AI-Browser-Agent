/** Two limited preview contracts; authenticated end-to-end acceptance is unverified.
 * Reads one visible attachment dialog and its Download link, never page/PDF text.
 */
import { DOCUMENT_ID, instrumentState } from "./page";
import { accessibleName } from "./names";
import { isRendered } from "./visibility";
import { PreviewFailure, previewFail, previewIdentity, validatePreview, validatePreviewDownload } from "../lib/attachment-preview";
import type { PreviewIdentity } from "../lib/attachment-preview";
export { PreviewFailure } from "../lib/attachment-preview";

const DIALOG = 'dialog[open],[role="dialog"]';
const ATTRIBUTES = [["data-page-id", "parent_id"], ["data-attachment-id", "attachment_id"],
  ["data-attachment-filename", "filename"], ["data-attachment-version", "version"]] as const;

function inspect(url: string, expected: PreviewIdentity, stage: (name: string) => void, allowNonPdf: boolean) {
  stage("destination");
  validatePreview(url, location.href, expected);
  if (location.href !== url || document.contentType !== "text/html") return previewFail();
  stage("dialog");
  const roots = [...document.querySelectorAll<HTMLElement>(DIALOG)]
    .filter(root => isRendered(root) && root.hasAttribute("data-attachment-id"));
  if (roots.length === 0) return previewFail("preview_control_missing");
  if (roots.length !== 1) return previewFail("preview_ambiguous");
  const root = roots[0]!;
  const links = [...root.querySelectorAll<HTMLAnchorElement>("a[href]")];
  if (links.length > 100) return previewFail("preview_ambiguous");
  const controls = links.filter(link => link.closest(DIALOG) === root && isRendered(link) &&
    /^(Download|Herunterladen)$/i.test(accessibleName(link).trim()) &&
    !link.closest('[aria-disabled="true"],[inert],[disabled]'));
  if (!controls.length) return previewFail("preview_control_missing");
  if (controls.length !== 1) return previewFail("preview_ambiguous");
  const link = controls[0]!;
  stage("preview_title");
  const raw = link.getAttribute("href")!;
  stage("observed_href");
  // Validate pre-normalized path too; URL() alone hides ../ segments.
  if (/[\\\s#]/.test(raw) || decodeURIComponent(raw.split(/[?#]/)[0]!).split("/").some(x => x === "." || x === "..")) return previewFail();
  const href = new URL(raw, url).href;
  stage("attachment_metadata");
  const observed = previewIdentity({
    parent_id: root.getAttribute("data-page-id"),
    attachment_id: root.getAttribute("data-attachment-id"), filename: root.getAttribute("data-attachment-filename"),
    version: root.getAttribute("data-attachment-version") }, allowNonPdf);
  for (const node of [root, link]) for (const [attribute, key] of ATTRIBUTES) {
    const value = node.getAttribute(attribute);
    if (value !== null && value !== observed[key]) return previewFail();
  }
  validatePreviewDownload(href, url, expected, observed);
  stage("snapshot");
  return { preview: { url: href, attachment: observed }, page_state: {
    url, document_id: DOCUMENT_ID, snapshot_id: `s-preview-${crypto.randomUUID()}`,
    navigation_id: instrumentState().navigation, title: "", visible_text: "", elements: [], headings: [],
    frames: [], truncated: { text: true, elements: true }, opaque_regions: [], warnings: [], dialogs: [],
    looks_like_login: false, semantic_regions: [] } };
}

const observerLeases = new Map<string, AbortController>();
if (typeof chrome !== "undefined" && chrome.runtime?.onConnect) {
  chrome.runtime.onConnect.addListener(port => {
    if (!port.name.startsWith("preview-observer:") || port.sender?.id !== chrome.runtime.id) return;
    const id = port.name.slice("preview-observer:".length);
    if (!id || observerLeases.has(id)) { port.disconnect(); return; }
    const controller = new AbortController(); observerLeases.set(id, controller);
    port.onDisconnect.addListener(() => { controller.abort(); observerLeases.delete(id); });
    port.postMessage({ ready: true });
  });
}

/** No elapsed-time readiness inference. A document-scoped port owns the observer. */
export async function inspectAttachmentPreview(url: string, identity: PreviewIdentity, wait: boolean, allowNonPdf = false, leaseId?: string) {
  const controller = leaseId ? observerLeases.get(leaseId) : undefined;
  if (leaseId && !controller) return { error: "operation_interrupted", stage: "observer_lease" };
  return waitForPreviewState(stage => inspect(url, previewIdentity(identity, allowNonPdf), stage, allowNonPdf), wait, controller?.signal);
}

export async function waitForPreviewState(read: (stage: (name: string) => void) => ReturnType<typeof inspect>, wait: boolean, signal?: AbortSignal) {
  return await new Promise<ReturnType<typeof inspect> | { error: string; stage: string }>(resolve => {
    let finished = false;
    let stage = "request_identity";
    const finish = (value: ReturnType<typeof inspect> | { error: string; stage: string }) => {
      if (finished) return; finished = true;
      observer.disconnect(); document.removeEventListener("readystatechange", check);
      document.removeEventListener("load", check, true); document.removeEventListener("transitionend", check, true);
      window.removeEventListener("pagehide", cancel); signal?.removeEventListener("abort", cancel); resolve(value);
    };
    const cancel = () => finish({ error: "operation_interrupted", stage: "observer_lease" });
    const check = () => {
      if (finished) return;
      try {
        // A matching dialog may be ready before unrelated page resources finish.
        finish(read(value => { stage = value; }));
      } catch (error) {
        const code = error instanceof PreviewFailure ? error.code : "preview_identity_mismatch";
        if (!wait || code !== "preview_control_missing") finish({ error: code, stage });
      }
    };
    const observer = new MutationObserver(check);
    if (wait) {
      observer.observe(document, { subtree: true, attributes: true, childList: true, characterData: true });
      document.addEventListener("readystatechange", check);
      document.addEventListener("load", check, true); document.addEventListener("transitionend", check, true);
      window.addEventListener("pagehide", cancel); signal?.addEventListener("abort", cancel, { once: true });
    }
    if (signal?.aborted) { cancel(); return; }
    check();
  });
}
