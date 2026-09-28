/** User-workflow-only identities. Never construct a download URL from these fields. */
import { isProductUrl } from "./product-scope";
export interface PreviewIdentity { parent_id: string; attachment_id: string; filename: string; version: string | null; }
export class PreviewFailure extends Error {
  constructor(readonly code: string, readonly stage = "validation") { super("Selected preview could not be authorized."); }
}
export const previewFail = (code = "preview_identity_mismatch", stage = "validation"): never => { throw new PreviewFailure(code, stage); };
function decode(raw: string): string {
  const value = decodeURIComponent(raw);
  if (/[\\%\x00-\x1f]/.test(value)) previewFail();
  return value;
}
export function safePreviewUrl(raw: string, source: string): URL {
  try {
    if (raw.length > 2048 || /[\s\\#]/.test(raw) || !isProductUrl(raw)) previewFail();
    // Inspect raw path before URL normalisation can hide traversal.
    const path = decode(raw.replace(/^https?:\/\/[^/]+/, "").split("?")[0]!);
    if (path.split("/").some(x => x === "." || x === "..")) previewFail();
    const url = new URL(raw);
    decode(url.search);
    if (url.origin !== new URL(source).origin || url.username || url.password ||
        [...url.searchParams.keys()].length !== new Set(url.searchParams.keys()).size) previewFail();
    return url;
  } catch { return previewFail(); }
}
export function safeAttachmentFilename(value: string): boolean {
  return value.length >= 5 && value.length <= 512 && !/[/\\%\x00-\x1f:*?"<>|]/.test(value) &&
    ![".", ".."].includes(value) && /^.+\.[A-Za-z0-9]{1,16}$/.test(value);
}
export function previewIdentity(raw: unknown, allowNonPdf = false): PreviewIdentity {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return previewFail();
  const value = raw as Record<string, unknown>;
  if (Object.keys(value).sort().join() !== "attachment_id,filename,parent_id,version" ||
      typeof value["parent_id"] !== "string" || !/^[0-9]{1,64}$/.test(value["parent_id"]) ||
      typeof value["attachment_id"] !== "string" || !/^[0-9]{1,64}$/.test(value["attachment_id"]) ||
      typeof value["filename"] !== "string" || value["filename"].length > 512 ||
      !safeAttachmentFilename(value["filename"]) || (!allowNonPdf && !/\.pdf$/i.test(value["filename"])) ||
      (value["version"] !== null && (typeof value["version"] !== "string" || !/^[1-9][0-9]{0,31}$/.test(value["version"])))) return previewFail();
  return value as unknown as PreviewIdentity;
}
export function validatePreview(raw: string, source: string, identity: PreviewIdentity): string {
  safePreviewUrl(raw, source);
  previewIdentity(identity, true);
  // The selected URL is bound by the owner/snapshot lease. Identity is then
  // corroborated by explicit data attributes on the visible preview dialog.
  return raw;
}
export function validatePreviewDownload(raw: string, source: string, expected: PreviewIdentity, observed: PreviewIdentity): string {
  const url = safePreviewUrl(raw, source);
  const filename = decode(url.pathname).split("/").at(-1);
  if (filename !== expected.filename ||
      expected.parent_id !== observed.parent_id || expected.attachment_id !== observed.attachment_id ||
      expected.filename !== observed.filename || (expected.version !== null && expected.version !== observed.version)) return previewFail();
  if (observed.version !== null && url.searchParams.get("version") !== observed.version) previewFail();
  return raw;
}
