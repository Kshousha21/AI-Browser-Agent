/** Persistent content-world API used by the side panel. */

import { BrowserRuntime } from "./executor";
import { inspectAttachmentPreview } from "./attachment-preview";
export { inspectAttachmentPreview } from "./attachment-preview";
import type { BrowserToolCall, ContentApi, ExtractOptions, PageState } from "./types";
export { accessibleName, computeRole } from "./names";
export { isRendered as isVisible } from "./visibility";

const runtime = new BrowserRuntime();

export function extractPageState(options: ExtractOptions): PageState {
  return runtime.extract(options);
}

export async function executeTool(
  call: BrowserToolCall,
  options: ExtractOptions,
) {
  return runtime.execute(call, options);
}

export function registryStats(): { reverse: number; snapshots: number } {
  return runtime.stats();
}

export function bindTab(tabId: number): void { runtime.bindTab(tabId); }
export function invalidate(): void { runtime.invalidate(); }
export function identity() { return runtime.identity(); }

export const api: ContentApi = {
  inspectAttachmentPreview,
  bindTab, invalidate, identity,
  extractPageState,
  executeTool,
  registryStats,
};
