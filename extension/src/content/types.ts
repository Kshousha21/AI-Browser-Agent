import type { ActionResult, PageState } from "@shared/protocol";

export interface ExtractOptions {
  maxVisibleTextChars: number;
  maxElements: number;
}

export interface ElementDescriptor {
  agentId: string;
  tag: string;
  role: string;
  name: string;
  documentOrder: number;
  href: string;
  formAction: string;
  formMethod: string;
  formTarget: string;
  domId: string;
  testId: string;
  dataKey: string;
  semanticRegion: string;
  selected: boolean;
}

export interface ResolvedElement {
  element: HTMLElement;
  descriptor: ElementDescriptor;
  reresolved: boolean;
}

export interface BrowserToolCall {
  leaseId?: string;
  documentId?: string;
  expectedUrl?: string;
  tool: string;
  arguments: Record<string, unknown>;
  snapshotId: string;
}

export interface ContentApi {
  inspectAttachmentPreview: typeof import("./attachment-preview").inspectAttachmentPreview;
  bindTab(tabId: number): void;
  invalidate(): void;
  identity(): { document_id: string; snapshot_id: string; unchanged: boolean };
  extractPageState(options: ExtractOptions): PageState;
  executeTool(call: BrowserToolCall, options: ExtractOptions): Promise<ActionResult>;
  registryStats(): { reverse: number; snapshots: number };
}

export type { ActionResult, PageElement, PageState, PageRegion } from "@shared/protocol";
