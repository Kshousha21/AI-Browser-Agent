/** A local UI-created, one-file grant. Tool arguments cannot create this authority. */
import type { ToolRequestPayload } from "@shared/protocol";
import { previewFail, safeAttachmentFilename } from "../lib/attachment-preview";

export type UserFileAction = "preview" | "download" | "prepare";
export interface FileSelection { resultRef: string; filename: string; fileType: string | null; }
export interface ResearchSelection extends FileSelection { sourceKey: string; }
// Only an admitted, frozen ResearchGrant can create later per-file authority.
// This symbol is not exported and can never be supplied in a tool/wire payload.
const RESEARCH_ADMISSION = Symbol("research-operation-admission");
export class UserFileGrant {
  private previewId: string | null = null;
  private downloaded = false;
  readonly selection: Readonly<FileSelection>;
  constructor(readonly id: string, readonly action: UserFileAction, _expiresAt: number,
              selection: FileSelection, readonly owner: number, readonly version: number, readonly epoch: number,
              admission?: typeof RESEARCH_ADMISSION) {
    if ((admission !== undefined && admission !== RESEARCH_ADMISSION) ||
        !id || !["preview", "download", "prepare"].includes(action) ||
        !selection.resultRef ||
        !safeAttachmentFilename(selection.filename) || (selection.fileType !== null &&
          selection.filename.split(".").pop()!.toLowerCase() !== selection.fileType.toLowerCase()) ||
        (action === "prepare" && !/\.pdf$/i.test(selection.filename))) previewFail("operation_interrupted");
    this.selection = Object.freeze({ ...selection });
  }
  check(request: ToolRequestPayload, owner: number | null, version: number, epoch: number): void {
    if (!["get_page_state", "open_attachment_preview", "inspect_attachment_preview", "acquire_attachment", "download_attachment"].includes(request.tool) ||
        this.owner !== owner || this.version !== version || this.epoch !== epoch ||
        request.arguments["user_action_id"] !== this.id || request.arguments["user_result_ref"] !== this.selection.resultRef) {
      previewFail("operation_interrupted", "human_grant");
    }
  }
  preview(request: ToolRequestPayload): void {
    const id = request.arguments["preview_id"], url = request.arguments["url"];
    const attachment = request.arguments["attachment"] as { filename?: unknown } | undefined;
    if (this.previewId !== null || typeof id !== "string" || typeof url !== "string" ||
        attachment?.filename !== this.selection.filename || this.downloaded) return previewFail("preview_identity_mismatch", "human_file");
    this.previewId = id;
  }
  inspect(request: ToolRequestPayload): void {
    if (!this.previewId || request.arguments["preview_id"] !== this.previewId || this.downloaded) previewFail("preview_identity_mismatch", "human_file");
  }
  download(request: ToolRequestPayload, filename: string): void {
    if (this.downloaded || filename !== this.selection.filename || this.action === "preview" ||
        (request.tool === "acquire_attachment" ? this.action !== "prepare" : this.action !== "download") ||
        (this.previewId !== null ? request.arguments["preview_id"] !== this.previewId : request.arguments["preview_id"] !== undefined)) {
      previewFail("preview_identity_mismatch", "human_file");
    }
    this.downloaded = true;
  }
}

/** A fresh bounded plan admits immutable operations; elapsed time is not authority. */
export class ResearchGrant {
  private source: string | null = null;
  get sourceUrl(): string | null { return this.source; }
  private selections = new Map<string, Readonly<ResearchSelection>>();
  private used = new Set<string>();
  private returned = false;
  private active = false;
  private specific: Readonly<{ selection: Readonly<FileSelection>; sourceUrl: string }> | null = null;
  get hasSpecificAdmission(): boolean { return this.specific !== null; }
  constructor(readonly id: string, readonly maximum: number, readonly owner: number,
              readonly version: number, readonly epoch: number) {
    if (!id || !Number.isSafeInteger(maximum) || maximum < 1 || maximum > 5) previewFail("operation_interrupted", "research_grant");
  }
  check(id: unknown, owner: number | null, version: number, epoch: number): void {
    if (id !== this.id || owner !== this.owner || version !== this.version || epoch !== this.epoch) previewFail("operation_interrupted", "research_grant");
  }
  admitSpecific(selection: FileSelection, sourceUrl: string, observedUrl: string | undefined, expiresAt: number): void {
    if (this.maximum !== 1 || this.specific || this.sourceUrl !== null || !observedUrl || sourceUrl !== observedUrl) {
      return previewFail("operation_interrupted", "research_specific");
    }
    const admitted = new UserFileGrant(this.id, "prepare", expiresAt, selection, this.owner, this.version, this.epoch);
    this.specific = Object.freeze({ selection: admitted.selection, sourceUrl });
  }
  plan(sourceUrl: string, observedUrl: string | undefined, files: ResearchSelection[], _expiresAt: number): void {
    const selected = this.specific;
    if (selected && (files.length !== 1 || sourceUrl !== selected.sourceUrl ||
        files[0]?.resultRef !== selected.selection.resultRef || files[0]?.filename !== selected.selection.filename ||
        files[0]?.fileType !== selected.selection.fileType)) return previewFail("operation_interrupted", "research_specific");
    if (this.sourceUrl !== null || !observedUrl || sourceUrl !== observedUrl || !files.length || files.length > this.maximum ||
        new Set(files.map(file => file.sourceKey)).size !== files.length || new Set(files.map(file => file.resultRef)).size !== files.length ||
        files.some(file => !/^[a-f0-9]{64}$/.test(file.sourceKey) || !file.resultRef || !safeAttachmentFilename(file.filename) ||
          !/\.pdf$/i.test(file.filename) || file.fileType?.toLowerCase() !== "pdf")) return previewFail("operation_interrupted", "research_plan");
    this.source = sourceUrl;
    this.selections = new Map(files.map(file => [file.sourceKey, Object.freeze({ ...file })]));
  }
  file(sourceKey: string, file: FileSelection, currentUrl: string | undefined): UserFileGrant {
    const expected = this.selections.get(sourceKey);
    if (!expected || this.used.has(sourceKey) || this.active || currentUrl !== this.sourceUrl ||
        file.filename !== expected.filename || file.fileType !== expected.fileType ||
        (!this.returned && file.resultRef !== expected.resultRef)) return previewFail("operation_interrupted", "research_file");
    const grant = new UserFileGrant(this.id, "prepare", 0, file, this.owner, this.version, this.epoch, RESEARCH_ADMISSION);
    this.used.add(sourceKey); this.active = true; this.returned = false;
    return grant;
  }
  returnTarget(url: unknown): string {
    if (!this.sourceUrl || !this.active || url !== this.sourceUrl) return previewFail("operation_interrupted", "research_return");
    return this.sourceUrl;
  }
  returnedToResults(): void { this.active = false; this.returned = true; }
}
