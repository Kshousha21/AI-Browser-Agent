/** Authoritative, transient attachment choices. Text from a model never enters this component. */
export type FileAction = "preview" | "download" | "prepare";
export interface FileChoice {
  result_ref: string;
  title: string | null;
  attachment_filename?: string | null;
  content_type?: string | null;
  file_type?: string | null;
  last_modified?: string | null;
  publisher?: string | null;
  year?: string | number | null;
}
export interface FileChoices {
  offer_id: string | null;
  results: FileChoice[];
  expires_at_ms?: number;
  expires_in_ms?: number;
  freshness?: "state_bound";
  state?: string;
}
export interface FileCompletion {
  title: string;
  file_type: string | null;
  action: FileAction;
  status: string;
  download_id?: number;
  document_id?: string;
}
export interface FileClickAuthority {
  expiresAt: number;
  filename: string;
  fileType: string | null;
}
type Action = (kind: FileAction, selection: {offer_id: string; result_ref: string}, authority: FileClickAuthority) => void;
const EXPIRED = "Results changed or were cleared; refresh results";

function text(value: unknown, fallback = ""): string {
  if (typeof value !== "string" && typeof value !== "number") return fallback;
  const label = String(value).replace(/[\u0000-\u001f\u007f]/g, " ").trim().slice(0, 512);
  // Neither a completion payload nor metadata may expose an absolute local path.
  if (/^(?:[a-z]:[\\/]|\\\\|file:|\/(?:[^/]|$))/i.test(label)) return fallback;
  return label || fallback;
}

export class FileCards {
  private choices: HTMLElement | null = null;
  private readonly outcomes: HTMLElement[] = [];
  private offerId: string | null = null;
  private busy = false;
  private connected = false;
  private generation = 0;
  private refreshButton: HTMLButtonElement | null = null;
  private choiceActions: HTMLElement[] = [];
  private choiceButtons: HTMLButtonElement[] = [];
  private readonly actionsByRef = new Map<string, HTMLButtonElement[]>();
  private outcomeButtons: HTMLButtonElement[] = [];
  private pendingChoice: {ref: string; generation: number; action: FileAction} | null = null;
  private readonly preparedChoices = new Map<string, string>();

  constructor(private readonly container: HTMLElement, private readonly onAction: Action,
    private readonly onRefresh: () => void, private readonly onAsk: (documentId: string) => void,
    private readonly onShowDownload: (downloadId: number) => void,
    private readonly onAskChoice?: (selection: {offer_id: string; result_ref: string}, authority: FileClickAuthority) => void,
    private readonly onAskPrepared?: (documentId: string) => void) {}

  setBusy(value: boolean): void { this.busy = value; this.update(); }
  setConnected(value: boolean): void {
    this.connected = value;
    if (!value && this.offerId) this.invalidate();
    this.update();
  }

  showChoices(payload: FileChoices): void {
    this.removeChoices();
    if (!this.connected) return;
    const rows = payload.results.filter(row => row.content_type === "attachment").slice(0, 10);
    if (!payload.offer_id || !rows.length) return;
    this.offerId = payload.offer_id;
    const section = document.createElement("section");
    section.className = "file-cards assistant-file-cards";
    const notice = document.createElement("p"); notice.className = "file-card-detail";
    notice.textContent = "Your click authorizes only this file. Prepare locally downloads and indexes the PDF without a provider call. Preview may load or render the attachment.";
    section.append(notice);
    this.choices = section;
    const generation = this.generation;
    for (const row of rows) {
      const card = this.card(row.attachment_filename || row.title || "Attachment", row.file_type ?? null);
      const metadata = [text(row.publisher), text(row.year), text(row.last_modified)].filter(Boolean);
      if (metadata.length) {
        const detail = document.createElement("p"); detail.className = "file-card-detail";
        detail.textContent = metadata.join(" · "); card.append(detail);
      }
      const actions = document.createElement("div"); actions.className = "file-card-actions";
      const add = (label: string, kind: FileAction): void => {
        const button = this.button(label, () => {
          if (!this.connected || this.busy || generation !== this.generation || !this.offerId) return;
          const selection = {offer_id: this.offerId, result_ref: row.result_ref};
          const authority = {expiresAt: 0, filename: row.attachment_filename || row.title || "",
            fileType: row.file_type ?? null};
          // Reserve synchronously. A new click gets a NEW one-use grant; the
          // displayed catalogue is not itself an approval and is not consumed.
          this.busy = true;
          this.pendingChoice = {ref: row.result_ref, generation, action: kind};
          this.update();
          this.onAction(kind, selection, authority);
        });
        this.choiceButtons.push(button); actions.append(button);
      };
      add("Open preview", "preview"); add("Download", "download");
      if (row.file_type?.toLowerCase() === "pdf") add("Prepare locally", "prepare");
      if (row.file_type?.toLowerCase() === "pdf" && this.onAskChoice) {
        const ask = this.button("Ask about this PDF", () => {
          if (!this.connected || this.busy || generation !== this.generation || !this.offerId) return;
          const prepared = this.preparedChoices.get(row.result_ref);
          if (prepared) { (this.onAskPrepared || this.onAsk)(prepared); return; }
          this.onAskChoice!({offer_id: this.offerId, result_ref: row.result_ref}, {
            expiresAt: 0, filename: row.attachment_filename || row.title || "", fileType: row.file_type ?? null});
        });
        this.choiceButtons.push(ask); actions.append(ask);
      }
      this.choiceActions.push(actions); card.append(actions); section.append(card);
      this.actionsByRef.set(row.result_ref, Array.from(actions.querySelectorAll("button")));
    }
    this.container.append(section);
    this.update();
  }

  invalidate(message = EXPIRED): void { this.retire(message); }

  complete(file: FileCompletion): void {
    if (!["downloaded", "prepared", "previewed"].includes(file.status)) return;
    const pending = this.pendingChoice;
    this.pendingChoice = null;
    if (file.status === "prepared" && file.action === "prepare" && pending?.action === "prepare" &&
        pending.generation === this.generation && this.offerId &&
        typeof file.document_id === "string" && /^[a-f0-9]{64}$/.test(file.document_id)) {
      // Bind to the exact clicked ref, never a filename (duplicates are common).
      this.preparedChoices.set(pending.ref, file.document_id);
    }
    const card = this.card(file.title, file.file_type);
    card.className += " file-card-complete";
    const status = document.createElement("p");
    status.textContent = file.status === "prepared" ? "Prepared locally" : file.status === "downloaded" ? "Downloaded" : "Preview opened";
    card.append(status);
    if (pending && pending.generation === this.generation && this.offerId && file.status !== "prepared") {
      // Completion is a receipt, not new authority. Reuse only the exact clicked
      // card's callbacks; retirement/Clear disables these along with originals.
      const followups = document.createElement("div"); followups.className = "file-card-actions";
      for (const original of this.actionsByRef.get(pending.ref) ?? []) {
        const button = this.button(original.textContent || "", () => original.click());
        this.choiceButtons.push(button); followups.append(button);
      }
      this.choiceActions.push(followups); card.append(followups);
    }
    const actions = document.createElement("div"); actions.className = "file-card-actions";
    const add = (label: string, action: () => void): void => {
      const button = this.button(label, () => { if (!this.busy && this.connected) action(); });
      this.outcomeButtons.push(button); actions.append(button);
    };
    if (file.status === "prepared" && typeof file.document_id === "string" && /^[a-f0-9]{64}$/.test(file.document_id)) {
      const id = file.document_id;
      add(this.onAskPrepared ? "Ask about this PDF" : "Ask about this file", () => (this.onAskPrepared || this.onAsk)(id));
    }
    if (typeof file.download_id === "number" && Number.isSafeInteger(file.download_id) && file.download_id >= 0) {
      const id = file.download_id; add("Show download", () => this.onShowDownload(id));
    }
    card.append(actions); this.outcomes.push(card); this.container.append(card); this.update();
  }

  clear(): void {
    this.removeChoices();
    for (const button of this.outcomeButtons) button.disabled = true;
    for (const card of this.outcomes) card.remove();
    this.outcomes.length = 0; this.outcomeButtons = []; this.busy = false;
  }
  dispose(): void { this.clear(); this.connected = false; }

  private update(): void {
    for (const button of this.choiceButtons) button.disabled = this.busy || !this.connected || !this.offerId;
    for (const button of this.outcomeButtons) button.disabled = this.busy || !this.connected;
    if (this.refreshButton) this.refreshButton.disabled = this.busy || !this.connected;
  }
  private retire(message: string): void {
    this.pendingChoice = null; this.preparedChoices.clear();
    this.offerId = null; this.generation += 1;
    for (const button of this.choiceButtons) button.disabled = true;
    for (const actions of this.choiceActions) actions.remove();
    this.choiceButtons = []; this.choiceActions = []; this.actionsByRef.clear();
    if (this.choices) {
      const status = document.createElement("p"); status.className = "file-card-state"; status.textContent = message;
      // Only one retirement notice, even when Stop follows a consumed operation.
      for (const node of this.choices.querySelectorAll(".file-card-state, .file-card-refresh")) node.remove();
      this.refreshButton = this.button("Refresh results", () => this.refresh());
      this.refreshButton.className = "file-card-refresh";
      this.choices.append(status, this.refreshButton);
    }
    this.update();
  }
  private refresh(): void {
    if (!this.connected || this.busy) return;
    this.busy = true; this.invalidate(); this.onRefresh();
  }
  private removeChoices(): void {
    this.pendingChoice = null; this.preparedChoices.clear();
    this.offerId = null; this.generation += 1;
    this.choices?.remove(); this.choices = null;
    if (this.refreshButton) this.refreshButton.disabled = true;
    for (const button of this.choiceButtons) button.disabled = true;
    this.choiceActions = []; this.choiceButtons = []; this.actionsByRef.clear(); this.refreshButton = null;
  }
  private card(title: string, fileType: string | null): HTMLElement {
    const card = document.createElement("article"); card.className = "file-card";
    const heading = document.createElement("div"); heading.className = "file-card-heading";
    const badge = document.createElement("span"); badge.className = "file-card-type";
    badge.textContent = text(fileType, "FILE").slice(0, 12).toUpperCase();
    const name = document.createElement("strong"); name.textContent = text(title, "Attachment");
    heading.append(badge, name); card.append(heading); return card;
  }
  private button(label: string, click: () => void): HTMLButtonElement {
    const button = document.createElement("button"); button.type = "button";
    button.textContent = label; button.onclick = click; return button;
  }
}
