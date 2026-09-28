/** Explicit user selection; never infer download or remote-context consent from model text. */
export type DocumentOperation = "document_prepare" | "document_question" | "document_refresh";
type Start = (kind: DocumentOperation, payload: Record<string, unknown>) => void;
interface Choice { result_ref: string; title: string | null; attachment_filename?: string | null; }
export const EXPIRED_RESULTS = "Results expired; refresh results";
export function documentError(code: string): string {
  const messages: Record<string, string> = {
    stale_snapshot: "The page or selected file state changed. Refresh results and select the file again.",
    stale_result_ref: "Results changed or were cleared; refresh results",
    attachment_preview_only: "Only a preview link with unsupported or incomplete attachment identity is available. Refresh cannot repair missing metadata.",
    preview_selection_limit: "Select one preview-linked PDF alone, or up to two direct PDFs. Mixed or multiple-preview selections are not supported.",
    preview_control_missing: "The selected preview has no identifiable Download control. No approved download was started.",
    preview_ambiguous: "The preview's attachment or Download control is ambiguous. Nothing was prepared.",
    preview_identity_mismatch: "The preview attachment or download target changed or does not match. Refresh and select again for fresh permission.",
    preview_unexpected_download: "Preview triggered an unexpected download. It was not approved or ingested; cancellation was requested where attributable.",
    attachment_unavailable: "No unambiguous authorized preview-PDF or direct-PDF candidate is available. Refresh cannot create missing attachment identity or links.",
    unsupported_attachment_type: "Unsupported document type; only PDF attachments can be prepared.",
    read_only_blocked: "This legacy request was blocked. Reload the extension and select the document again.",
    no_current_results: "No eligible current search results on this page.",
    refresh_unavailable: "Refresh was rejected. Check that the backend and extension match, then explicitly refresh again.",
    operation_interrupted: "Document operation stopped. Refresh results before selecting again.",
    research_scope_changed: "Research stopped because its tab, document or authorization changed.",
    research_no_relevant_documents: "No unambiguous relevant PDF in the current bounded results.",
    research_search_incomplete: "Research could not obtain current authorized PDF metadata.",
    invalid_pdf: "The downloaded file is not a valid PDF. Try downloading the attachment again.",
    pdf_encrypted: "This PDF requires a password. Prepare a copy that opens without a password and permits text extraction.",
    pdf_extraction_not_allowed: "This PDF does not permit text extraction. Ask the document owner for an extractable copy.",
    pdf_crypto_unavailable: "PDF encryption support is missing from the backend. Install the updated requirements, restart the backend, and try again.",
    pdf_parse_failed: "The PDF parser could not read this file. Preparation stopped before model analysis.",
    pdf_too_large: "This PDF exceeds the local file-size limit. Choose a smaller PDF.",
    pdf_limit_exceeded: "This PDF exceeds a local preparation limit for pages or extracted content. Choose a smaller document or page range.",
    pdf_not_found: "The downloaded PDF is no longer available locally. Download and prepare it again.",
    pdf_read_failed: "The backend could not read the local PDF. Check that the download is complete and accessible, then retry.",
    ingestion_failed: "Local PDF preparation failed. Model analysis did not start.",
    index_failed: "Local PDF indexing failed. Model analysis did not start.",
    index_incompatible: "The local PDF index is incompatible with the configured embedding model.",
    router_timeout: "The local router timed out while waiting for Ollama. The model may still be loading or busy. Try again shortly.",
    router_unavailable: "The local router could not run. Check that Ollama is running and the configured router model is installed, then try again.",
    router_invalid_response: "The local router returned an invalid classification. Try your request again.",
    document_retry_unavailable: "There is no previous question for this PDF. Enter the question to retry.",
    document_not_ready: "This PDF is not prepared for questions yet. Choose Prepare locally or Ask about this PDF on its file card.",
    rag_remote_context_denied: "Prepared locally. To answer with the configured remote model, explicitly allow this question to send selected PDF text and page images.",
    rag_invalid_request: "The document question is invalid. Enter a question and select the PDF again.",
    rag_context_budget: "The selected evidence exceeds the answer limit. Ask a narrower question about this PDF.",
    rag_invalid_answer: "The model returned an answer in an invalid format. The PDF is still prepared; try the question again.",
    rag_unknown_source: "The model cited a source outside the selected evidence. The answer was rejected; try the question again.",
    rag_invalid_evidence: "The model's citation did not match the supplied PDF evidence. The answer was rejected; the PDF is still prepared. Try the question again.",
    rag_unexpected_tool_call: "The model requested an action instead of answering from the PDF. The answer was rejected.",
    rag_provider_not_allowed: "The configured model cannot answer document questions. Check the backend provider configuration.",
    rag_configuration_error: "The document answer provider configuration is invalid. Check the backend configuration.",
    rag_provider_failed: "The model could not complete the document answer. Check the model service and try again.",
    rag_provider_timeout: "The model exceeded the document answer time limit. The PDF is still prepared; try the question again.",
    rag_retrieval_failed: "The selected PDF evidence could not be retrieved. Prepare the PDF again before asking.",
    rag_visual_unavailable: "The selected PDF page image is unavailable locally. Prepare the PDF again before asking about its figures.",
    rag_invalid_page: "The requested page is invalid or ambiguous for the selected PDF.",
    rag_render_failed: "The selected PDF page could not be rendered safely.",
  };
  return messages[code] ?? "Document operation stopped";
}
const required = <T extends HTMLElement>(id: string): T => document.getElementById(id) as T;

/** Shared chat progress labels; this does not create the retired legacy panel. */
export function documentProgress(stage: string): string {
  const labels: Record<string, string> = {
    preview_confirmation: "Preview confirmation required", confirmation_required: "Download confirmation required",
    searching_reports: "Searching website documents...", preparing: "Preparing selected PDF",
    preview_opening: "Opening preview...", waiting_preview: "Waiting for PDF viewer...",
    locating_download: "Locating Download control...", acquiring: "Downloading...",
    validating: "Validating PDF", ingesting: "Parsing PDF...", indexing: "Indexing PDF...",
    searching_document: "Searching inside reports...", retrieving: "Searching document...",
    relevant_pages: "Relevant pages found", rendering: "Rendering selected page...",
    analyzing_visual: "Analyzing selected figure...", answering: "Preparing grounded answer...",
    report_unavailable: "One selected report is unavailable",
    text_retrieval: "Text retrieval", visual_retrieval: "Visual retrieval",
    multimodal_fusion: "Combined evidence", rag_generation: "Grounded answer",
  };
  return labels[stage] || "Preparing selected documents";
}

/** Legacy protocol test/client component; not instantiated or shipped by the chat UI. */
export class DocumentPanel {
  private readonly root = required<HTMLDetailsElement>("documents");
  private readonly choices = required<HTMLDivElement>("document-choices");
  private readonly ready = required<HTMLDivElement>("document-ready");
  private readonly question = required<HTMLTextAreaElement>("document-question");
  private readonly remote = required<HTMLInputElement>("document-remote");
  private readonly prepare = required<HTMLButtonElement>("document-prepare");
  private readonly ask = required<HTMLButtonElement>("document-ask");
  private readonly refresh = required<HTMLButtonElement>("document-refresh");
  private readonly status = required<HTMLElement>("document-status");
  private offerId: string | null = null;
  private busy = false;
  private connected = false;
  private expiresAt = 0;
  private expiryTimer: number | null = null;
  constructor(start: Start) {
    this.prepare.onclick = () => {
      this.checkExpiry();
      const refs = this.selected(this.choices);
      if (this.busy || !this.offerId || refs.length < 1 || refs.length > 2) return;
      this.status.textContent = `${refs.length} document(s) explicitly selected.`;
      start("document_prepare", { offer_id: this.offerId, result_refs: refs,
        question: this.question.value.trim() || null, allow_remote_context: this.remote.checked });
    };
    this.refresh.onclick = () => {
      if (this.busy || !this.connected) return;
      this.expireChoices();
      start("document_refresh", {});
    };
    this.ask.onclick = () => {
      const ids = this.selected(this.ready), question = this.question.value.trim();
      if (this.busy || !question || ids.length < 1 || ids.length > 2) return;
      start("document_question", { document_ids: ids, question, allow_remote_context: this.remote.checked });
    };
    this.root.addEventListener("change", () => this.update());
    this.question.addEventListener("input", () => this.update());
    this.update();
  }
  private selected(parent: HTMLElement): string[] {
    return [...parent.querySelectorAll<HTMLInputElement>("input:checked")].map(input => input.value);
  }
  private update(): void {
    this.checkExpiry();
    const selected = this.selected(this.choices).length, prepared = this.selected(this.ready).length;
    this.prepare.disabled = this.busy || !this.offerId || selected < 1 || selected > 2;
    this.ask.disabled = this.busy || !this.question.value.trim() || prepared < 1 || prepared > 2;
    this.refresh.disabled = this.busy || !this.connected;
    for (const parent of [this.choices, this.ready]) {
      const full = this.selected(parent).length >= 2;
      for (const input of parent.querySelectorAll<HTMLInputElement>("input")) input.disabled = this.busy || (full && !input.checked);
    }
    const available = (!!this.offerId && this.choices.childElementCount > 0) || this.ready.childElementCount > 0;
    this.question.disabled = this.busy || !available; this.remote.disabled = this.busy || !available;
  }
  setBusy(value: boolean): void {
    this.busy = value;
    if (!value) this.remote.checked = false;
    this.update();
  }
  setConnected(value: boolean): void { this.connected = value; this.update(); }
  clearChoices(): void {
    this.offerId = null; this.expiresAt = 0;
    if (this.expiryTimer !== null) window.clearTimeout(this.expiryTimer);
    this.expiryTimer = null;
    this.choices.replaceChildren(); this.remote.checked = false; this.update();
  }
  expireChoices(): void {
    this.clearChoices(); this.root.hidden = false;
    this.status.textContent = EXPIRED_RESULTS;
  }
  private checkExpiry(): void {
    if (this.offerId && this.expiresAt && Date.now() >= this.expiresAt) this.expireChoices();
  }
  reset(): void {
    this.clearChoices(); this.ready.replaceChildren();
    this.question.value = ""; this.remote.checked = false;
    this.status.textContent = ""; this.busy = false;
    this.root.open = false; this.root.hidden = true;
    this.update();
  }
  showChoices(payload: { offer_id: string | null; results: Choice[]; state?: string;
      expires_at_ms?: number; expires_in_ms?: number; refresh_id?: string }): void {
    this.clearChoices(); this.offerId = payload.offer_id;
    if (!payload.offer_id) { if (payload.state) this.expireChoices(); return; }
    this.root.hidden = false; this.root.open = true;
    if (!payload.results.length) {
      this.offerId = null; this.status.textContent = documentError("no_current_results"); return;
    }
    this.expiresAt = Math.min(payload.expires_at_ms ?? Infinity,
      Date.now() + Math.min(payload.expires_in_ms ?? 300000, 300000));
    this.status.textContent = "Select one preview PDF alone, or up to two direct PDFs. Preview may load/render the attachment in Chrome and requires its own Allow once. Downloads require separate Allow once.";
    for (const row of payload.results.slice(0, 10)) this.row(this.choices, row.result_ref, (row.title || row.attachment_filename || "Untitled result").slice(0, 512));
    this.expiryTimer = window.setTimeout(() => this.checkExpiry(), Math.max(0, this.expiresAt - Date.now()));
    this.update();
  }
  showReady(payload: { documents: Array<{ document_id: string; title?: string }> }): void {
    this.ready.replaceChildren(); this.root.hidden = false;
    for (const row of payload.documents.slice(0, 10)) this.row(this.ready, row.document_id,
      row.title && !/^(?:[a-z]:[\\/]|\\\\|file:|\/)/i.test(row.title) ? row.title.slice(0, 200) + " - ready for questions" : "Prepared PDF - ready for questions");
    this.update();
  }
  openDocument(documentId: string): void {
    if (this.busy || !/^[a-f0-9]{64}$/.test(documentId)) return;
    const rows = [...this.ready.querySelectorAll<HTMLInputElement>("input")];
    if (!rows.some(row => row.value === documentId)) return;
    for (const row of rows) row.checked = row.value === documentId;
    this.root.hidden = false; this.root.open = true; this.remote.checked = false;
    this.update(); this.question.focus();
  }
  private row(parent: HTMLElement, value: string, text: string): void {
    const label = document.createElement("label"), input = document.createElement("input");
    input.type = "checkbox"; input.value = value;
    label.append(input, document.createTextNode(" " + text)); parent.append(label);
  }
  progress(stage: string): string {
    const label = documentProgress(stage);
    this.status.textContent = label; return label;
  }
  failed(code: string): void { this.clearChoices(); this.complete(documentError(code)); }
  complete(status: string): void { this.status.textContent = status; this.question.value = ""; this.remote.checked = false; }
}
