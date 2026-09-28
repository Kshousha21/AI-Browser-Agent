import type {
  AssistantDeltaPayload,
  AssistantDonePayload,
  ConfirmationRequestPayload,
  ControlStatePayload,
  Envelope,
  ErrorPayload,
  HelloAckPayload,
  RunStatusPayload,
  TimelineStepPayload,
  ToolRequestPayload,
} from "@shared/protocol";
import { makeEnvelope, parseEnvelope, PROTOCOL_VERSION, PROTOCOL_REVISION, compatibleHello, PROTOCOL_MISMATCH } from "@shared/protocol";

import { isAllowed } from "../lib/allowlist";
import {
  buildSocketUrl,
  loadSettings,
  onSettingsChanged,
  type ExtensionSettings,
} from "../lib/settings";
import {
  activeTab,
  browserSession,
  executeBrowserTool,
  extractPageState,
} from "./browser-tools";
import { ConfirmationController } from "./confirmation";
import { documentError, documentProgress } from "./documents";
import { FileCards, type FileAction, type FileClickAuthority, type FileCompletion, type FileChoices } from "./file-cards";
import { currentResearchRows, observedResearchRows, groundedText, pdfLabel, type GroundedAnswer, type ResearchRow } from "./research-view";
import { resolvePanelOwner, supportedPanelTab } from "../lib/panel-scope";

type ConnectionState = "disconnected" | "connecting" | "connected" | "error";

function required<T extends Element>(id: string): T {
  const element = document.getElementById(id);
  if (!element) throw new Error(`missing element #${id}`);
  return element as unknown as T;
}

const ui = {
  title: required<HTMLElement>("page-title"),
  url: required<HTMLElement>("page-url"),
  statusDot: required<HTMLElement>("status-dot"),
  statusText: required<HTMLElement>("status-text"),
  runStatus: required<HTMLElement>("run-status"),
  tokenUsage: required<HTMLElement>("token-usage"),
  elapsed: required<HTMLElement>("elapsed"),
  timeline: required<HTMLElement>("timeline"),
  conversation: required<HTMLElement>("conversation"),
  emptyState: required<HTMLElement>("empty-state"),
  composer: required<HTMLFormElement>("composer"),
  input: required<HTMLTextAreaElement>("input"),
  send: required<HTMLButtonElement>("send"),
  stop: required<HTMLButtonElement>("stop"),
  pause: required<HTMLButtonElement>("pause"),
  resume: required<HTMLButtonElement>("resume"),
  confirmationPause: required<HTMLButtonElement>("confirmation-pause"),
  confirmationResume: required<HTMLButtonElement>("confirmation-resume"),
  clear: required<HTMLButtonElement>("clear"),
  optionsButton: required<HTMLButtonElement>("options-button"),
  researchMode: required<HTMLInputElement>("research-mode"),
  researchRemote: required<HTMLInputElement>("research-remote"),
  researchSelection: required<HTMLElement>("research-selection"),
  researchSelectionClear: required<HTMLButtonElement>("research-selection-clear"),
};
const confirmation = new ConfirmationController();
const ownedDownloads = new Set<number>();
const files = new FileCards(ui.conversation,
  (action, selection, authority) => { void startFileAction(action, selection, authority); },
  () => { void refreshDocumentChoices(); },
  id => selectPreparedPdf(id),
  id => {
    if (!awaitingTurn && !ended && handshakeComplete && ownedDownloads.has(id)) {
      const epoch = sessionEpoch;
      void chrome.downloads.search({id}).then(([item]) => {
        if (epoch !== sessionEpoch || ended || !ownedDownloads.has(id)) return;
        if (!item?.exists) throw new Error("unavailable");
        chrome.downloads.show(id);
      }).catch(() => { if (epoch === sessionEpoch) addMessage("error", "This browser download is no longer available."); });
    }
  },
  (selection, authority) => {
    if (awaitingTurn || ended || !handshakeComplete) return;
    specificPrepared = null;
    specificPdf = {...selection, ...authority};
    ui.researchMode.checked = false; ui.researchRemote.checked = false;
    ui.researchSelection.textContent = "Ask only about " + pdfLabel(authority.filename) + ". Send a question to prepare and research this PDF.";
    updateResearchSelectionControls();
    ui.input.focus();
  },
  id => selectPreparedPdf(id));

let settings: ExtensionSettings | null = null;
let socket: WebSocket | null = null;
let connectionState: ConnectionState = "disconnected";
let handshakeComplete = false;
let ready: Promise<void> | null = null;
let listeners: AbortController | null = null;
let awaitingTurn = false;
let streamingBubble: HTMLElement | null = null;
let activeRunId: string | null = null;
let timer: number | null = null;
let runStarted = 0;
let ownerReady = false;
let ended = false;
// Clear invalidates pending UI callbacks without changing the immutable tab owner.
let sessionEpoch = 0;
let refreshId: string | null = null;
let refreshVersion = -1;
let refreshCancelled = false;
let documentPreflight = false;
let fileActionId: string | null = null;
let fileActionVersion = -1;
let fileActionCancelled = false;
let researchId: string | null = null;
let researchVersion = -1;
let researchCancelled = false;
let researchMaximum = 3;
let pendingQueryId: string | null = null;
let pendingQueryVersion = -1;
let activeDocumentContext = false;
let currentFileChoices: FileChoices | null = null;
let specificPdf: ({offer_id: string; result_ref: string} & FileClickAuthority) | null = null;
let specificPrepared: string | null = null;
const preparedTitles = new Map<string, string>();

function selectPreparedPdf(id: string): void {
  if (awaitingTurn || ended || !handshakeComplete || !preparedTitles.has(id)) return;
  specificPdf = null; specificPrepared = id;
  ui.researchMode.checked = false; ui.researchRemote.checked = false;
  ui.researchSelection.textContent = "Ask only about " + pdfLabel(preparedTitles.get(id)) + ".";
  updateResearchSelectionControls();
  ui.input.focus();
}

function invalidateResearch(clearChoices = true): void {
  researchCancelled = true; specificPdf = null; specificPrepared = null;
  if (clearChoices) { currentFileChoices = null; activeDocumentContext = false; pendingQueryId = null; }
  ui.researchSelection.textContent = activeDocumentContext ? "Follow-up questions use the current prepared PDF selection." : "";
  ui.researchRemote.checked = false;
  browserSession.endResearch();
  updateResearchSelectionControls();
}

function updateResearchSelectionControls(): void {
  const selected = specificPdf !== null || specificPrepared !== null;
  // Ask is already an explicit single-PDF scope. A generic checkbox must not
  // silently replace it with automatic, possibly provider-backed discovery.
  ui.researchMode.disabled = awaitingTurn || selected;
  ui.researchSelectionClear.hidden = !selected && !activeDocumentContext;
  ui.researchSelectionClear.disabled = awaitingTurn;
}

function setConnection(state: ConnectionState, text: string): void {
  files.setConnected(state === "connected");
  connectionState = state;
  ui.statusDot.dataset["state"] = state;
  ui.statusText.textContent = text;
}

function setRunStatus(text: string): void {
  ui.runStatus.textContent = text;
}

function addMessage(kind: "user" | "assistant" | "error", text: string): HTMLElement {
  ui.emptyState.hidden = true;
  const bubble = document.createElement("div");
  bubble.className = `message ${kind}`;
  bubble.textContent = text;
  ui.conversation.append(bubble);
  ui.conversation.scrollTop = ui.conversation.scrollHeight;
  return bubble;
}

function setBusy(busy: boolean): void {
  awaitingTurn = busy;
  ui.send.disabled = busy || ended || !ownerReady;
  ui.stop.disabled = !busy;
  ui.clear.disabled = false;
  if (!busy) { browserSession.stop(); confirmation.cancel(); }
  updateControlButtons();
  updateResearchSelectionControls();
  ui.researchRemote.disabled = busy;
  if (!busy) ui.researchRemote.checked = false;
  files.setBusy(busy);
  if (busy) {
    runStarted = performance.now();
    if (timer !== null) clearInterval(timer);
    timer = window.setInterval(() => {
      ui.elapsed.textContent = `${((performance.now() - runStarted) / 1000).toFixed(1)}s`;
    }, 100);
  } else if (timer !== null) {
    clearInterval(timer);
    timer = null;
  }
}

async function refreshHeader(): Promise<void> {
  const tab = await browserSession.ownerTab();
  const allowed = tab?.url && settings && isAllowed(tab.url);
  ui.title.textContent = allowed ? (tab.title || "web page") : "Outside supported scope";
  ui.url.textContent = allowed ? new URL(tab.url!).origin + new URL(tab.url!).pathname : " ";
}

function closeSocket(): void {
  invalidateResearch(); researchId = null; preparedTitles.clear();
  refreshId = null; refreshCancelled = false;
  documentPreflight = false;
  files.clear(); ownedDownloads.clear(); fileActionId = null; fileActionCancelled = false;
  browserSession.stop();
  confirmation.cancel();
  listeners?.abort();
  listeners = null;
  socket?.close();
  socket = null;
  handshakeComplete = false;
  ready = null;
}

function connect(current: ExtensionSettings): Promise<void> {
  closeSocket();
  if (!current.token) {
    setConnection("error", "No token set - open Settings");
    return Promise.reject(new Error("no token configured"));
  }
  setConnection("connecting", `Connecting to ${current.backendUrl}...`);
  const controller = new AbortController();
  listeners = controller;
  const { signal } = controller;

  ready = new Promise<void>((resolve, reject) => {
    signal.addEventListener("abort", () => reject(new Error("Connection reset")), { once: true });
    let ws: WebSocket;
    let protocolMismatch = false;
    try {
      ws = new WebSocket(buildSocketUrl(current));
    } catch (error) {
      setConnection("error", "Bad backend URL");
      reject(error);
      return;
    }
    socket = ws;
    ws.addEventListener("open", () => {
      ws.send(JSON.stringify(makeEnvelope("auth", { token: current.token })));
      ws.send(JSON.stringify(makeEnvelope("hello", {
        client: "sidepanel/0.2.0",
        protocol_version: PROTOCOL_VERSION,
        protocol_revision: PROTOCOL_REVISION,
      })));
    }, { signal });
    ws.addEventListener("message", (event: MessageEvent<string>) => {
      if (ws !== socket) return;
      try {
        const envelope = parseEnvelope(event.data);
        if (envelope.type === "hello_ack" && !handshakeComplete) {
          const payload = envelope.payload as HelloAckPayload;
          if (!compatibleHello(payload)) {
            protocolMismatch = true;
            setConnection("error", PROTOCOL_MISMATCH);
            setRunStatus("Protocol mismatch");
            reject(new Error(PROTOCOL_MISMATCH));
            ws.close(1002, PROTOCOL_MISMATCH);
            return;
          }
          handshakeComplete = true;
          researchMaximum = Number.isSafeInteger(payload.research_max_pdfs) &&
            payload.research_max_pdfs! >= 1 && payload.research_max_pdfs! <= 5 ? payload.research_max_pdfs! : 3;
          setConnection("connected", `Backend connected - ${payload.provider}; protocol compatible`);
          resolve();
        } else {
          void handleFrame(envelope, ws);
        }
      } catch (error) {
        console.error("[ai-browser-agent] frame error", error);
      }
    }, { signal });
    ws.addEventListener("error", () => {
      setConnection("error", "Connection failed - is the backend running?");
    }, { signal });
    ws.addEventListener("close", (event) => {
      invalidateResearch(); researchId = null; preparedTitles.clear();
      sessionEpoch++; refreshId = null; refreshCancelled = false; documentPreflight = false;
      files.clear(); ownedDownloads.clear(); fileActionId = null; fileActionCancelled = true;
      const connected = handshakeComplete;
      handshakeComplete = false;
      socket = null;
      ready = null;
      const message =
        protocolMismatch || event.reason === PROTOCOL_MISMATCH ? PROTOCOL_MISMATCH :
        event.code === 1008
          ? `Rejected by backend: ${event.reason || "check the token"}`
          : connected ? "Backend offline" : "Could not connect";
      setConnection(connected ? "disconnected" : "error", message);
      if (awaitingTurn) {
        finishStream();
        addMessage("error", message);
        setBusy(false);
      }
      reject(new Error(message));
    }, { signal });
  });
  ready.catch(() => undefined);
  return ready;
}

async function handleFrame(envelope: Envelope, source: WebSocket): Promise<void> {
  if (source !== socket || !handshakeComplete) return;
  if (envelope.type === "routing_status") {
    if (!awaitingTurn || envelope.corr_id !== pendingQueryId || browserSession.version !== pendingQueryVersion || browserSession.mode !== "running") return;
    const route = envelope.payload as {route: string; adapter: string; router_model: string; router_latency_ms: number; prepared_followup: boolean; document_count: number; route_source?: string};
    if (!["READ","SEARCH","ACTION","NAVIGATION","DOCUMENT","IMAGE","GENERAL","UNSUPPORTED"].includes(route.route) ||
        !["GenericSiteAdapter"].includes(route.adapter)) return;
    if (route.route === "DOCUMENT" && route.prepared_followup === true) {
      if (!Number.isSafeInteger(route.document_count) || route.document_count < 1 || route.document_count > researchMaximum) return;
      // This correlates prepared-PDF answer frames only. It grants NO acquisition.
      researchId = pendingQueryId; researchVersion = browserSession.version; researchCancelled = false;
    } else if (route.route !== "DOCUMENT") {
      researchId = null; invalidateResearch(false); activeDocumentContext = false;
    }
    activeDocumentContext = route.route === "DOCUMENT" && (route.prepared_followup === true || researchId !== null);
    const entry = document.createElement("li");
    entry.textContent = (route.route_source === "main_request" ? "Route changed: " : "Router: ") + route.route + " ? " + route.adapter;
    if (typeof route.router_model === "string" && /^[a-zA-Z0-9:._/-]{1,100}$/.test(route.router_model)) entry.textContent += " ? " + route.router_model;
    ui.timeline.append(entry); updateResearchSelectionControls();
    return;
  }
  const fileFrame = envelope.payload as { user_action_id?: string; research_id?: string };
  const researchFrame = !!fileFrame.research_id || (!!researchId && fileFrame.user_action_id === researchId);
  if (researchFrame && ((fileFrame.research_id || fileFrame.user_action_id) !== researchId || researchCancelled ||
      browserSession.version !== researchVersion || browserSession.mode !== "running")) {
    if (envelope.type === "document_done" && fileFrame.research_id === researchId && researchId) {
      researchId = null; activeRunId = null; setBusy(false); setRunStatus("Stopped");
    }
    return;
  }
  if (fileFrame.user_action_id && fileFrame.user_action_id !== fileActionId && !researchFrame) return;
  const fileInterrupted = fileActionCancelled || (fileActionId !== null &&
    (browserSession.version !== fileActionVersion || browserSession.mode !== "running"));
  if (fileFrame.user_action_id && !researchFrame && fileInterrupted && envelope.type !== "document_done") return;
  if (envelope.type === "research_plan" || envelope.type === "research_file") {
    if (!researchFrame || !researchId) return;
    try {
      if (envelope.type === "research_plan") {
        const plan = envelope.payload as { source_url: string; files: ResearchRow[] };
        const matchRows = browserSession.researchGrant?.hasSpecificAdmission ? observedResearchRows : currentResearchRows;
        if (!Array.isArray(plan.files) || !matchRows(plan.files, currentFileChoices, researchMaximum)) throw new Error("invalid research plan");
        browserSession.approveResearchPlan(researchId, plan.source_url, plan.files.map(file => ({
          sourceKey: file.source_key, resultRef: file.result_ref, filename: file.filename, fileType: file.file_type})), currentFileChoices!.expires_at_ms!);
        addMessage("assistant", "Researching " + plan.files.length + " relevant PDF(s):\n" + plan.files.map(file => pdfLabel(file.filename)).join("\n"));
      } else {
        const file = envelope.payload as ResearchRow & { expires_at_ms: number };
        if (!observedResearchRows([file], currentFileChoices, researchMaximum)) throw new Error("invalid research file");
        browserSession.activateResearchFile(researchId, file.source_key,
          {resultRef: file.result_ref, filename: file.filename, fileType: file.file_type});
      }
    } catch {
      invalidateResearch(); files.invalidate(); setRunStatus("Research authority changed");
      source.send(JSON.stringify(makeEnvelope("stop", {run_id: activeRunId})));
      addMessage("error", "Research stopped: selected PDF authority changed. Refresh results and ask again.");
      researchId = null; setBusy(false);
    }
    return;
  }
  if (envelope.type === "document_choices") {
    const payload = envelope.payload as { refresh_id?: string };
    if (payload.refresh_id && (payload.refresh_id !== refreshId || refreshCancelled)) return;
    if (payload.refresh_id && (browserSession.version !== refreshVersion || browserSession.mode !== "running")) {
      refreshCancelled = true; files.invalidate(); return;
    }
    const choices = envelope.payload as Parameters<FileCards["showChoices"]>[0];
    currentFileChoices = choices.offer_id ? choices : null;
    if (specificPdf && specificPdf.offer_id !== choices.offer_id) {
      specificPdf = null; ui.researchSelection.textContent = "";
      updateResearchSelectionControls();
    }
    if (choices.offer_id) { files.showChoices(choices); ui.emptyState.hidden = true; }
    else files.invalidate();
    return;
  }
  if (envelope.type === "document_ready") {
    const payload = envelope.payload as {documents: Array<{document_id: string; title?: string}>};
    for (const item of payload.documents) if (/^[a-f0-9]{64}$/.test(item.document_id) && item.title) preparedTitles.set(item.document_id, pdfLabel(item.title));
    return;
  }
  if (envelope.type === "document_progress") {
    const payload = envelope.payload as { stage: string; run_id: string; current?: number; total?: number; page?: number; pages?: number[]; status?: string; results?: number; generator_received_images?: boolean };
    activeRunId = payload.run_id; browserSession.runId = payload.run_id;
    let label = documentProgress(payload.stage);
    if (Number.isSafeInteger(payload.current) && Number.isSafeInteger(payload.total)) label += " " + payload.current + "/" + payload.total;
    if (Number.isSafeInteger(payload.page)) label += " " + payload.page;
    if (Array.isArray(payload.pages)) label += ": " + payload.pages.filter(page => Number.isSafeInteger(page) && page > 0).slice(0, 5).join(", ");
    if (["text_retrieval", "visual_retrieval", "multimodal_fusion", "rag_generation"].includes(payload.stage)) {
      const entry = document.createElement("li");
      entry.textContent = documentProgress(payload.stage) + (payload.status === "ok" ? ": OK" : ": unavailable");
      if (Number.isSafeInteger(payload.results) && payload.results! >= 0 && payload.results! <= 20) entry.textContent += " ? " + payload.results + " results";
      if (typeof payload.generator_received_images === "boolean") entry.textContent += payload.generator_received_images ? " ? images supplied" : " ? text supplied";
      ui.timeline.append(entry);
    }
    setRunStatus(label); updateControlButtons(); return;
  }
  if (envelope.type === "document_done") {
    if (researchFrame) {
      const completed = envelope.payload as {status: string; error?: string; prepared?: Array<{document_id: string; title?: string}>;
        unavailable?: number; answer?: GroundedAnswer};
      for (const item of completed.prepared || []) if (/^[a-f0-9]{64}$/.test(item.document_id)) {
        preparedTitles.set(item.document_id, pdfLabel(item.title));
        files.complete({action: "prepare", status: "prepared", title: pdfLabel(item.title), file_type: "pdf", document_id: item.document_id});
      }
      if (completed.answer && ["answered", "insufficient_evidence"].includes(completed.status)) {
        addMessage("assistant", groundedText(completed.answer, preparedTitles));
        if (completed.unavailable) addMessage("assistant", String(completed.unavailable) + " selected report(s) unavailable; only successfully prepared evidence was used.");
        setRunStatus("Finished");
      } else {
        addMessage("error", documentError(completed.error || "document_operation_failed"));
        setRunStatus("Error");
      }
      // End this question's grants/consent, not the still-valid card catalogue.
      // Genuine invalidation already clears choices through its own handler.
      activeDocumentContext = (completed.prepared?.length ?? 0) > 0;
      researchId = null; invalidateResearch(false); activeRunId = null; setBusy(false); return;
    }
    if (fileFrame.user_action_id) {
      const completed = envelope.payload as {status: string; error?: string; file?: Omit<FileCompletion, "status">;
        diagnostic?: {stage?: string; checks?: Record<string, boolean>}};
      if (fileInterrupted || completed.status === "failed" || !completed.file) {
        // An accepted, correlated terminal error is not erased merely because
        // the backend paused the run after rejecting a stale snapshot.
        const code = fileActionCancelled ? "operation_interrupted" : completed.error || "document_operation_failed";
        const stages = ["before_lease", "before_navigation", "guard_target", "guard_regex", "guard_rules", "observer_lease",
          "navigation_dispatch", "navigation_wait", "human_grant", "human_file", "validation", "authority",
          "owner_destination", "html_document", "chrome_document", "owner_after_observation", "changed_href", "preview_wait"];
        const stage = stages.includes(completed.diagnostic?.stage ?? "") ? completed.diagnostic!.stage : null;
        const diagnosticCode = ["operation_interrupted", "stale_snapshot", "stale_result_ref", "preview_identity_mismatch",
          "preview_control_missing", "preview_ambiguous", "preview_unexpected_download", "document_operation_failed"].includes(code)
          ? code : "file_action_failed";
        const changed = ["probe_present", "url_matches", "snapshot_matches", "document_matches", "dom_unchanged"]
          .filter(key => completed.diagnostic?.checks?.[key] === false);
        const message = (code === "operation_interrupted" && !fileInterrupted
          ? "File action stopped before completion. No automatic retry was made."
          : documentError(code)) + (stage ? " [" + diagnosticCode + "; " + stage +
            (changed.length ? "; failed: " + changed.join(", ") : "") + "]" : "");
        if (fileInterrupted || ["stale_result_ref", "stale_snapshot", "preview_identity_mismatch", "attachment_preview_only",
          "unsupported_attachment_type", "read_only_blocked", "attachment_unavailable"].includes(code)) files.invalidate(message);
        addMessage("error", message);
        setRunStatus(fileInterrupted ? "Stopped" : "Error");
      } else {
        const status = {ready: "prepared", downloaded: "downloaded", preview_opened: "previewed"}[completed.status];
        if (status) {
          const id = completed.file.download_id;
          if (typeof id === "number" && Number.isSafeInteger(id) && id >= 0) ownedDownloads.add(id);
          files.complete({...completed.file, status}); ui.emptyState.hidden = true;
          if (completed.file.document_id && /^[a-f0-9]{64}$/.test(completed.file.document_id)) preparedTitles.set(completed.file.document_id, pdfLabel(completed.file.title));
          setRunStatus("Finished");
        } else {
          files.invalidate(); addMessage("error", "Unsupported file-operation response."); setRunStatus("Error");
        }
      }
      fileActionId = null; fileActionCancelled = false; activeRunId = null; setBusy(false); return;
    }
    const refresh = envelope.payload as { refresh_id?: string; error?: string };
    if (refresh.refresh_id) {
      if (refresh.refresh_id !== refreshId) return;
      refreshCancelled ||= browserSession.version !== refreshVersion || browserSession.mode !== "running";
      if (refreshCancelled || refresh.error) {
        const message = documentError(refreshCancelled ? "operation_interrupted" : refresh.error!);
        files.invalidate(message); addMessage("error", message);
      }
      setRunStatus(refreshCancelled ? "Stopped" : refresh.error ? "Error" : "Results refreshed - select documents");
      refreshId = null; refreshCancelled = false; activeRunId = null; setBusy(false); return;
    }
    const payload = envelope.payload as { status: string; error?: string; prepared: unknown[];
      answer?: GroundedAnswer | null };
    if (payload.answer) for (const citation of payload.answer.citations) if (citation.filename) preparedTitles.set(citation.document_id, pdfLabel(citation.filename));
    const text = payload.answer ? groundedText(payload.answer, preparedTitles) :
      payload.status === "failed" ? `${documentError(payload.error || "document_operation_failed")}. Previously prepared documents remain available.` :
      "Selected documents are ready for questions.";
    addMessage(payload.status === "failed" ? "error" : "assistant", text);
    if (payload.status === "failed") files.invalidate(documentError(payload.error || "document_operation_failed"));
    activeRunId = null; setRunStatus(payload.status === "failed" ? "Error" : "Finished"); setBusy(false); return;
  }
  if (envelope.type === "control_state") {
    const state = envelope.payload as ControlStatePayload;
    if (!awaitingTurn || (activeRunId !== null && activeRunId !== state.run_id)) return;
    browserSession.control(state);
    confirmation.setPaused(state.state !== "running");
    setRunStatus(state.reason);
    updateControlButtons();
    return;
  }
  if (envelope.type === "confirmation_request") {
    const payload = envelope.payload as ConfirmationRequestPayload;
    if (!awaitingTurn) return;
    confirmation.setPaused(browserSession.mode !== "running");
    if (browserSession.mode === "running") setRunStatus("Awaiting confirmation");
    const decision = await confirmation.ask(payload);
    if (source === socket && source.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify(makeEnvelope(
        "confirmation_response",
        { decision },
        envelope.id,
      )));
    }
    return;
  }
  if (envelope.type === "tool_request") {
    const request = envelope.payload as ToolRequestPayload;
    activeRunId = request.run_id;
    browserSession.runId = request.run_id;
    const current = settings ?? (await loadSettings());
    if (source !== socket) return;
    const result = await executeBrowserTool(request, current);
    if (source === socket && source.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify(makeEnvelope(
        "tool_result",
        { result },
        envelope.id,
      )));
    }
    return;
  }
  if (envelope.type === "timeline_step") {
    addTimeline(envelope.payload as TimelineStepPayload);
    return;
  }
  if (envelope.type === "run_status") {
    const payload = envelope.payload as RunStatusPayload;
    activeRunId = payload.run_id;
    browserSession.runId = payload.run_id;
    updateControlButtons();
    if (browserSession.mode === "running") setRunStatus(
      payload.step > 0
        ? `Step ${payload.step}/${payload.max_steps} - ${payload.status}`
        : payload.status,
    );
    ui.tokenUsage.textContent = `${payload.usage["total_tokens"] ?? 0} tokens`;
    ui.elapsed.textContent = `${(payload.elapsed_ms / 1000).toFixed(1)}s`;
    return;
  }
  if (envelope.type === "assistant_delta") {
    const payload = envelope.payload as AssistantDeltaPayload;
    if (!streamingBubble) {
      streamingBubble = addMessage("assistant", "");
      streamingBubble.classList.add("pending");
    }
    streamingBubble.textContent += payload.text;
    return;
  }
  if (envelope.type === "assistant_done") {
    if (researchId) return; // Metadata search completion cannot end document work.
    const payload = envelope.payload as AssistantDonePayload;
    if (streamingBubble) streamingBubble.textContent = payload.text;
    else addMessage("assistant", payload.text);
    finishStream();
    activeRunId = null;
    setRunStatus("Finished");
    ui.tokenUsage.textContent = `${payload.usage.total_tokens ?? 0} tokens`;
    ui.elapsed.textContent = `${(payload.elapsed_ms / 1000).toFixed(1)}s`;
    setBusy(false);
    return;
  }
  if (envelope.type === "error") {
    if (researchId) { invalidateResearch(); researchId = null; }
    if (fileActionId !== null) { files.invalidate(); fileActionId = null; fileActionCancelled = true; }
    const payload = envelope.payload as ErrorPayload;
    if (refreshId !== null) {
      refreshId = null; refreshCancelled = true;
      files.invalidate();
    }
    finishStream();
    addMessage("error", `${payload.code}: ${payload.message}`);
    setRunStatus("Error");
    setBusy(false);
  }
}

function addTimeline(payload: TimelineStepPayload): void {
  const item = document.createElement("details");
  item.className = `timeline-step ${payload.result.success ? "ok" : "failed"}`;
  const summary = document.createElement("summary");
  summary.textContent =
    `${payload.step}. ${payload.tool}` +
    (payload.target ? ` - ${payload.target}` : "") +
    ` - ${payload.result.success ? "ok" : payload.result.error}`;
  const body = document.createElement("div");
  body.className = "timeline-body";
  body.textContent =
    `Reason: ${payload.reason}\nResult: ${payload.result.message || "Completed"}\n` +
    `Duration: ${payload.result.duration_ms}ms` +
    (payload.result.reresolved ? "\nTarget was re-resolved after a stale snapshot." : "");
  item.append(summary, body);
  ui.timeline.append(item);
}

function finishStream(): void {
  streamingBubble?.classList.remove("pending");
  streamingBubble = null;
}

async function ensureConnected(current: ExtensionSettings): Promise<void> {
  if (socket?.readyState === WebSocket.OPEN && handshakeComplete) return;
  if (ready && connectionState === "connecting") return ready;
  return connect(current);
}

async function send(): Promise<void> {
  const epoch = sessionEpoch;
  const text = ui.input.value.trim();
  if (!text || awaitingTurn || !ownerReady || ended) return;
  if (specificPdf || specificPrepared || ui.researchMode.checked) {
    return startResearch(text);
  }
  const current = settings ?? (await loadSettings());
  if (epoch !== sessionEpoch || ended) return;
  settings = current;
  const tab = await activeTab();
  if (epoch !== sessionEpoch || ended || awaitingTurn) return;
  if (tab?.id === undefined || !tab.url) return void addMessage("error", "Return to this assistant's owning tab.");
  if (!isAllowed(tab.url)) {
    return void addMessage("error", "Open a normal HTTP/HTTPS website and click the extension.");
  }
  addMessage("user", text);
  files.invalidate();
  ui.input.value = "";
  ui.timeline.replaceChildren();
  ui.tokenUsage.textContent = "0 tokens";
  setRunStatus("Reading page...");
  activeRunId = null;
  browserSession.begin();
  setBusy(true);
  try {
    await ensureConnected(current);
    if (epoch !== sessionEpoch) return;
    if (ended || (await activeTab())?.id !== tab.id) throw new Error("Return to this assistant's owning tab.");
    if (epoch !== sessionEpoch) return;
    browserSession.begin();
    const version = browserSession.version;
    const page = await extractPageState(tab.id, current, true, () => {
      if (ended || epoch !== sessionEpoch || version !== browserSession.version || browserSession.mode !== "running") throw new Error("Page observation was interrupted.");
    });
    if (epoch !== sessionEpoch) return;
    if (!socket || socket.readyState !== WebSocket.OPEN) throw new Error("Backend offline");
    const frame = makeEnvelope("user_message", {
      text, page_state: page,
      allow_remote_context: ui.researchRemote.checked,
    });
    pendingQueryId = frame.id; pendingQueryVersion = browserSession.version;
    socket.send(JSON.stringify(frame));
  } catch (error) {
    if (epoch !== sessionEpoch) return;
    addMessage("error", error instanceof Error ? error.message : String(error));
    setRunStatus("Error");
    setBusy(false);
  }
}

async function refreshDocumentChoices(): Promise<void> {
  const epoch = sessionEpoch;
  if (awaitingTurn || !ownerReady || ended || !handshakeComplete || socket?.readyState !== WebSocket.OPEN) {
    files.setBusy(awaitingTurn); return;
  }
  const source = socket;
  setBusy(true); // Reserve admission synchronously, before any owner lookup.
  documentPreflight = true;
  let owner: chrome.tabs.Tab | null;
  try { owner = await activeTab(); } catch { owner = null; }
  if (epoch !== sessionEpoch || source !== socket || ended) return;
  documentPreflight = false;
  if (!owner || !settings || !supportedPanelTab(owner)) {
    files.invalidate(); setRunStatus("Return to this assistant's owning tab"); setBusy(false); return;
  }
  const target = browserSession.target;
  browserSession.begin(); if (target) browserSession.adopt(target, true);
  activeRunId = null;
  const frame = makeEnvelope("document_refresh", {});
  refreshId = frame.id; refreshCancelled = false;
  refreshVersion = browserSession.version;
  setRunStatus("Refreshing current document choices...");
  source.send(JSON.stringify(frame));
}

async function startFileAction(action: FileAction, selection: {offer_id: string; result_ref: string},
    authority: FileClickAuthority): Promise<void> {
  const epoch = sessionEpoch;
  if (awaitingTurn || !ownerReady || ended || !handshakeComplete || socket?.readyState !== WebSocket.OPEN) {
    files.setBusy(awaitingTurn); return;
  }
  const source = socket;
  setBusy(true); documentPreflight = true;
  let owner: chrome.tabs.Tab | null;
  try { owner = await activeTab(); } catch { owner = null; }
  if (epoch !== sessionEpoch || source !== socket || ended) return;
  documentPreflight = false;
  if (!owner || !settings || !supportedPanelTab(owner)) {
    files.invalidate(); setRunStatus("Return to the owning page and refresh results"); setBusy(false); return;
  }
  const target = browserSession.target;
  browserSession.begin(); if (target) browserSession.adopt(target, true);
  const frame = makeEnvelope("document_file_action", {...selection, action});
  try {
    browserSession.beginUserFileAction(frame.id, action, authority.expiresAt,
      {resultRef: selection.result_ref, filename: authority.filename, fileType: authority.fileType});
    fileActionId = frame.id; fileActionVersion = browserSession.version; fileActionCancelled = false;
    activeRunId = null; setRunStatus(action === "prepare" ? "Preparing selected file locally..." : action === "download" ? "Downloading selected file..." : "Opening selected preview...");
    addMessage("user", action === "prepare" ? "Prepare this file locally." : action === "download" ? "Download this file." : "Open this file's preview.");
    source.send(JSON.stringify(frame));
  } catch {
    fileActionId = null; fileActionCancelled = true; files.invalidate();
    addMessage("error", "The selected file action is no longer available. Refresh results.");
    setRunStatus("Error"); setBusy(false);
  }
}

async function startResearch(question: string): Promise<void> {
  if (awaitingTurn || !ownerReady || ended || question.length > 1000) return;
  const epoch = sessionEpoch, selected = specificPdf, preparedId = specificPrepared;
  const remoteConsent = ui.researchRemote.checked;
  setBusy(true); documentPreflight = true;
  try {
    const current = settings ?? await loadSettings();
    if (epoch !== sessionEpoch || ended) return;
    settings = current;
    const owner = await activeTab();
    if (epoch !== sessionEpoch || ended) return;
    if (!owner?.id || !owner.url || !supportedPanelTab(owner)) throw new Error("owner");
    if (!selected && !preparedId && new URL(owner.url).searchParams.has("preview")) {
      // A preview is not a search-results catalogue. Do not discard valid cards
      // or invoke automatic metadata/provider discovery on this unsupported scope.
      documentPreflight = false; setBusy(false);
      addMessage("error", "Choose Ask about this PDF on a file card to ask about one PDF. For multiple PDFs, return to the search results first.");
      setRunStatus("Choose a PDF or return to search results");
      return;
    }
    if (selected && (currentFileChoices?.offer_id !== selected.offer_id ||
        !currentFileChoices.results.some(row => row.result_ref === selected.result_ref))) throw new Error("selection");
    await ensureConnected(current);
    if (epoch !== sessionEpoch || ended) return;
    const target = browserSession.target;
    browserSession.begin();
    if (target && target.tab_id === owner.id && target.url === owner.url) browserSession.adopt(target, true);
    if (selected && !browserSession.target) throw new Error("selection");
    const version = browserSession.version;
    const page = browserSession.target ?? await extractPageState(owner.id, current, true, () => {
      if (epoch !== sessionEpoch || ended || version !== browserSession.version || browserSession.mode !== "running") throw new Error("interrupted");
    });
    if (epoch !== sessionEpoch || ended || version !== browserSession.version || !socket ||
        socket.readyState !== WebSocket.OPEN || !handshakeComplete) return;
    const frame = makeEnvelope("document_research", {mode: selected || preparedId ? "specific" : "automatic", question,
      page_state: page, allow_remote_context: remoteConsent,
      ...(selected ? {offer_id: selected.offer_id, result_ref: selected.result_ref} : preparedId ? {document_id: preparedId} : {})});
    browserSession.beginResearch(frame.id, selected || preparedId ? 1 : researchMaximum,
      selected ? {sourceUrl: page.url, expiresAt: selected.expiresAt, selection: {
        resultRef: selected.result_ref, filename: selected.filename, fileType: selected.fileType}} : undefined);
    pendingQueryId = frame.id; pendingQueryVersion = browserSession.version;
    researchId = frame.id; researchVersion = browserSession.version; researchCancelled = false;
    fileActionId = null; fileActionCancelled = false; specificPdf = null; specificPrepared = null; documentPreflight = false;
    updateResearchSelectionControls();
    activeRunId = null;
    ui.input.value = ""; ui.researchSelection.textContent = "";
    ui.timeline.replaceChildren(); ui.tokenUsage.textContent = "0 tokens";
    addMessage("user", question);
    setRunStatus(selected || preparedId ? "Researching this selected PDF..." : "Searching website documents...");
    socket.send(JSON.stringify(frame));
  } catch {
    if (epoch !== sessionEpoch || ended) return;
    documentPreflight = false; invalidateResearch(); researchId = null;
    files.invalidate();
    addMessage("error", "Research could not start with current tab/result authority. Refresh results and ask again.");
    setRunStatus("Error"); setBusy(false);
  }
}

ui.composer.addEventListener("submit", (event) => {
  event.preventDefault();
  void send();
});
ui.input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    void send();
  }
});
ui.stop.addEventListener("click", () => {
  const stoppedResearch = researchId !== null;
  invalidateResearch(); researchId = null;
  fileActionCancelled = true; files.invalidate();
  if (documentPreflight || refreshId !== null) sessionEpoch++;
  if (documentPreflight) { documentPreflight = false; files.invalidate(); setBusy(false); }
  if (refreshId !== null) { refreshCancelled = true; files.invalidate(); }
  browserSession.stop();
  confirmation.cancel();
  if (socket?.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify(makeEnvelope("stop", { run_id: activeRunId })));
    setRunStatus("Stopping...");
    ui.stop.disabled = true;
  }
  if (stoppedResearch) { setBusy(false); setRunStatus("Stopped"); }
});
ui.researchMode.addEventListener("change", () => {
  if (specificPdf || specificPrepared) ui.researchMode.checked = false;
  ui.researchRemote.checked = false;
  updateResearchSelectionControls();
});
ui.researchSelectionClear.addEventListener("click", () => {
  if (awaitingTurn || ended) return;
  specificPdf = null; specificPrepared = null; activeDocumentContext = false;
  if (socket?.readyState === WebSocket.OPEN && handshakeComplete) socket.send(JSON.stringify(makeEnvelope("document_context_clear", {})));
  ui.researchSelection.textContent = ""; ui.researchRemote.checked = false;
  updateResearchSelectionControls();
});
ui.clear.addEventListener("click", () => {
  // Disconnect discards the backend DocumentSession (including its ready set)
  // and cancels its run using existing cleanup. No persistent store is touched.
  sessionEpoch++;
  closeSocket(); confirmation.reset();
  browserSession.target = null;
  activeRunId = null; runStarted = 0;
  ui.input.value = "";
  ui.researchMode.checked = false; ui.researchRemote.checked = false; ui.researchSelection.textContent = "";
  ui.conversation.replaceChildren(ui.emptyState);
  ui.emptyState.hidden = false;
  ui.timeline.replaceChildren();
  finishStream();
  setBusy(false);
  ui.tokenUsage.textContent = "0 tokens"; ui.elapsed.textContent = "0.0s";
  setConnection("disconnected", "Session cleared - reconnects on Send");
  setRunStatus("Ready");
});
ui.optionsButton.addEventListener("click", () => void chrome.runtime.openOptionsPage());
function updateControlButtons(): void {
  const paused = browserSession.mode === "paused" || browserSession.mode === "waiting";
  ui.send.textContent = !awaitingTurn ? "Send" : paused ? "Paused" : browserSession.mode === "resuming" ? "Resuming..." : "Working...";
  ui.pause.disabled = !awaitingTurn || !activeRunId || paused;
  ui.resume.disabled = !awaitingTurn || !paused;
  ui.confirmationPause.disabled = ui.pause.disabled;
  ui.confirmationResume.disabled = ui.resume.disabled;
}
function controlMessage(type: "pause" | "resume" | "session_event", event?: string): void {
  if (socket?.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify(makeEnvelope(type, { run_id: activeRunId, ...(event ? { event } : {}) })));
  }
}
function pauseRun(): void {
  if (!awaitingTurn) return;
  invalidateResearch();
  fileActionCancelled = true; files.invalidate();
  browserSession.pause(); confirmation.setPaused(true);
  setRunStatus("Paused - You are controlling the browser");
  updateControlButtons(); controlMessage("pause");
}
function resumeRun(): void {
  if (!awaitingTurn || ended || !ownerReady) return;
  confirmation.cancel(); browserSession.resume();
  setRunStatus("Resuming - Reading this tab's fresh page");
  updateControlButtons(); controlMessage("resume");
}
ui.pause.addEventListener("click", pauseRun);
ui.resume.addEventListener("click", resumeRun);
ui.confirmationPause.addEventListener("click", pauseRun);
ui.confirmationResume.addEventListener("click", resumeRun);
browserSession.onInvalidation = event => {
  invalidateResearch();
  fileActionCancelled = true; files.invalidate();
  if (refreshId !== null) refreshCancelled = true;
  confirmation.setPaused(true); updateControlButtons();
  setRunStatus(event === "target_tab_closed" ? "Target tab closed" : "Target changed - Resume to read fresh state");
  controlMessage("session_event", event);
};
chrome.tabs.onActivated.addListener(info => {
  if (!ownerReady || ended) return;
  browserSession.activated(info.tabId, info.windowId);
  if (info.tabId === browserSession.ownerTabId) void refreshHeader();
});
chrome.tabs.onUpdated.addListener((id, info) => {
  if (!ownerReady || ended || id !== browserSession.ownerTabId) return;
  if (info.url && settings && !isAllowed(info.url)) {
    endOwner("target_document_changed"); return;
  }
  if (!awaitingTurn && id === browserSession.target?.tab_id && (info.url || info.status === "loading")) {
    invalidateResearch();
    files.invalidate();
    controlMessage("session_event", "target_document_changed");
  }
  browserSession.updated(id, info);
  if (info.status === "complete" || info.title) void refreshHeader();
});
chrome.tabs.onRemoved.addListener(id => {
  if (!ownerReady || ended) return;
  if (id === browserSession.ownerTabId) { endOwner("target_tab_closed"); return; }
  browserSession.removed(id);
});
chrome.windows.onFocusChanged.addListener(id => {
  if (!ownerReady || ended || id === chrome.windows.WINDOW_ID_NONE) return;
  if (id !== browserSession.ownerWindowId) browserSession.invalidate("active_target_changed");
});
chrome.tabs.onDetached.addListener(id => {
  if (id === browserSession.ownerTabId) browserSession.invalidate("active_target_changed");
});
chrome.tabs.onAttached.addListener((id, info) => {
  if (id === browserSession.ownerTabId) { browserSession.ownerWindowId = info.newWindowId; void refreshHeader(); }
});
function endOwner(event: "target_tab_closed" | "target_document_changed" | "side_panel_closed"): void {
  if (ended) return;
  ended = true;
  browserSession.stop(); confirmation.cancel();
  controlMessage("session_event", event);
  closeSocket(); setBusy(false);
  ui.send.disabled = true;
  setConnection("disconnected", "Tab session ended");
  setRunStatus(event === "target_tab_closed" ? "Owning tab closed" : "Reopen the assistant on a supported tab");
}
window.addEventListener("pagehide", () => {
  endOwner("side_panel_closed");
});
onSettingsChanged(() => void (async () => {
  const epoch = sessionEpoch;
  if (!ownerReady || ended) return;
  settings = await loadSettings();
  const owner = await browserSession.ownerTab();
  if (epoch !== sessionEpoch || ended) return;
  if (!owner || !supportedPanelTab(owner)) { endOwner("target_document_changed"); return; }
  closeSocket();
  if (awaitingTurn) { setBusy(false); setRunStatus("Stopped - settings changed"); }
  await ensureConnected(settings).catch(() => undefined);
})());
void (async () => {
  const epoch = sessionEpoch;
  ui.send.disabled = true;
  settings = await loadSettings();
  try {
    const id = await resolvePanelOwner(location.href);
    const tab = await chrome.tabs.get(id);
    if (!supportedPanelTab(tab)) throw new Error("Outside supported scope");
    browserSession.bindOwner(id, tab.windowId);
    ownerReady = true; ui.send.disabled = false;
  } catch {
    setConnection("error", "Open the assistant from a supported tab's toolbar icon");
    return;
  }
  await refreshHeader();
  if (epoch !== sessionEpoch || ended) return;
  await ensureConnected(settings).catch(() => undefined);
})();
