/**
 * Wire protocol shared by the extension (TypeScript) and the backend (Python).
 *
 * The Python mirror lives at backend/browser/protocol.py. The two files are kept
 * in sync by hand; the field names below are the contract.
 *
 * Every frame on the socket is an Envelope. Nothing is sent bare.
 */

import buildContract from "./build-contract.json";
export const PROTOCOL_VERSION = buildContract.protocol_version;
export const PROTOCOL_REVISION = buildContract.protocol_revision;
export const APPLICATION_VERSION = buildContract.application_version;
export const PROTOCOL_MISMATCH = "Backend/extension protocol mismatch. Rebuild and reload extension/dist, then restart backend.";

export function compatibleHello(payload: unknown): boolean {
  if (!payload || typeof payload !== "object") return false;
  const value = payload as Record<string, unknown>;
  return value["protocol_version"] === PROTOCOL_VERSION && value["protocol_revision"] === PROTOCOL_REVISION;
}

/** Frames the client (side panel) sends to the backend. */
export type ClientMessageType =
  | "auth"
  | "hello"
  | "user_message"
  | "document_prepare"
  | "document_question"
  | "document_refresh"
  | "document_file_action"
  | "document_research"
  | "document_context_clear"
  | "tool_result"
  | "confirmation_response"
  | "pause"
  | "resume"
  | "session_event"
  | "stop"
  | "ping";

/** Frames the backend sends to the client. */
export type ServerMessageType =
  | "hello_ack"
  | "assistant_delta"
  | "assistant_done"
  | "tool_request"
  | "confirmation_request"
  | "run_status"
  | "control_state"
  | "timeline_step"
  | "document_choices"
  | "document_progress"
  | "document_ready"
  | "document_done"
  | "research_plan"
  | "research_file"
  | "routing_status"
  | "error"
  | "pong";

export type MessageType = ClientMessageType | ServerMessageType;

/**
 * The envelope. `id` is unique per frame. `corr_id` points at the `id` of the
 * frame that caused this one, so a stream of deltas can be tied to the request
 * that produced it.
 */
export interface Envelope<T = unknown> {
  v: number;
  id: string;
  type: MessageType;
  ts: string;
  corr_id?: string | null;
  payload: T;
}

/* ------------------------------------------------------------------ */
/* Page state                                                          */
/* ------------------------------------------------------------------ */

export interface PageElement {
  /** Stable within one snapshot; also stamped onto the DOM node as data-agent-id. */
  agent_id: string;
  tag: string;
  role: string;
  /** Accessible name, computed per the accname algorithm. May be "". */
  name: string;
  text: string;
  placeholder: string;
  value: string;
  input_type: string;
  /** Absolute link destination, retained for approval-time identity checks. */
  href: string;
  /** Effective enclosing form destination and submission metadata. */
  form_action: string;
  form_method: string;
  form_target: string;
  /** Generic structural metadata used by configured backend site adapters. */
  dom_id: string;
  test_id: string;
  data_key: string;
  semantic_region: string;
  selected: boolean;
  enabled: boolean;
  visible: boolean;
  occluded: boolean;
  in_viewport: boolean;
  bbox: number[];
  document_order: number;
}

export interface PageHeading {
  level: number;
  text: string;
}

export interface PageRegion {
  region_id: string;
  tag: string;
  role: string;
  name: string;
  text: string;
  dom_id: string;
  test_id: string;
  data_key: string;
  class_tokens: string[];
  href: string;
  visible: boolean;
  ancestor_region_ids: string[];
  agent_element_ids: string[];
}

export interface PageState {
  /** Set by the extension transport, never by page-authored DOM data. */
  tab_id?: number | null;
  snapshot_id: string;
  /** Changes on a full document load; stable across same-document mutations. */
  document_id: string;
  /** MAIN-world history transition counter. */
  navigation_id: number;
  url: string;
  title: string;
  frames: Array<{ frame_id: string; url: string; origin: string }>;
  visible_text: string;
  headings: PageHeading[];
  elements: PageElement[];
  truncated: { text: boolean; elements: boolean };
  opaque_regions: string[];
  warnings: string[];
  dialogs: Array<{
    type: "alert" | "confirm" | "prompt";
    message: string;
    dismissed: boolean;
  }>;
  looks_like_login: boolean;
  /** Raw generic structure for backend adapters; never rendered directly. */
  semantic_regions: PageRegion[];
  /** Bounded site-specific evidence; normalized by the backend and never rendered directly. */
  adapter_payload?: Record<string, unknown> | null;
  site_context?: Record<string, unknown> | null;
}

/* ------------------------------------------------------------------ */
/* Payloads                                                            */
/* ------------------------------------------------------------------ */

export interface AuthPayload {
  token: string;
}

export interface HelloPayload {
  client: string;
  protocol_version: number;
  protocol_revision: number;
}

export interface HelloAckPayload {
  backend_version: string;
  server: string;
  protocol_version: number;
  protocol_revision: number;
  session_id: string;
  provider: string;
  model: string;
  research_max_pdfs?: number;
  research_max_visual_pages?: number;
}

export interface UserMessagePayload {
  allow_remote_context?: boolean;
  text: string;
  page_state: PageState;
  allowlist: string[];
}

/** Only a direct panel click sends this. No URL, bulk refs or provider authority. */
export interface DocumentFileActionPayload {
  offer_id: string;
  result_ref: string;
  action: "preview" | "download" | "prepare";
}

/** Explicit task authorization; content transmission consent remains independent. */
export interface DocumentResearchPayload {
  mode: "automatic" | "specific";
  question: string;
  page_state: PageState;
  allowlist: string[];
  allow_remote_context: boolean;
  offer_id?: string;
  result_ref?: string;
  document_id?: string;
}

export interface AssistantDeltaPayload {
  text: string;
}

export interface AssistantDonePayload {
  /** Full assistant answer, so a client that dropped deltas can still recover. */
  text: string;
  finish_reason: string;
  usage: Record<string, number>;
  elapsed_ms: number;
}

export type ToolError =
  | "tool_not_available_in_current_profile"
  | "no_active_search_results"
  | "element_not_found"
  | "stale_snapshot"
  | "target_tab_changed"
  | "target_document_changed"
  | "target_tab_closed"
  | "run_paused"
  | "element_occluded"
  | "element_disabled"
  | "timeout"
  | "blocked_by_dialog"
  | "navigation_blocked"
  | "permission_denied"
  | "frame_unreachable"
  | "injection_failed"
  | "invalid_arguments"
  | "password_field";

export interface ActionResult {
  success: boolean;
  action: string;
  element_id?: string | null;
  element_name?: string | null;
  snapshot_id: string;
  page_changed: boolean;
  reresolved: boolean;
  duration_ms: number;
  error?: ToolError | null;
  message: string;
  data: Record<string, unknown>;
}

export interface ToolRequestPayload {
  tab_id?: number | null;
  document_id?: string;
  control_epoch?: number;
  reset_target?: boolean;
  tab_revision?: string | null;
  run_id: string;
  step: number;
  max_steps: number;
  tool: string;
  arguments: Record<string, unknown>;
  snapshot_id: string;
  reason: string;
}

export interface ToolResultPayload {
  result: ActionResult;
}

export interface ConfirmationRequestPayload {
  run_id: string;
  step: number;
  tool: string;
  target: string;
  arguments: Record<string, unknown>;
  reason: string;
  risk: "high";
  risk_reasons: string[];
  scope: string;
}

export interface ConfirmationResponsePayload {
  decision: "allow_once" | "allow_run" | "cancel";
}

export interface ControlStatePayload {
  run_id: string;
  state: "running" | "paused" | "resuming" | "waiting" | "stopped";
  epoch: number;
  reason: string;
}

export interface TabMetadata {
  tab_id: number;
  active: boolean;
  allowed: boolean;
  opened_by_agent?: boolean;
  title?: string;
  location?: string;
  revision?: string;
  details?: "redacted";
}

export interface StopPayload {
  run_id?: string | null;
}

export interface RunStatusPayload {
  run_id: string;
  status: string;
  step: number;
  max_steps: number;
  elapsed_ms: number;
  usage: Record<string, number>;
}

export interface TimelineStepPayload {
  run_id: string;
  step: number;
  tool: string;
  target: string;
  reason: string;
  result: ActionResult;
}

export type ErrorCode =
  | "unauthorized"
  | "bad_request"
  | "protocol_error"
  | "protocol_mismatch"
  | "provider_error"
  | "internal_error";

export interface ErrorPayload {
  code: ErrorCode;
  message: string;
}

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

export function makeEnvelope<T>(
  type: MessageType,
  payload: T,
  corrId?: string | null,
): Envelope<T> {
  return {
    v: PROTOCOL_VERSION,
    id: crypto.randomUUID(),
    type,
    ts: new Date().toISOString(),
    corr_id: corrId ?? null,
    payload,
  };
}

/** Narrow an unknown parsed JSON value to an Envelope. Throws on garbage. */
export function parseEnvelope(raw: string): Envelope {
  const data: unknown = JSON.parse(raw);
  if (typeof data !== "object" || data === null) {
    throw new Error("envelope is not an object");
  }
  const e = data as Record<string, unknown>;
  if (typeof e["type"] !== "string" || typeof e["id"] !== "string") {
    throw new Error("envelope is missing id/type");
  }
  return {
    v: typeof e["v"] === "number" ? e["v"] : PROTOCOL_VERSION,
    id: e["id"],
    type: e["type"] as MessageType,
    ts: typeof e["ts"] === "string" ? e["ts"] : new Date().toISOString(),
    corr_id: typeof e["corr_id"] === "string" ? e["corr_id"] : null,
    payload: e["payload"],
  };
}
