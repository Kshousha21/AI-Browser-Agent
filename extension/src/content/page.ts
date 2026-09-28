import { normalizeText } from "./names";
import { scanInjection } from "./injection";
import { ElementRegistry } from "./registry";
import type { ExtractOptions, PageState } from "./types";
import { hiddenReason, hiddenTag, isRendered } from "./visibility";
import { collectSemanticRegions } from "./regions";

const BLOCK_TAGS = new Set([
  "ADDRESS", "ARTICLE", "ASIDE", "BLOCKQUOTE", "BR", "DD", "DIV", "DL", "DT",
  "FIELDSET", "FIGCAPTION", "FIGURE", "FOOTER", "FORM", "H1", "H2", "H3",
  "H4", "H5", "H6", "HEADER", "HR", "LI", "MAIN", "NAV", "OL", "P", "PRE",
  "SECTION", "TABLE", "TD", "TH", "TR", "UL",
]);

export const INSTRUMENT_ATTRIBUTE = "data-ai-browser-agent-runtime";
export const DOCUMENT_ID = `d-${crypto.randomUUID()}`;

export interface InstrumentState {
  inflight: number;
  navigation: number;
  dialogs: Array<{
    type: "alert" | "confirm" | "prompt";
    message: string;
    dismissed: boolean;
  }>;
}

export function instrumentState(): InstrumentState {
  const raw = document.documentElement?.getAttribute(INSTRUMENT_ATTRIBUTE);
  if (!raw) return { inflight: 0, navigation: 0, dialogs: [] };
  try {
    const value = JSON.parse(raw) as Partial<InstrumentState>;
    return {
      inflight: typeof value.inflight === "number" ? value.inflight : 0,
      navigation: typeof value.navigation === "number" ? value.navigation : 0,
      dialogs: Array.isArray(value.dialogs) ? value.dialogs : [],
    };
  } catch {
    return { inflight: 0, navigation: 0, dialogs: [] };
  }
}

function collectText(node: Node, output: string[], warnings: Set<string>): void {
  if (node.nodeType === Node.TEXT_NODE) {
    if (node.textContent?.trim()) output.push(node.textContent);
    return;
  }
  if (node.nodeType !== Node.ELEMENT_NODE) return;
  const element = node as Element;
  if (hiddenTag(element)) return;
  const reason = hiddenReason(element);
  if (reason) {
    if (scanInjection(element.textContent ?? "").length > 0) {
      warnings.add("security:hidden_injection_pattern");
    }
    if (reason === "matching-colours") {
      warnings.add("security:obfuscated_text_excluded");
    }
    return;
  }
  const block = BLOCK_TAGS.has(element.tagName.toUpperCase());
  if (block) output.push("\n");
  for (const child of Array.from(element.childNodes)) {
    collectText(child, output, warnings);
  }
  if (block) output.push("\n");
}

function visibleText(
  limit: number,
): { text: string; truncated: boolean; warnings: string[] } {
  const parts: string[] = [];
  const warnings = new Set<string>();
  if (document.body) collectText(document.body, parts, warnings);
  const text = parts
    .join(" ")
    .replace(/[^\S\n]+/g, " ")
    .replace(/ *\n */g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
  for (const warning of scanInjection(text)) warnings.add(warning);
  return text.length > limit
    ? { text: text.slice(0, limit), truncated: true, warnings: [...warnings] }
    : { text, truncated: false, warnings: [...warnings] };
}

function headings(): Array<{ level: number; text: string }> {
  const output: Array<{ level: number; text: string }> = [];
  for (const node of Array.from(
    document.querySelectorAll<HTMLElement>(
      "h1,h2,h3,h4,h5,h6,[role='heading']",
    ),
  )) {
    if (!isRendered(node)) continue;
    const text = normalizeText(node.textContent ?? "");
    if (!text) continue;
    const native = /^H([1-6])$/.exec(node.tagName);
    const aria = Number.parseInt(node.getAttribute("aria-level") ?? "", 10);
    output.push({
      level: native?.[1] ? Number(native[1]) : Number.isFinite(aria) ? aria : 2,
      text,
    });
  }
  return output;
}

function looksLikeLogin(): boolean {
  const urlSignal = /(?:^|[/?#._-])(login|signin|auth|sso)(?:[/?#._-]|$)/i.test(
    location.href,
  );
  const password = document.querySelector("input[type='password']");
  const submit = document.querySelector(
    "button[type='submit'],input[type='submit'],button:not([type])",
  );
  return urlSignal || Boolean(password && submit);
}

export function extractSnapshot(
  registry: ElementRegistry,
  options: ExtractOptions,
): PageState {
  const textLimit =
    Number.isFinite(options.maxVisibleTextChars) && options.maxVisibleTextChars > 0
      ? options.maxVisibleTextChars
      : 20_000;
  const elementLimit =
    Number.isFinite(options.maxElements) && options.maxElements > 0
      ? options.maxElements
      : 300;
  const text = visibleText(textLimit);
  const rebuilt = registry.rebuild(elementLimit);
  const state = instrumentState();
  return {
    snapshot_id: rebuilt.snapshotId,
    document_id: DOCUMENT_ID,
    navigation_id: state.navigation,
    url: location.href,
    title: document.title,
    frames: [{ frame_id: "f0", url: location.href, origin: location.origin }],
    visible_text: text.text,
    headings: headings(),
    elements: rebuilt.elements,
    truncated: { text: text.truncated, elements: rebuilt.truncated },
    opaque_regions: [],
    warnings: text.warnings,
    dialogs: state.dialogs,
    looks_like_login: looksLikeLogin(),
    semantic_regions: collectSemanticRegions(),
    adapter_payload: null,
    site_context: null,
  };
}
