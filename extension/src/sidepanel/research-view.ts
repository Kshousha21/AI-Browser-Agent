/** Human intent and local provenance presentation; never model-created targets. */
import type { FileChoices } from "./file-cards";
export interface ResearchRow { source_key: string; result_ref: string; filename: string; file_type: string; }
export function explicitResearch(text: string): boolean {
  return /^(?:research\s+\S|find\s+(?:the\s+latest\s+)?information\s+about\s+\S|look\s+through\s+.{1,300}\b(?:pdfs?|reports?)\b|find\s+.{1,300}\bin\s+(?:these|the)\s+reports\b)/i.test(text.trim());
}
export function pdfLabel(value: unknown): string {
  if (typeof value !== "string" || /[\u0000-\u001f\u007f]/.test(value) ||
      /^(?:[a-z]:[\\/]|\\\\|file:|https?:|\/)/i.test(value)) return "Selected PDF";
  return value.trim().slice(0, 200) || "Selected PDF";
}
export function currentResearchRows(rows: ResearchRow[], choices: FileChoices | null, maximum: number): boolean {
  return observedResearchRows(rows, choices, maximum);
}
/** Current operations must match observed records; time alone grants or revokes nothing. */
export function observedResearchRows(rows: ResearchRow[], choices: FileChoices | null, maximum: number): boolean {
  if (!choices?.offer_id || !rows.length || rows.length > maximum ||
      new Set(rows.map(row => row.result_ref)).size !== rows.length) return false;
  return rows.every(row => {
    const matches = choices.results.filter(choice => choice.result_ref === row.result_ref);
    return matches.length === 1 && matches[0]!.content_type === "attachment" &&
      matches[0]!.file_type?.toLowerCase() === "pdf" && row.file_type === "pdf" &&
      (matches[0]!.attachment_filename || matches[0]!.title) === row.filename;
  });
}
export interface GroundedAnswer {
  answer: string; notices?: string[];
  citations: Array<{document_id: string; filename?: string; page_start: number; page_end: number;
    evidence_type?: string}>;
}
export function groundedText(answer: GroundedAnswer, titles: Map<string, string>): string {
  const citations = answer.citations.filter(citation => /^[a-f0-9]{64}$/.test(citation.document_id) &&
    Number.isSafeInteger(citation.page_start) && citation.page_start > 0 && citation.page_start <= 10000 &&
    citation.page_end === citation.page_start && titles.has(citation.document_id));
  return answer.answer + (citations.length ? "\n" + citations.map(citation =>
    "[" + pdfLabel(titles.get(citation.document_id)) + ", p. " + citation.page_start +
    (citation.evidence_type === "visual" ? " — visual" : citation.evidence_type === "both" ? " — text + visual" : "") + "]").join("\n") : "") +
    (answer.notices?.length ? "\n" + answer.notices.join("\n") : "");
}
