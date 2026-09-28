const PATTERNS: Array<[string, RegExp]> = [
  [
    "security:prompt_injection_pattern",
    /\b(?:ignore|disregard|forget)\b.{0,50}\b(?:previous|prior|system|developer|instructions?|prompt)\b/i,
  ],
  [
    "security:instruction_impersonation",
    /\b(?:system|developer)\s*(?:message|instruction|prompt)\s*:/i,
  ],
  [
    "security:suspicious_navigation_instruction",
    /\b(?:navigate|browse|redirect|go)\s+to\s+https?:\/\//i,
  ],
  [
    "security:credential_exfiltration_instruction",
    /\b(?:send|post|upload|reveal|exfiltrate)\b.{0,80}\b(?:api[_ -]?key|token|cookie|authorization|password|secret)\b/i,
  ],
  [
    "security:prompt_delimiter_text",
    /<\/?(?:untrusted_)?(?:webpage|page)[_ -]?(?:content|snapshot)>/i,
  ],
];

export function scanInjection(text: string): string[] {
  return PATTERNS.filter(([, pattern]) => pattern.test(text)).map(([code]) => code);
}
