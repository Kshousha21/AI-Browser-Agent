/** Exact URL equality only; no site-specific aliases. */
export function sameNavigationDestination(actual: string, expected: string): boolean {
  try { return new URL(actual).href === new URL(expected).href; } catch { return false; }
}
