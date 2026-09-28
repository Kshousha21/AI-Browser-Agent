/** HTTP(S) validation only. Chrome owns actual host authorization. */
export function isProductUrl(value: string): boolean {
  try {
    if (/[\\\s]/.test(value)) return false;
    const url = new URL(value);
    return ["http:", "https:"].includes(url.protocol) && !!url.hostname && !url.username && !url.password;
  } catch { return false; }
}
