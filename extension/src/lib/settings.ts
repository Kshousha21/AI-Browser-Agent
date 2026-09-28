/**
 * Extension settings, held in chrome.storage.local.
 *
 * chrome.storage.sync would push the backend token through the user's Google
 * account, so local is deliberate.
 */


export interface ExtensionSettings {
  /** Backend WebSocket base, e.g. ws://127.0.0.1:8765 */
  backendUrl: string;
  /** Shared secret; must match auth.token on the backend. */
  token: string;
  /** visible_text is cut at this many characters before it leaves the page. */
  maxVisibleTextChars: number;
}

export const DEFAULT_SETTINGS: ExtensionSettings = {
  backendUrl: "ws://127.0.0.1:8765",
  token: "",
  maxVisibleTextChars: 20000,
};

const STORAGE_KEY = "settings";

export async function loadSettings(): Promise<ExtensionSettings> {
  const stored = await chrome.storage.local.get(STORAGE_KEY);
  const raw: unknown = stored[STORAGE_KEY];
  if (typeof raw !== "object" || raw === null) {
    return { ...DEFAULT_SETTINGS };
  }
  const partial = raw as Partial<ExtensionSettings>;
  return {
    backendUrl: partial.backendUrl?.trim() || DEFAULT_SETTINGS.backendUrl,
    token: partial.token ?? DEFAULT_SETTINGS.token,
    maxVisibleTextChars:
      typeof partial.maxVisibleTextChars === "number" &&
      partial.maxVisibleTextChars > 0
        ? partial.maxVisibleTextChars
        : DEFAULT_SETTINGS.maxVisibleTextChars,
  };
}

export async function saveSettings(settings: ExtensionSettings): Promise<void> {
  await chrome.storage.local.set({ [STORAGE_KEY]: settings });
}

export function onSettingsChanged(handler: () => void): void {
  chrome.storage.onChanged.addListener((changes, area) => {
    if (area === "local" && STORAGE_KEY in changes) {
      handler();
    }
  });
}

/** Build the unauthenticated WebSocket URL.
 *
 * The token is sent in the first protocol frame after the socket opens. Keeping
 * it out of the URL prevents it from appearing in request and access logs.
 */
export function buildSocketUrl(settings: ExtensionSettings): string {
  const base = settings.backendUrl.replace(/\/+$/, "");
  return `${base}/ws`;
}
