/**
 * Options page: backend URL, token, and the page-text limit.
 *
 * It also holds the "grant site access" button, because chrome.permissions
 * .request needs a user gesture and this is the natural place to ask.
 */

import {
  DEFAULT_SETTINGS,
  loadSettings,
  saveSettings,
  type ExtensionSettings,
} from "../lib/settings";

/** Look up an element that the page's own HTML guarantees exists. */
function required<T extends Element>(id: string): T {
  const element = document.getElementById(id);
  if (!element) throw new Error(`missing element #${id}`);
  return element as unknown as T;
}

const ui = {
  form: required<HTMLFormElement>("form"),
  backendUrl: required<HTMLInputElement>("backend-url"),
  token: required<HTMLInputElement>("token"),
  site: required<HTMLInputElement>("site"),
  maxChars: required<HTMLInputElement>("max-chars"),
  test: required<HTMLButtonElement>("test"),
  grant: required<HTMLButtonElement>("grant"),
  status: required<HTMLElement>("status"),
};

function setStatus(text: string, kind: "ok" | "error" | "info" = "info"): void {
  ui.status.textContent = text;
  ui.status.dataset["kind"] = kind;
}

function readForm(): ExtensionSettings {
  const parsed = Number.parseInt(ui.maxChars.value, 10);
  return {
    backendUrl: ui.backendUrl.value.trim() || DEFAULT_SETTINGS.backendUrl,
    token: ui.token.value.trim(),
    maxVisibleTextChars:
      Number.isFinite(parsed) && parsed > 0
        ? parsed
        : DEFAULT_SETTINGS.maxVisibleTextChars,
  };
}

function fillForm(settings: ExtensionSettings): void {
  ui.backendUrl.value = settings.backendUrl;
  ui.token.value = settings.token;
  ui.maxChars.value = String(settings.maxVisibleTextChars);
}

ui.form.addEventListener("submit", (event) => {
  event.preventDefault();
  void (async () => {
    const settings = readForm();

    if (!/^wss?:\/\//i.test(settings.backendUrl)) {
      setStatus("Backend URL must start with ws:// or wss://", "error");
      return;
    }

    await saveSettings(settings);
    setStatus("Saved.", "ok");
  })();
});

ui.test.addEventListener("click", () => {
  void (async () => {
    const settings = readForm();
    // /health is plain HTTP on the same host and port as the socket.
    const httpBase = settings.backendUrl
      .replace(/^ws:/i, "http:")
      .replace(/^wss:/i, "https:")
      .replace(/\/+$/, "");

    setStatus("Testing…");
    try {
      const response = await fetch(`${httpBase}/health`);
      if (!response.ok) {
        setStatus(`Backend answered ${response.status}`, "error");
        return;
      }
      const body = (await response.json()) as { active_provider?: string };
      setStatus(
        `Backend reachable - ${body.active_provider ?? "unknown"}. ` +
          "This checks reachability only; protocol and authentication are checked when the panel connects.",
        "ok",
      );
    } catch (error) {
      setStatus(
        `Could not reach ${httpBase}/health: ${
          error instanceof Error ? error.message : String(error)
        }`,
        "error",
      );
    }
  })();
});

ui.grant.addEventListener("click", () => {
  void (async () => {
    try {
      const site = new URL(ui.site.value.trim());
      if (!["http:", "https:"].includes(site.protocol) || site.username || site.password) throw new Error("Enter a normal HTTP(S) site URL.");
      const origins = [`${site.protocol}//${site.hostname}/*`];
      const granted = await chrome.permissions.request({ origins });
      setStatus(
        granted
          ? `Access granted for: ${origins.join(", ")}`
          : "Permission was declined.",
        granted ? "ok" : "error",
      );
    } catch (error) {
      setStatus(
        error instanceof Error ? error.message : String(error),
        "error",
      );
    }
  })();
});

void (async () => {
  fillForm(await loadSettings());
})();
