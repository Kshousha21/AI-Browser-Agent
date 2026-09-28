(() => {
  if (globalThis.__aiBrowserAgentInstrumented) return;
  globalThis.__aiBrowserAgentInstrumented = true;

  const attribute = "data-ai-browser-agent-runtime";
  const state = { inflight: 0, navigation: 0, dialogs: [] };
  const sync = () => {
    if (document.documentElement) {
      document.documentElement.setAttribute(attribute, JSON.stringify(state));
    }
  };
  const recordDialog = (type, message) => {
    state.dialogs.push({ type, message: String(message ?? ""), dismissed: true });
    if (state.dialogs.length > 20) state.dialogs.shift();
    sync();
  };
  const network = (delta) => {
    state.inflight = Math.max(0, state.inflight + delta);
    sync();
  };

  const nativeFetch = globalThis.fetch;
  if (nativeFetch) {
    globalThis.fetch = function (...args) {
      network(1);
      return nativeFetch.apply(this, args).finally(() => network(-1));
    };
  }
  const nativeSend = XMLHttpRequest.prototype.send;
  XMLHttpRequest.prototype.send = function (...args) {
    network(1);
    this.addEventListener("loadend", () => network(-1), { once: true });
    return nativeSend.apply(this, args);
  };

  for (const name of ["pushState", "replaceState"]) {
    const native = history[name];
    history[name] = function (...args) {
      const result = native.apply(this, args);
      state.navigation += 1;
      sync();
      dispatchEvent(new CustomEvent("ai-browser-agent-page-changed"));
      return result;
    };
  }
  addEventListener("popstate", () => {
    state.navigation += 1;
    sync();
    dispatchEvent(new CustomEvent("ai-browser-agent-page-changed"));
  });

  globalThis.alert = (message) => recordDialog("alert", message);
  globalThis.confirm = (message) => {
    recordDialog("confirm", message);
    return false;
  };
  globalThis.prompt = (message) => {
    recordDialog("prompt", message);
    return null;
  };
  if (document.documentElement) sync();
  else document.addEventListener("DOMContentLoaded", sync, { once: true });
})();
