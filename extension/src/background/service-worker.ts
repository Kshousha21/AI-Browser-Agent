/**
 * Service worker: events only.
 *
 * The side panel owns the WebSocket and talks to chrome.tabs / chrome.scripting
 * itself, so there is deliberately no message routing, no connection state and
 * no page logic here. An MV3 worker is evicted at will; putting a socket in it
 * would mean reconnecting every few minutes.
 */
import "./acquisition-guard";
import { installTabPanels } from "./tab-panels";

chrome.runtime.onInstalled.addListener((details) => {
  console.info("[ai-browser-agent] installed", details.reason);
});

installTabPanels();
