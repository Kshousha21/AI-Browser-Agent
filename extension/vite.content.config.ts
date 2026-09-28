import { resolve } from "node:path";
import { defineConfig } from "vite";

/**
 * The content script, built as a single classic (IIFE) file.
 *
 * chrome.scripting.executeScript({files: [...]}) evaluates a classic script, so
 * this bundle cannot be an ES module and cannot be split into chunks. It hangs
 * its entry point off a global, which the side panel then calls in a second
 * executeScript.
 */
export default defineConfig({
  // The main build already copied public/ into dist/; without this the manifest
  // would be duplicated into dist/content/.
  publicDir: false,
  resolve: {
    alias: { "@shared": resolve(import.meta.dirname, "../shared") },
  },
  build: {
    outDir: resolve(import.meta.dirname, "dist/content"),
    emptyOutDir: false, // the main build already populated dist/
    target: "chrome116",
    sourcemap: true,
    minify: false,
    lib: {
      entry: resolve(import.meta.dirname, "src/content/extract.ts"),
      formats: ["iife"],
      name: "__aiBrowserAgentContent",
      fileName: () => "extract.js",
    },
  },
});
