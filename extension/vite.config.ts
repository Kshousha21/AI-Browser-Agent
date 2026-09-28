import { resolve } from "node:path";
import { defineConfig } from "vite";

/**
 * Main build: the two HTML surfaces and the service worker.
 *
 * The content script is built separately (vite.content.config.ts) because it
 * has to be a classic script, not an ES module -- chrome.scripting.executeScript
 * cannot inject a module.
 */
export default defineConfig({
  root: resolve(import.meta.dirname, "src"),
  base: "./",
  publicDir: resolve(import.meta.dirname, "public"),
  resolve: {
    alias: { "@shared": resolve(import.meta.dirname, "../shared") },
  },
  build: {
    outDir: resolve(import.meta.dirname, "dist"),
    emptyOutDir: true,
    target: "chrome116",
    sourcemap: true,
    minify: false, // an unminified bundle is far easier to debug in the panel
    rolldownOptions: {
      input: {
        sidepanel: resolve(import.meta.dirname, "src/sidepanel/index.html"),
        options: resolve(import.meta.dirname, "src/options/index.html"),
        "service-worker": resolve(import.meta.dirname, "src/background/service-worker.ts"),
      },
      output: {
        // The worker path is named in manifest.json, so it may not be hashed.
        entryFileNames: (chunk) =>
          chunk.name === "service-worker"
            ? "service-worker.js"
            : "assets/[name]-[hash].js",
        chunkFileNames: "assets/[name]-[hash].js",
        assetFileNames: "assets/[name]-[hash][extname]",
      },
    },
  },
});
