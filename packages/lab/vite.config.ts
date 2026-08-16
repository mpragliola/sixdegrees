import { defineConfig } from "vite";

export default defineConfig({
  base: process.env.VITE_BASE_PATH ?? "/",
  optimizeDeps: {
    exclude: ["@huggingface/transformers"],
  },
  worker: {
    format: "es",
  },
});
