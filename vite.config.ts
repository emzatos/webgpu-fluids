import { defineConfig } from "vite";
import typegpu from "unplugin-typegpu/vite";

export default defineConfig({
  plugins: [typegpu()],
  build: {
    rollupOptions: { input: { demo: "index.html", benchmark: "bench/index.html" } },
  },
});
