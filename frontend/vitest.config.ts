import { defineConfig } from "vitest/config";

// D45: 只测纯逻辑（turnState / exportSession / pageLabel / sse 解析 / 质量徽标判定）。
// 不引入 jsdom / @testing-library —— 没有组件渲染测试，环境保持默认的 `node`，
// 测试文件与源码同目录（`src/**/*.test.ts`），由 vitest 默认的 include 匹配。
export default defineConfig({
  test: {
    environment: "node",
    include: ["src/**/*.test.ts"],
  },
});
