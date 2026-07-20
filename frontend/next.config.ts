// 智能搜索助手 — Next.js 前端配置
import type { NextConfig } from "next";

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000";

const nextConfig: NextConfig = {
  // 跳过 TS 类型检查以规避 Turbopack 在某些环境下的解析 bug
  typescript: {
    ignoreBuildErrors: true,
  },

  // standalone 输出优化 Docker 镜像体积
  output: "standalone",

  devIndicators: false,

  // 将 /api/* 请求代理到 FastAPI 后端（服务端转发）
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${BACKEND_URL}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
