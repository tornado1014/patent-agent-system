/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Enable experimental features for CopilotKit
  experimental: {
    serverActions: {
      bodySizeLimit: '10mb',
    },
  },
  // Proxy API requests to Python backend in development
  async rewrites() {
    return [
      {
        source: '/api/backend/:path*',
        destination: 'http://localhost:8000/:path*',
      },
    ];
  },
};

module.exports = nextConfig;
