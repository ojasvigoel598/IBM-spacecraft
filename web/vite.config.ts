import path from 'path'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  build: {
    outDir: '../dist',
    emptyOutDir: true,
  },
  plugins: [
    react(),
    tailwindcss(),
    // Production-only CSP (dev needs Vite's inline react-refresh preamble).
    // The app has no inline scripts/styles beyond React's style attributes.
    {
      name: 'html-csp',
      apply: 'build',
      transformIndexHtml(html) {
        return {
          html,
          tags: [
            {
              tag: 'meta',
              attrs: {
                'http-equiv': 'Content-Security-Policy',
                content:
                  "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; " +
                  "img-src 'self' data:; font-src 'self' data:; connect-src 'self'; " +
                  "object-src 'none'; base-uri 'self'; frame-ancestors 'none'; form-action 'self'",
              },
              injectTo: 'head-prepend',
            },
          ],
        }
      },
    },
  ],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    proxy: {
      // Dev: forward same-origin /api calls to the local FastAPI backend.
      '/api': 'http://127.0.0.1:8100',
    },
  },
  preview: {
    proxy: {
      '/api': 'http://127.0.0.1:8100',
    },
  },
})
