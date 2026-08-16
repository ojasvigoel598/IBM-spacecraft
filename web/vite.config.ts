import path from 'path'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
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
