import { defineConfig, type Plugin } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const DEBUG_FORMS_DIR = path.resolve(__dirname, '../debug/forms')
const DEMO_SCREENSHOTS_DIR = path.resolve(__dirname, '../data/demo_forms/images')

function serveStaticDir(baseDir: string, urlPrefix: string): Plugin {
  return {
    name: `static-${urlPrefix}`,
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if (!req.url?.startsWith(urlPrefix)) return next()
        const rel = decodeURIComponent(req.url.slice(urlPrefix.length)).replace(/^\//, '')
        if (!rel || rel.includes('..') || rel.includes('/')) {
          res.statusCode = 400
          res.end('Invalid path')
          return
        }
        const filePath = path.join(baseDir, rel)
        const resolved = path.resolve(filePath)
        if (
          !resolved.startsWith(baseDir) ||
          !fs.existsSync(resolved) ||
          !fs.statSync(resolved).isFile()
        ) {
          res.statusCode = 404
          res.end('Not found')
          return
        }
        const ext = path.extname(resolved).toLowerCase()
        const mime: Record<string, string> = {
          '.png': 'image/png',
          '.jpg': 'image/jpeg',
          '.jpeg': 'image/jpeg',
        }
        res.setHeader('Content-Type', mime[ext] ?? 'application/octet-stream')
        fs.createReadStream(resolved).pipe(res)
      })
    },
  }
}

function debugFormsPlugin(): Plugin {
  return {
    name: 'debug-forms',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if (!req.url?.startsWith('/debug-forms')) {
          return next()
        }

        const urlPath = decodeURIComponent(req.url.replace('/debug-forms', ''))

        const listMatch = urlPath.match(/^\/([^/]+)\/list\/?$/)
        if (listMatch) {
          const folder = listMatch[1]
          const dir = path.join(DEBUG_FORMS_DIR, folder)
          if (!fs.existsSync(dir) || !fs.statSync(dir).isDirectory()) {
            res.statusCode = 404
            res.setHeader('Content-Type', 'application/json')
            res.end(JSON.stringify({ files: [] }))
            return
          }
          const files = fs
            .readdirSync(dir)
            .filter((f) => /\.(png|jpe?g|gif|webp)$/i.test(f))
            .sort()
          res.setHeader('Content-Type', 'application/json')
          res.end(JSON.stringify({ files }))
          return
        }

        const fileMatch = urlPath.match(/^\/([^/]+)\/(.+)$/)
        if (fileMatch) {
          const [, folder, filename] = fileMatch
          if (filename.includes('..') || filename.includes('/')) {
            res.statusCode = 400
            res.end('Invalid path')
            return
          }
          const filePath = path.join(DEBUG_FORMS_DIR, folder, filename)
          const resolved = path.resolve(filePath)
          if (
            !resolved.startsWith(DEBUG_FORMS_DIR) ||
            !fs.existsSync(resolved) ||
            !fs.statSync(resolved).isFile()
          ) {
            res.statusCode = 404
            res.end('Not found')
            return
          }
          const ext = path.extname(resolved).toLowerCase()
          const mime: Record<string, string> = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.webp': 'image/webp',
          }
          res.setHeader('Content-Type', mime[ext] ?? 'application/octet-stream')
          fs.createReadStream(resolved).pipe(res)
          return
        }

        next()
      })
    },
  }
}

export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
    debugFormsPlugin(),
    serveStaticDir(DEMO_SCREENSHOTS_DIR, '/demo-screenshots'),
  ],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8001',
        changeOrigin: true,
      },
    },
  },
})
