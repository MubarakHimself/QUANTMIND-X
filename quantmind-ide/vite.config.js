import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';
import monacoEditorPlugin from 'vite-plugin-monaco-editor';

export default defineConfig({
    plugins: [
        sveltekit(),
        monacoEditorPlugin.default({
            languageWorkers: ['editorWorkerService', 'typescript', 'json', 'css', 'html']
        })
    ],
    server: {
        host: true,
        port: 1420,
        strictPort: true,
        proxy: {
            '/api': {
                target: 'http://localhost:8000',
                changeOrigin: true
            },
            '/ws': {
                target: 'ws://localhost:8000',
                ws: true
            }
        }
    },
    optimizeDeps: {
        exclude: ['@sveltejs/kit'],
        include: ['monaco-editor', 'chart.js', 'd3']
    },
    worker: {
        format: 'es'
    }
});
