import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
    plugins: [
        sveltekit()
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
