import { defineConfig } from 'vitest/config';
import path from 'path';
import { sveltekit } from '@sveltejs/kit/vite';

export default defineConfig({
    test: {
        environment: 'jsdom',
        include: ['src/**/*.{test,spec}.{js,ts}'],
        globals: true
    },
    plugins: [
        sveltekit({ hot: false }),
    ],
    resolve: {
        alias: {
            $lib: path.resolve(__dirname, './src/lib'),
            '$lib/config/api': path.resolve(__dirname, './src/lib/config/api.ts')
        }
    }
});
