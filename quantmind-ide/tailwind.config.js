/** @type {import('tailwindcss').Config} */
import { skeleton } from '@skeletonlabs/tw-plugin';

export default {
  content: [
    './src/**/*.{html,svelte,js,ts}',
  ],
  theme: {
    extend: {},
  },
  plugins: [
    skeleton({
      themes: {
        skeleton: [ '@skeletonlabs/skeleton/themes/iceberg.css' ]
      }
    }),
  ],
}