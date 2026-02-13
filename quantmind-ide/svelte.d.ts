// Svelte component module declaration
// This allows TypeScript to recognize .svelte files as modules with default exports
/// <reference types="svelte" />
declare module '*.svelte' {
	export { SvelteComponent as default } from 'svelte';
}
