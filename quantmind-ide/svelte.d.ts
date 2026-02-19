// Svelte component module declaration
// This allows TypeScript to recognize .svelte files as modules with default exports
/// <reference types="svelte" />
/// <reference types="@sveltejs/kit" />

declare module '*.svelte' {
	import { SvelteComponent } from 'svelte';
	export default class Component extends SvelteComponent {}
}

// SvelteKit $app module declarations
declare module '$app/stores' {
	import type { Readable } from 'svelte/store';

	interface Page {
		url: URL;
		params: Record<string, string>;
		data: Record<string, unknown>;
		error: Error | null;
		status: number;
		form: Record<string, unknown> | null;
	}

	interface Navigation {
		active: boolean;
		type: 'load' | 'goto' | 'popstate' | 'link';
		from: Location & { params: Record<string, string> };
		to: Location & { params: Record<string, string> };
		delta: number;
	}

	interface gotoOptions {
		replaceState?: boolean;
		noScroll?: boolean;
		keepFocus?: boolean;
		state?: Record<string, unknown>;
	}

	export const page: Readable<Page>;
	export const navigating: Readable<Navigation | null>;
	export const preloading: Readable<boolean>;
	export const session: Readable<Record<string, unknown>>;
	// eslint-disable-next-line @typescript-eslint/no-invalid-void-type
	export function goto(href: string, opts?: gotoOptions): Promise<void | number>;
	export function invalidate(href: string): Promise<void>;
	export function invalidateAll(): Promise<void>;
	export function pushState(url: string, state: Record<string, unknown>): Promise<void>;
	export function replaceState(url: string, state: Record<string, unknown>): Promise<void>;
}
