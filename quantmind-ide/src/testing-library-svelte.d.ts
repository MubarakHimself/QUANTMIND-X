// Type declarations for @testing-library/svelte

declare module '@testing-library/svelte' {
  import { SvelteComponent } from 'svelte';

  export interface RenderResult<S> {
    container: HTMLElement;
    component: S;
    debug: (element?: HTMLElement) => string;
    rerender: (props: Partial<S>) => void;
  }

  export interface FireEvent {
    (element: HTMLElement, event: Event): void;
    click: (element: HTMLElement, init?: MouseEventInit) => void;
    change: (element: HTMLElement, init?: EventInit) => void;
    input: (element: HTMLElement, init?: EventInit) => void;
    keydown: (element: HTMLElement, init?: KeyboardEventInit) => void;
    keyup: (element: HTMLElement, init?: KeyboardEventInit) => void;
    submit: (element: HTMLElement, init?: EventInit) => void;
    focus: (element: HTMLElement) => void;
    blur: (element: HTMLElement) => void;
  }

  export interface WaitForOptions {
    timeout?: number;
    interval?: number;
  }

  export function render<S extends SvelteComponent>(
    component: new (options: any) => S,
    options?: { props?: Record<string, any> }
  ): RenderResult<S>;

  export const fireEvent: FireEvent;

  export function waitFor(
    callback: () => void | Promise<void>,
    options?: WaitForOptions
  ): Promise<void>;

  export function waitForElementToBeRemoved(
    callback: () => HTMLElement | HTMLElement[] | null,
    options?: WaitForOptions
  ): Promise<void>;

  export function act(callback: () => void): void;

  export function screen: {
    getByText: (text: string | RegExp) => HTMLElement;
    getByRole: (role: string, options?: any) => HTMLElement;
    getByLabelText: (text: string | RegExp) => HTMLElement;
    getByTestId: (testId: string) => HTMLElement;
    queryByText: (text: string | RegExp) => HTMLElement | null;
    queryByRole: (role: string, options?: any) => HTMLElement | null;
    findByText: (text: string | RegExp) => Promise<HTMLElement>;
    findByRole: (role: string, options?: any) => Promise<HTMLElement>;
  };
}
