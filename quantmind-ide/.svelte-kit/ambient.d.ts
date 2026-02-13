
// this file is generated — do not edit it


/// <reference types="@sveltejs/kit" />

/**
 * Environment variables [loaded by Vite](https://vitejs.dev/guide/env-and-mode.html#env-files) from `.env` files and `process.env`. Like [`$env/dynamic/private`](https://svelte.dev/docs/kit/$env-dynamic-private), this module cannot be imported into client-side code. This module only includes variables that _do not_ begin with [`config.kit.env.publicPrefix`](https://svelte.dev/docs/kit/configuration#env) _and do_ start with [`config.kit.env.privatePrefix`](https://svelte.dev/docs/kit/configuration#env) (if configured).
 * 
 * _Unlike_ [`$env/dynamic/private`](https://svelte.dev/docs/kit/$env-dynamic-private), the values exported from this module are statically injected into your bundle at build time, enabling optimisations like dead code elimination.
 * 
 * ```ts
 * import { API_KEY } from '$env/static/private';
 * ```
 * 
 * Note that all environment variables referenced in your code should be declared (for example in an `.env` file), even if they don't have a value until the app is deployed:
 * 
 * ```
 * MY_FEATURE_FLAG=""
 * ```
 * 
 * You can override `.env` values from the command line like so:
 * 
 * ```sh
 * MY_FEATURE_FLAG="enabled" npm run dev
 * ```
 */
declare module '$env/static/private' {
	export const SHELL: string;
	export const npm_command: string;
	export const __EGL_EXTERNAL_PLATFORM_CONFIG_DIRS: string;
	export const QT_ACCESSIBILITY: string;
	export const npm_config_userconfig: string;
	export const COLORTERM: string;
	export const XDG_CONFIG_DIRS: string;
	export const VSCODE_DEBUGPY_ADAPTER_ENDPOINTS: string;
	export const npm_config_cache: string;
	export const NVM_INC: string;
	export const TERM_PROGRAM_VERSION: string;
	export const GTK_IM_MODULE: string;
	export const NODE: string;
	export const SSH_AUTH_SOCK: string;
	export const XDG_DATA_HOME: string;
	export const DRI_PRIME: string;
	export const XDG_CONFIG_HOME: string;
	export const XCURSOR_PATH: string;
	export const PYDEVD_DISABLE_FILE_VALIDATION: string;
	export const COLOR: string;
	export const npm_config_local_prefix: string;
	export const XMODIFIERS: string;
	export const ZYPAK_LIB: string;
	export const FLATPAK_ID: string;
	export const npm_config_globalconfig: string;
	export const EDITOR: string;
	export const GTK_MODULES: string;
	export const XDG_SEAT: string;
	export const PWD: string;
	export const ALSA_CONFIG_PATH: string;
	export const LOGNAME: string;
	export const XDG_SESSION_DESKTOP: string;
	export const XDG_SESSION_TYPE: string;
	export const npm_config_init_module: string;
	export const npm_config_tmp: string;
	export const COSMIC_DATA_CONTROL_ENABLED: string;
	export const _: string;
	export const BUNDLED_DEBUGPY_PATH: string;
	export const VSCODE_GIT_ASKPASS_NODE: string;
	export const container: string;
	export const GI_TYPELIB_PATH: string;
	export const HOME: string;
	export const IM_CONFIG_PHASE: string;
	export const LANG: string;
	export const GITHUB_TOKEN: string;
	export const _JAVA_AWT_WM_NONREPARENTING: string;
	export const LS_COLORS: string;
	export const XDG_CURRENT_DESKTOP: string;
	export const npm_package_version: string;
	export const VIRTUAL_ENV: string;
	export const PYTHONSTARTUP: string;
	export const WAYLAND_DISPLAY: string;
	export const SBX_CHROME_API_RQ: string;
	export const AT_SPI_BUS_ADDRESS: string;
	export const GIT_ASKPASS: string;
	export const PULSE_CLIENTCONFIG: string;
	export const DCONF_PROFILE: string;
	export const INIT_CWD: string;
	export const CHROME_DESKTOP: string;
	export const CLUTTER_IM_MODULE: string;
	export const QT_QPA_PLATFORM: string;
	export const XDG_CACHE_HOME: string;
	export const npm_lifecycle_script: string;
	export const NVM_DIR: string;
	export const VSCODE_GIT_ASKPASS_EXTRA_ARGS: string;
	export const VSCODE_PYTHON_AUTOACTIVATE_GUARD: string;
	export const CLAUDE_CODE_SSE_PORT: string;
	export const npm_config_npm_version: string;
	export const TERM: string;
	export const npm_package_name: string;
	export const PYTHON_BASIC_REPL: string;
	export const npm_config_prefix: string;
	export const ZYPAK_ZYGOTE_STRATEGY_SPAWN: string;
	export const LIBVIRT_DEFAULT_URI: string;
	export const USER: string;
	export const SANDBOX_LD_LIBRARY_PATH: string;
	export const VSCODE_GIT_IPC_HANDLE: string;
	export const NPM_CONFIG_GLOBALCONFIG: string;
	export const npm_lifecycle_event: string;
	export const SHLVL: string;
	export const NVM_CD_FLAGS: string;
	export const MOZ_ENABLE_WAYLAND: string;
	export const GSM_SKIP_SSH_AGENT_WORKAROUND: string;
	export const FLATPAK_SANDBOX_DIR: string;
	export const QT_IM_MODULE: string;
	export const XDG_VTNR: string;
	export const CLINE_ACTIVE: string;
	export const XDG_SESSION_ID: string;
	export const VIRTUAL_ENV_PROMPT: string;
	export const npm_config_user_agent: string;
	export const ZYPAK_BIN: string;
	export const XDG_STATE_HOME: string;
	export const npm_execpath: string;
	export const FC_FONTATIONS: string;
	export const LD_LIBRARY_PATH: string;
	export const XDG_RUNTIME_DIR: string;
	export const DBUS_SYSTEM_BUS_ADDRESS: string;
	export const npm_package_json: string;
	export const GST_PLUGIN_SYSTEM_PATH: string;
	export const BUN_INSTALL: string;
	export const VSCODE_GIT_ASKPASS_MAIN: string;
	export const QT_AUTO_SCREEN_SCALE_FACTOR: string;
	export const XDG_DATA_DIRS: string;
	export const GDK_BACKEND: string;
	export const npm_config_noproxy: string;
	export const PATH: string;
	export const npm_config_node_gyp: string;
	export const QT_ENABLE_HIGHDPI_SCALING: string;
	export const PYTHONUSERBASE: string;
	export const DBUS_SESSION_BUS_ADDRESS: string;
	export const npm_config_global_prefix: string;
	export const ALSA_CONFIG_DIR: string;
	export const NVM_BIN: string;
	export const PULSE_SERVER: string;
	export const npm_node_execpath: string;
	export const OLDPWD: string;
	export const TERM_PROGRAM: string;
	export const NODE_ENV: string;
}

/**
 * Similar to [`$env/static/private`](https://svelte.dev/docs/kit/$env-static-private), except that it only includes environment variables that begin with [`config.kit.env.publicPrefix`](https://svelte.dev/docs/kit/configuration#env) (which defaults to `PUBLIC_`), and can therefore safely be exposed to client-side code.
 * 
 * Values are replaced statically at build time.
 * 
 * ```ts
 * import { PUBLIC_BASE_URL } from '$env/static/public';
 * ```
 */
declare module '$env/static/public' {
	
}

/**
 * This module provides access to runtime environment variables, as defined by the platform you're running on. For example if you're using [`adapter-node`](https://github.com/sveltejs/kit/tree/main/packages/adapter-node) (or running [`vite preview`](https://svelte.dev/docs/kit/cli)), this is equivalent to `process.env`. This module only includes variables that _do not_ begin with [`config.kit.env.publicPrefix`](https://svelte.dev/docs/kit/configuration#env) _and do_ start with [`config.kit.env.privatePrefix`](https://svelte.dev/docs/kit/configuration#env) (if configured).
 * 
 * This module cannot be imported into client-side code.
 * 
 * ```ts
 * import { env } from '$env/dynamic/private';
 * console.log(env.DEPLOYMENT_SPECIFIC_VARIABLE);
 * ```
 * 
 * > [!NOTE] In `dev`, `$env/dynamic` always includes environment variables from `.env`. In `prod`, this behavior will depend on your adapter.
 */
declare module '$env/dynamic/private' {
	export const env: {
		SHELL: string;
		npm_command: string;
		__EGL_EXTERNAL_PLATFORM_CONFIG_DIRS: string;
		QT_ACCESSIBILITY: string;
		npm_config_userconfig: string;
		COLORTERM: string;
		XDG_CONFIG_DIRS: string;
		VSCODE_DEBUGPY_ADAPTER_ENDPOINTS: string;
		npm_config_cache: string;
		NVM_INC: string;
		TERM_PROGRAM_VERSION: string;
		GTK_IM_MODULE: string;
		NODE: string;
		SSH_AUTH_SOCK: string;
		XDG_DATA_HOME: string;
		DRI_PRIME: string;
		XDG_CONFIG_HOME: string;
		XCURSOR_PATH: string;
		PYDEVD_DISABLE_FILE_VALIDATION: string;
		COLOR: string;
		npm_config_local_prefix: string;
		XMODIFIERS: string;
		ZYPAK_LIB: string;
		FLATPAK_ID: string;
		npm_config_globalconfig: string;
		EDITOR: string;
		GTK_MODULES: string;
		XDG_SEAT: string;
		PWD: string;
		ALSA_CONFIG_PATH: string;
		LOGNAME: string;
		XDG_SESSION_DESKTOP: string;
		XDG_SESSION_TYPE: string;
		npm_config_init_module: string;
		npm_config_tmp: string;
		COSMIC_DATA_CONTROL_ENABLED: string;
		_: string;
		BUNDLED_DEBUGPY_PATH: string;
		VSCODE_GIT_ASKPASS_NODE: string;
		container: string;
		GI_TYPELIB_PATH: string;
		HOME: string;
		IM_CONFIG_PHASE: string;
		LANG: string;
		GITHUB_TOKEN: string;
		_JAVA_AWT_WM_NONREPARENTING: string;
		LS_COLORS: string;
		XDG_CURRENT_DESKTOP: string;
		npm_package_version: string;
		VIRTUAL_ENV: string;
		PYTHONSTARTUP: string;
		WAYLAND_DISPLAY: string;
		SBX_CHROME_API_RQ: string;
		AT_SPI_BUS_ADDRESS: string;
		GIT_ASKPASS: string;
		PULSE_CLIENTCONFIG: string;
		DCONF_PROFILE: string;
		INIT_CWD: string;
		CHROME_DESKTOP: string;
		CLUTTER_IM_MODULE: string;
		QT_QPA_PLATFORM: string;
		XDG_CACHE_HOME: string;
		npm_lifecycle_script: string;
		NVM_DIR: string;
		VSCODE_GIT_ASKPASS_EXTRA_ARGS: string;
		VSCODE_PYTHON_AUTOACTIVATE_GUARD: string;
		CLAUDE_CODE_SSE_PORT: string;
		npm_config_npm_version: string;
		TERM: string;
		npm_package_name: string;
		PYTHON_BASIC_REPL: string;
		npm_config_prefix: string;
		ZYPAK_ZYGOTE_STRATEGY_SPAWN: string;
		LIBVIRT_DEFAULT_URI: string;
		USER: string;
		SANDBOX_LD_LIBRARY_PATH: string;
		VSCODE_GIT_IPC_HANDLE: string;
		NPM_CONFIG_GLOBALCONFIG: string;
		npm_lifecycle_event: string;
		SHLVL: string;
		NVM_CD_FLAGS: string;
		MOZ_ENABLE_WAYLAND: string;
		GSM_SKIP_SSH_AGENT_WORKAROUND: string;
		FLATPAK_SANDBOX_DIR: string;
		QT_IM_MODULE: string;
		XDG_VTNR: string;
		CLINE_ACTIVE: string;
		XDG_SESSION_ID: string;
		VIRTUAL_ENV_PROMPT: string;
		npm_config_user_agent: string;
		ZYPAK_BIN: string;
		XDG_STATE_HOME: string;
		npm_execpath: string;
		FC_FONTATIONS: string;
		LD_LIBRARY_PATH: string;
		XDG_RUNTIME_DIR: string;
		DBUS_SYSTEM_BUS_ADDRESS: string;
		npm_package_json: string;
		GST_PLUGIN_SYSTEM_PATH: string;
		BUN_INSTALL: string;
		VSCODE_GIT_ASKPASS_MAIN: string;
		QT_AUTO_SCREEN_SCALE_FACTOR: string;
		XDG_DATA_DIRS: string;
		GDK_BACKEND: string;
		npm_config_noproxy: string;
		PATH: string;
		npm_config_node_gyp: string;
		QT_ENABLE_HIGHDPI_SCALING: string;
		PYTHONUSERBASE: string;
		DBUS_SESSION_BUS_ADDRESS: string;
		npm_config_global_prefix: string;
		ALSA_CONFIG_DIR: string;
		NVM_BIN: string;
		PULSE_SERVER: string;
		npm_node_execpath: string;
		OLDPWD: string;
		TERM_PROGRAM: string;
		NODE_ENV: string;
		[key: `PUBLIC_${string}`]: undefined;
		[key: `${string}`]: string | undefined;
	}
}

/**
 * Similar to [`$env/dynamic/private`](https://svelte.dev/docs/kit/$env-dynamic-private), but only includes variables that begin with [`config.kit.env.publicPrefix`](https://svelte.dev/docs/kit/configuration#env) (which defaults to `PUBLIC_`), and can therefore safely be exposed to client-side code.
 * 
 * Note that public dynamic environment variables must all be sent from the server to the client, causing larger network requests — when possible, use `$env/static/public` instead.
 * 
 * ```ts
 * import { env } from '$env/dynamic/public';
 * console.log(env.PUBLIC_DEPLOYMENT_SPECIFIC_VARIABLE);
 * ```
 */
declare module '$env/dynamic/public' {
	export const env: {
		[key: `PUBLIC_${string}`]: string | undefined;
	}
}
