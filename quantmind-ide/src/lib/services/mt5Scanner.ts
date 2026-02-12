/**
 * MT5 Scanner Service
 * Detects MT5 installation and provides launching capabilities
 * for Windows, Linux (Wine), and macOS platforms
 */

export interface MT5Status {
  found: boolean;
  path?: string;
  version?: string;
  platform: 'windows' | 'linux' | 'macos' | 'unknown';
}

export interface MT5Config {
  terminalPath?: string;
  login?: number;
  password?: string;
  server?: string;
}

export class MT5Scanner {
  private platform: 'windows' | 'linux' | 'macos' | 'unknown';
  private commonPaths: string[] = [];

  constructor() {
    this.detectPlatform();
    this.setCommonPaths();
  }

  /**
   * Detect the current platform
   */
  private detectPlatform(): void {
    if (typeof window === 'undefined') {
      // Server-side detection (Node.js)
      const platform = process?.platform || '';
      if (platform.includes('win32')) {
        this.platform = 'windows';
      } else if (platform.includes('linux')) {
        this.platform = 'linux';
      } else if (platform.includes('darwin')) {
        this.platform = 'macos';
      } else {
        this.platform = 'unknown';
      }
    } else {
      // Browser-based detection
      const userAgent = navigator.userAgent.toLowerCase();
      if (userAgent.includes('win')) {
        this.platform = 'windows';
      } else if (userAgent.includes('linux')) {
        this.platform = 'linux';
      } else if (userAgent.includes('mac')) {
        this.platform = 'macos';
      } else {
        this.platform = 'unknown';
      }
    }
  }

  /**
   * Set common MT5 installation paths based on platform
   */
  private setCommonPaths(): void {
    switch (this.platform) {
      case 'windows':
        this.commonPaths = [
          'C:\\Program Files\\MetaTrader 5\\terminal64.exe',
          'C:\\Program Files (x86)\\MetaTrader 5\\terminal.exe',
          'C:\\Users\\{USER}\\AppData\\Roaming\\MetaQuotes\\Terminal\\{TERMINAL_ID}\\terminal64.exe',
        ];
        break;
      case 'linux':
        this.commonPaths = [
          '~/.wine/drive_c/Program Files/MetaTrader 5/terminal.exe',
          '/opt/metatrader5/terminal.exe',
        ];
        break;
      case 'macos':
        this.commonPaths = [
          '/Applications/MetaTrader 5.app/Contents/MacOS/terminal',
          '/Applications/MetaTrader 5 Terminal.app/Contents/MacOS/terminal',
        ];
        break;
      default:
        this.commonPaths = [];
    }
  }

  /**
   * Scan for MT5 installation
   * Calls backend API to scan filesystem for MT5 installation
   */
  async scanForMT5(): Promise<MT5Status> {
    try {
      // Call backend API endpoint for MT5 scanning
      const response = await fetch('http://localhost:8000/api/mt5/scan', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          custom_paths: this.commonPaths.length > 0 ? this.commonPaths : undefined,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      // Return the scan results from backend
      return {
        found: data.found,
        path: data.installations && data.installations.length > 0
          ? data.installations[0].path
          : undefined,
        version: undefined, // Backend doesn't detect version yet
        platform: data.platform as 'windows' | 'linux' | 'macos' | 'unknown',
      };
    } catch (error) {
      console.error('Failed to scan for MT5:', error);
      // Fallback to web-compatible status
      return {
        found: false,
        platform: this.platform,
      };
    }
  }

  /**
   * Launch MT5 Web Terminal
   */
  async launchMT5Web(): Promise<void> {
    if (typeof window !== 'undefined') {
      window.open('https://web.metaquotes.net/en/terminal', '_blank');
    }
  }

  /**
   * Launch MT5 Desktop application
   * Calls backend API to launch MT5 with configuration
   */
  async launchMT5Desktop(config: MT5Config): Promise<boolean> {
    try {
      // First, scan for MT5 installation if no path provided
      let terminalPath = config.terminalPath;

      if (!terminalPath) {
        const scanResult = await this.scanForMT5();
        if (!scanResult.found || !scanResult.path) {
          console.error('MT5 not found. Please install MetaTrader 5.');
          return false;
        }
        terminalPath = scanResult.path;
      }

      // Call backend API endpoint for MT5 launching
      const response = await fetch('http://localhost:8000/api/mt5/launch', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          terminal_path: terminalPath,
          login: config.login,
          password: config.password,
          server: config.server,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      if (data.success) {
        console.log('MT5 launched successfully:', data);
        return true;
      } else {
        console.error('Failed to launch MT5:', data.error);
        return false;
      }
    } catch (error) {
      console.error('Failed to launch MT5:', error);
      return false;
    }
  }

  /**
   * Get detected platform
   */
  getPlatform(): string {
    return this.platform;
  }

  /**
   * Get common MT5 installation paths
   */
  getCommonPaths(): string[] {
    return [...this.commonPaths];
  }

  /**
   * Get MT5 Web Terminal URL
   */
  getWebTerminalURL(): string {
    return 'https://web.metaquotes.net/en/terminal';
  }

  /**
   * Check if platform is supported for MT5
   */
  isPlatformSupported(): boolean {
    return ['windows', 'linux', 'macos'].includes(this.platform);
  }

  /**
   * Get platform-specific recommendations
   */
  getPlatformRecommendations(): string {
    switch (this.platform) {
      case 'windows':
        return 'MT5 Desktop is fully supported. You can connect directly or use Web Terminal.';
      case 'linux':
        return 'MT5 can run via Wine. Consider using Web Terminal for easier setup.';
      case 'macos':
        return 'MT5 Desktop is supported. Web Terminal is also available.';
      default:
        return 'Platform not detected. Web Terminal is recommended.';
    }
  }
}

// Export singleton instance
export const mt5Scanner = new MT5Scanner();
