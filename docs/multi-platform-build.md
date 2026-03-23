# Multi-Platform Build Guide

FR72: cross-platform build - The ITT compiles and launches on Linux, Windows, and macOS

## Overview

This document describes the build process for QUANTMINDX on Linux, Windows, and macOS platforms. The project uses Tauri for the desktop application wrapper.

## Prerequisites

### Common Requirements
- Node.js 18+ (LTS recommended)
- npm 9+
- Git

### Platform-Specific Requirements

#### Linux (Ubuntu/Debian)
```bash
# Install build dependencies
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    curl \
    wget \
    file \
    libssl-dev \
    libwebkit2gtk-4.0-dev \
    libappindicator3-1 \
    librsvg2-dev

# Install Rust (required for Tauri)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
```

#### macOS
```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install create-dmg
brew install rust

# Note: Xcode command line tools required
xcode-select --install
```

#### Windows
```bash
# Install via Chocolatey
choco install visualstudio2022buildtools -y
choco install rust -y

# Or install manually:
# - Visual Studio Build Tools 2022
# - Rust toolchain from rustup.rs
```

## Build Process

### 1. Checkout Source

```bash
git clone https://github.com/your-repo/quantmindx.git
cd quantmindx
```

### 2. Install Dependencies

```bash
npm install
```

### 3. Verify Development Mode

```bash
# Test frontend build
npm run build

# Test Tauri development
npm run tauri dev
```

### 4. Production Build

```bash
# Build for current platform
npm run tauri build

# The executable will be in:
# src-tauri/target/release/bundle/
```

## Build Verification Checklist

### Pre-Build Checklist
- [ ] All dependencies installed
- [ ] No TypeScript compilation errors
- [ ] No lint errors
- [ ] Tests passing (`npm test`)

### Build Verification

| Check | Linux | Windows | macOS |
|-------|-------|---------|-------|
| Frontend compiles | [ ] | [ ] | [ ] |
| Tauri compiles | [ ] | [ ] | [ ] |
| Executable runs | [ ] | [ ] | [ ] |
| Window displays | [ ] | [ ] | [ ] |
| No crash on startup | [ ] | [ ] | [ ] |
| API server starts | [ ] | [ ] | [ ] |

### Post-Build Verification
- [ ] Executable in `src-tauri/target/release/`
- [ ] App launches without errors
- [ ] UI renders correctly
- [ ] Backend services functional

## Troubleshooting

### Linux
- **WebKit not found**: Install `libwebkit2gtk-4.0-dev`
- **Permission denied**: Ensure execution permissions on build scripts
- **Rust not found**: Add to PATH: `source ~/.cargo/env`

### Windows
- **MSVC not found**: Install Visual Studio Build Tools
- **Path too long**: Enable long paths in Windows registry
- **Antivirus blocking**: Add exceptions for build directories

### macOS
- **Code signing errors**: Set up Apple Developer certificate
- ** notarization failed**: Run `npm run tauri build -- --ci` for CI builds

## CI/CD Integration

See `.github/workflows/multi-platform-build.yml` for automated build verification.

## Platform-Specific Notes

### Linux
- Primary deployment target (Crontab server)
- GTK-based webview
- Most testing done on this platform

### Windows
- Secondary target for desktop users
- WebView2 (Edge-based)
- Some features may behave differently

### macOS
- Development platform
- WebKit-based webview
- Retina display support

## Testing Across Platforms

For comprehensive testing:
1. Use CI/CD to build on all platforms
2. Test actual executables, not just source
3. Include UI screenshot verification
4. Test with various screen sizes
5. Verify file system operations

---

**Related Stories:**
- Story 11.4: ITT Rebuild Portability (backup/restore)
- Story 11.1: Nightly Rsync Cron (deployment automation)