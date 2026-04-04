#!/usr/bin/env bash
# MT5 Compiler Setup Script — run on Contabo
set -e
export DISPLAY=:99
export WINEDEBUG=-all
export DEBIAN_FRONTEND=noninteractive

MT5_DIR="/opt/mt5-compiler"
WINE_PREFIX="/opt/wine64"
LOG="/tmp/mt5-setup.log"

exec > >(tee "$LOG") 2>&1

echo "=== $(date) ==="
echo "MT5 Compiler Setup Starting"

# Kill stale processes
pkill -9 wine 2>/dev/null; pkill -9 Xvfb 2>/dev/null; pkill -9 wineserver 2>/dev/null; true
sleep 2

# Start Xvfb
echo "Starting Xvfb..."
Xvfb :99 -screen 0 1024x768x24 &
sleep 3

# Set up clean wine prefix for 64-bit Windows
echo "Setting up Wine prefix..."
rm -rf "$WINE_PREFIX"
mkdir -p "$WINE_PREFIX"
WINEPREFIX="$WINE_PREFIX" WINEARCH=win64 wineboot -u 2>&1 | tail -3
sleep 5

# Copy installer to wine prefix drive
echo "Copying MT5 installer..."
cp "$MT5_DIR/mt5setup.exe" "$WINE_PREFIX/drive_c/mt5setup.exe"

# Run the MT5 installer silently
echo "Running MT5 installer..."
cd "$WINE_PREFIX/drive_c"
WINEPREFIX="$WINE_PREFIX" WINEARCH=win64 WINEDEBUG=-all timeout 180 wine mt5setup.exe /S /D=C:\\MT5 2>&1 | tail -20
INSTALLER_EXIT=$?

echo "Installer exit code: $INSTALLER_EXIT"
sleep 5

# Check what was installed
echo "Checking installation..."
ls -la "$WINE_PREFIX/drive_c/" | head -20
ls -la "$WINE_PREFIX/drive_c/MT5" 2>/dev/null && echo "MT5 installed!" || echo "MT5 dir not found"
find "$WINE_PREFIX/drive_c" -name "mql5.exe" 2>/dev/null
find "$WINE_PREFIX/drive_c/MT5" -name "*.exe" 2>/dev/null | head -10

# If MT5 installed, copy compiler files to /opt/mt5-compiler
if [ -d "$WINE_PREFIX/drive_c/MT5" ]; then
    echo "Copying MT5 compiler files..."
    cp -r "$WINE_PREFIX/drive_c/MT5" /opt/
    ls /opt/MT5/
    echo "=== SUCCESS ==="
else
    echo "=== FAILED - MT5 not installed ==="
    ls "$WINE_PREFIX/drive_c/"
fi
