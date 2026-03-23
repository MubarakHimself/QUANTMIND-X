#!/usr/bin/env bash
# Build MQL5 Compiler Docker image on Contabo
#
# IMPORTANT: Must be run on Contabo where MT5 installer is available.
#
# Prerequisites:
#   1. Download MetaTrader 5 from https://www.metatrader5.com/en/download
#   2. Place the installer as 'mt5setup.exe' in the same directory as this script
#   3. Run: ./build.sh
set -e

IMAGE_TAG="${1:-mql5-compiler:latest}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MT5_INSTALLER="${SCRIPT_DIR}/mt5setup.exe"

echo "=== MQL5 Compiler Docker Build ==="
echo "Image tag: ${IMAGE_TAG}"
echo "Script dir: ${SCRIPT_DIR}"

# Check for MT5 installer
if [ ! -f "${MT5_INSTALLER}" ]; then
    echo "ERROR: MT5 installer not found: ${MT5_INSTALLER}"
    echo ""
    echo "Please download MetaTrader 5 from:"
    echo "  https://www.metatrader5.com/en/download"
    echo ""
    echo "Place the installer as '${SCRIPT_DIR}/mt5setup.exe'"
    echo "Then run this script again."
    exit 1
fi

echo "MT5 installer: $(ls -lh "${MT5_INSTALLER}" | awk '{print $5, $9}')"

echo ""
echo "Building Docker image..."
echo "(This will take 10-20 minutes due to MT5 web installer)"

# Build with installer in context
docker build \
    --build-arg MT5_INSTALLER=mt5setup.exe \
    -t "${IMAGE_TAG}" \
    "${SCRIPT_DIR}"

echo ""
echo "=== Build complete ==="
echo "Image: ${IMAGE_TAG}"
echo ""
echo "To run locally for testing:"
echo "  docker run --rm -v /tmp/mql5:/mnt/mql5/Experts ${IMAGE_TAG} /opt/mt5-compiler/compile.sh MyEA.mq5"
echo ""
echo "To push to GHCR:"
echo "  echo \"\${GITHUB_TOKEN}\" | docker login ghcr.io -u \"\${GITHUB_ACTOR}\" --password-stdin"
echo "  docker push ${IMAGE_TAG}"
