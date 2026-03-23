#!/usr/bin/env bash
# Build MQL5 Compiler Docker image on Contabo
#
# IMPORTANT: This must be run on Contabo where the MT5 installer is stored.
# The installer is proprietary and can't be stored in the GitHub repo.
#
# Usage:
#   ./build.sh [image_tag]
#
# Prerequisites on Contabo:
#   1. Download MetaTrader 5 from https://www.metatrader5.com/en/download
#   2. Place the installer (mt5setup.exe) in the same directory as this script
set -e

IMAGE_TAG="${1:-mql5-compiler:latest}"
MT5_INSTALLER="${MT5_INSTALLER:-mt5setup.exe}"
DOCKERFILE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== MQL5 Compiler Docker Build ==="
echo "Image tag: ${IMAGE_TAG}"
echo "MT5 installer: ${MT5_INSTALLER}"
echo "Dockerfile dir: ${DOCKERFILE_DIR}"

# Check for MT5 installer
if [ ! -f "${DOCKERFILE_DIR}/${MT5_INSTALLER}" ]; then
    echo "ERROR: MT5 installer not found: ${DOCKERFILE_DIR}/${MT5_INSTALLER}"
    echo ""
    echo "Please download MetaTrader 5 from:"
    echo "  https://www.metatrader5.com/en/download"
    echo ""
    echo "Place the installer as '${MT5_INSTALLER}' in ${DOCKERFILE_DIR}/"
    echo "Then run this script again."
    exit 1
fi

# Check Docker is available
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed or not in PATH"
    exit 1
fi

echo ""
echo "Building Docker image with MT5 installer..."
docker build \
    --build-arg MT5_INSTALLER="${MT5_INSTALLER}" \
    -t "${IMAGE_TAG}" \
    "${DOCKERFILE_DIR}"

echo ""
echo "=== Build complete ==="
echo "Image: ${IMAGE_TAG}"
echo ""
echo "To run locally for testing:"
echo "  docker run --rm -v /tmp/mql5:/mnt/mql5/Experts ${IMAGE_TAG} /opt/mt5-compiler/compile.sh MyEA.mq5"
echo ""
echo "To push to GHCR:"
echo "  echo \"\$GITHUB_TOKEN\" | docker login ghcr.io -u \"\$GITHUB_ACTOR\" --password-stdin"
echo "  docker push ${IMAGE_TAG}"
