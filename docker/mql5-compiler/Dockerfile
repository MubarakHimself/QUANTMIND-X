# MQL5 Compiler Docker Image
# MetaTrader 5 MQL5 compiler running via Wine on Linux
#
# IMPORTANT: MetaTrader 5 is proprietary software by MetaQuotes.
#            You must accept the license agreement and download from:
#            https://www.metatrader5.com/en/download
#
# Build args:
#   MT5_INSTALLER   - Path to MT5 installer file within build context (default: mt5setup.exe)
#   WINE_PREFIX      - Wine prefix path for MT5 (default: /opt/wineprefix/mt5)
#
# Build example (on Contabo where installer is available):
#   docker build \
#     --build-arg MT5_INSTALLER=mt5setup.exe \
#     -t mql5-compiler:latest .
#
# Or use the build.sh script:
#   ./build.sh ghcr.io/quantmind/quantmindx/mql5-compiler:latest

FROM debian:12-slim

LABEL maintainer="QUANTMINDX"
LABEL description="MetaTrader 5 MQL5 Compiler (Wine-based)"
LABEL org.opencontainers.image.source="https://github.com/quantmind/quantmindx"

# Install Wine + dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wine \
    wine32 \
    libc6:i386 \
    libwine:i386 \
    fonts-wine \
    cabextract \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Wine prefix and MT5 installation paths
ARG WINE_PREFIX=/opt/wineprefix/mt5
ENV WINEPREFIX=${WINE_PREFIX}
ENV WINEDEBUG=-all
ENV MT5_DIR=/opt/mt5-compiler

RUN mkdir -p ${MT5_DIR} ${WINE_PREFIX}/drive_c/MetaTrader\ 5

# Copy compile entrypoint
COPY compile.sh /opt/mt5-compiler/compile.sh
RUN chmod +x /opt/mt5-compiler/compile.sh

# Default volume mount point (source .mq5 files mounted here)
WORKDIR /mnt/mql5/Experts

ENTRYPOINT ["/opt/mt5-compiler/compile.sh"]
