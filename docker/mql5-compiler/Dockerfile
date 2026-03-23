# MQL5 Compiler Docker Image
# MetaTrader 5 MQL5 compiler running via Wine on Linux
#
# Build on Contabo where MT5 installer is available:
#   docker build -t mql5-compiler:latest .
#
# The MT5 installer is a web installer (connects to MetaQuotes CDN).
# During build we run it with Xvfb for headless GUI installation.

FROM debian:12-slim

LABEL maintainer="QUANTMINDX"
LABEL description="MetaTrader 5 MQL5 Compiler (Wine-based)"
LABEL org.opencontainers.image.source="https://github.com/quantmind/quantmindx"

# Install Wine (full suite) + Xvfb for running MT5 installer
# The 'wine' metapackage pulls in wine64 + tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    wine \
    xvfb \
    cabextract \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set up Wine prefix for MT5
ENV WINEPREFIX=/opt/wineprefix/mt5
ENV WINEDEBUG=-all
ENV DISPLAY=:99

# Copy compile entrypoint script
COPY compile.sh /opt/mt5-compiler/compile.sh
RUN chmod +x /opt/mt5-compiler/compile.sh

# Copy MT5 installer (assumes mt5setup.exe is in build context)
ARG MT5_INSTALLER=mt5setup.exe
COPY ${MT5_INSTALLER} /tmp/mt5setup.exe

# Pre-init Wine prefix (creates Wine internals)
RUN mkdir -p ${WINEPREFIX} && \
    wineboot -u || true

# Install MT5 with virtual display
# /S = silent install, /D = install directory
RUN echo "Installing MetaTrader 5 (web installer, may take several minutes)..." && \
    Xvfb :99 -screen 0 1024x768x24 & \
    XVFB_PID=\$! && \
    sleep 3 && \
    wine /tmp/mt5setup.exe /S /D=C:\\MT5 && \
    wait \$XVFB_PID || kill \$XVFB_PID 2>/dev/null; \
    rm -f /tmp/mt5setup.exe; \
    echo "MT5 installation complete."

# Verify mql5.exe exists
RUN find /opt/wineprefix -name "mql5.exe" 2>/dev/null | head -3 || \
    echo "WARNING: mql5.exe not found after installation"

# Working directory for volume mount (source .mq5 files)
WORKDIR /mnt/mql5/Experts

ENTRYPOINT ["/opt/mt5-compiler/compile.sh"]
