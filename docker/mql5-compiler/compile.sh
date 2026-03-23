#!/usr/bin/env bash
# MQL5 Compiler Entry Point
# Runs inside mql5-compiler Docker container via Wine
# Usage: compile.sh <filename>
set -e

WINE_PREFIX="${WINEPREFIX:-/opt/wineprefix/mt5}"

# The MT5 installer installs to C:\MT5 inside the Wine prefix
MT5_DIR="${WINE_PREFIX}/drive_c/MT5"
EXPERTS_DIR="/mnt/mql5/Experts"

# Find the MQL5 compiler executable
COMPILE_BIN=""

for candidate in \
    "${MT5_DIR}/MQL5/mql5.exe" \
    "${MT5_DIR}/mql5.exe" \
    "${MT5_DIR}/metaeditor64.exe" \
    "${WINE_PREFIX}/drive_c/MQL5/mql5.exe"; do
    if [ -f "${candidate}" ]; then
        COMPILE_BIN="${candidate}"
        break
    fi
done

if [ -z "$COMPILE_BIN" ]; then
    echo "ERROR: MQL5 compiler (mql5.exe) not found"
    echo "Searched in: ${MT5_DIR}"
    find "${WINE_PREFIX}" -name "mql5.exe" 2>/dev/null | head -5 || true
    find "${WINE_PREFIX}" -name "*.exe" 2>/dev/null | grep -i mql | head -5 || true
    exit 1
fi

if [ -z "$1" ]; then
    echo "ERROR: No filename provided"
    echo "Usage: compile.sh <filename>"
    exit 1
fi

FILENAME="$1"
MQ5_PATH="${EXPERTS_DIR}/${FILENAME}"

if [ ! -f "${MQ5_PATH}" ]; then
    echo "ERROR: File not found: ${MQ5_PATH}"
    exit 1
fi

echo "Compiling ${FILENAME}..."
echo "Compiler: ${COMPILE_BIN}"
echo "Source:   ${MQ5_PATH}"

cd "${EXPERTS_DIR}"

# Run MT5 MQL5 compiler via Wine
# MT5 compiler: mql5.exe <source.mq5>
# Outputs .ex5 to same directory as source
WINEDEBUG=-all wine "${COMPILE_BIN}" "${MQ5_PATH}" 2>&1

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    EX5_PATH="${MQ5_PATH%.mq5}.ex5"
    if [ -f "${EX5_PATH}" ]; then
        echo "SUCCESS: Compiled to ${EX5_PATH}"
        echo "${EX5_PATH}"
        exit 0
    else
        # Check if ex5 was written elsewhere
        FOUND_EX5=$(find "${EXPERTS_DIR}" -name "*.ex5" -newer "${MQ5_PATH}" 2>/dev/null | head -1)
        if [ -n "${FOUND_EX5}" ]; then
            echo "SUCCESS: Found compiled file ${FOUND_EX5}"
            echo "${FOUND_EX5}"
            exit 0
        fi
        echo "WARNING: Exit 0 but .ex5 not found"
        exit 0
    fi
else
    echo "ERROR: Compilation failed (exit ${EXIT_CODE})"
    exit $EXIT_CODE
fi
