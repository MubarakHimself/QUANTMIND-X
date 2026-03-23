#!/usr/bin/env bash
# MQL5 Compiler Entry Point
# Runs inside mql5-compiler Docker container via Wine
# Usage: compile.sh <filename>
set -e

WINE_PREFIX="${WINEPREFIX:-/opt/wineprefix/mt5}"
MT5_DIR="${WINE_PREFIX}/drive_c/MetaTrader 5"
EXPERTS_DIR="/mnt/mql5/Experts"

# Find the MQL5 compiler executable
# MT5 installs the compiler as mql5.exe in the terminal folder
COMPILE_SCRIPT=""

for candidate in \
    "${MT5_DIR}/mql5.exe" \
    "${MT5_DIR}/terminal.exe" \
    "${WINE_PREFIX}/drive_c/mql5.exe"; do
    if [ -f "${candidate}" ]; then
        COMPILE_SCRIPT="${candidate}"
        break
    fi
done

if [ -z "$COMPILE_SCRIPT" ]; then
    echo "ERROR: MQL5 compiler (mql5.exe) not found in ${MT5_DIR}"
    echo "MT5 was not properly installed in the Docker image."
    echo "Please rebuild with a valid MT5 installer."
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
echo "Compiler: ${COMPILE_SCRIPT}"

cd "${EXPERTS_DIR}"

# Run MT5 MQL5 compiler via Wine
# MT5 compiler: mql5.exe <source.mq5>
# Outputs .ex5 to same directory as source
WINEDEBUG=-all wine "${COMPILE_SCRIPT}" "${MQ5_PATH}" 2>&1

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    EX5_PATH="${MQ5_PATH%.mq5}.ex5"
    if [ -f "${EX5_PATH}" ]; then
        echo "SUCCESS: Compiled to ${EX5_PATH}"
        echo "${EX5_PATH}"
        exit 0
    else
        echo "WARNING: Exit 0 but .ex5 not found"
        exit 0
    fi
else
    echo "ERROR: Compilation failed (exit ${EXIT_CODE})"
    exit $EXIT_CODE
fi
