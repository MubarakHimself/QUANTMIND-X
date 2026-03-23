#!/bin/bash
# OWASP ZAP Wrapper Script for Local Development
# Usage: ./scripts/zap_scan_wrapper.sh [target_url]
# Example: ./scripts/zap_scan_wrapper.sh http://localhost:8000

set -e

TARGET_URL="${1:-http://localhost:8000}"
REPORT_DIR="${ZAP_REPORT_DIR:-reports/zap}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "============================================"
echo "OWASP ZAP Security Scan"
echo "============================================"
echo "Target: $TARGET_URL"
echo "Report directory: $REPORT_DIR"
echo "Timestamp: $TIMESTAMP"
echo "============================================"

# Create report directory
mkdir -p "$REPORT_DIR"

# Check if ZAP is installed
if ! command -v zap.sh &> /dev/null && ! command -v zap-baseline.py &> /dev/null; then
    echo "ZAP not found. Installing..."

    # Option 1: Docker (recommended)
    if command -v docker &> /dev/null; then
        echo "Using Docker to run ZAP..."
        docker run -v "$(pwd)/${REPORT_DIR}:/zap/wrk:rw" \
            -t owasp/zap2docker-stable \
            zap-baseline.py \
            -t "$TARGET_URL" \
            -J "baseline_report_${TIMESTAMP}.json" \
            -r "baseline_report_${TIMESTAMP}.html" \
            -m 1 \
            -T 60 \
            -j
    else
        echo "Error: Docker not found. Please install ZAP or Docker."
        echo ""
        echo "To install ZAP manually:"
        echo "  1. Download from https://github.com/zaproxy/zaproxy/releases"
        echo "  2. Extract and run: ./zap.sh -baseline -t $TARGET_URL"
        echo ""
        echo "Or install Docker and run this script again."
        exit 1
    fi
else
    # Use installed ZAP
    echo "Using installed ZAP..."

    if command -v zap.sh &> /dev/null; then
        zap.sh -baseline -t "$TARGET_URL" \
            -J "${REPORT_DIR}/baseline_report_${TIMESTAMP}.json" \
            -r "${REPORT_DIR}/baseline_report_${TIMESTAMP}.html" \
            -m 1 \
            -T 60 \
            -j
    else
        zap-baseline.py -t "$TARGET_URL" \
            -J "${REPORT_DIR}/baseline_report_${TIMESTAMP}.json" \
            -r "${REPORT_DIR}/baseline_report_${TIMESTAMP}.html" \
            -m 1 \
            -T 60 \
            -j
    fi
fi

echo ""
echo "============================================"
echo "Scan Complete!"
echo "============================================"
echo ""
echo "Reports saved to:"
echo "  - JSON: ${REPORT_DIR}/baseline_report_${TIMESTAMP}.json"
echo "  - HTML: ${REPORT_DIR}/baseline_report_${TIMESTAMP}.html"
echo ""

# Parse and display summary
if [ -f "${REPORT_DIR}/baseline_report_${TIMESTAMP}.json" ]; then
    echo "Vulnerability Summary:"
    echo "----------------------"

    # Count by severity (if jq is available)
    if command -v jq &> /dev/null; then
        CRITICAL=$(jq '[.site[].alerts[] | select(.riskdesc | contains("Critical"))] | length' "${REPORT_DIR}/baseline_report_${TIMESTAMP}.json" 2>/dev/null || echo "0")
        HIGH=$(jq '[.site[].alerts[] | select(.riskdesc | contains("High"))] | length' "${REPORT_DIR}/baseline_report_${TIMESTAMP}.json" 2>/dev/null || echo "0")
        MEDIUM=$(jq '[.site[].alerts[] | select(.riskdesc | contains("Medium"))] | length' "${REPORT_DIR}/baseline_report_${TIMESTAMP}.json" 2>/dev/null || echo "0")
        LOW=$(jq '[.site[].alerts[] | select(.riskdesc | contains("Low"))] | length' "${REPORT_DIR}/baseline_report_${TIMESTAMP}.json" 2>/dev/null || echo "0")

        echo "  Critical: $CRITICAL"
        echo "  High:     $HIGH"
        echo "  Medium:   $MEDIUM"
        echo "  Low:      $LOW"
    else
        echo "  (Install jq for detailed summary: sudo apt install jq)"
    fi
fi

echo ""
echo "Next steps:"
echo "  1. Review HTML report for details"
echo "  2. If Critical/High vulnerabilities found:"
echo "     - Create GitHub issue"
echo "     - Create risk waiver if accepted"
echo "  3. Link report to: _bmad-output/test-artifacts/security-scan-reports/"
echo ""

# Exit with appropriate code
if [ "$CRITICAL" -gt 0 ] || [ "$HIGH" -gt 0 ]; then
    echo "WARNING: Critical or High vulnerabilities found!"
    exit 1
fi

exit 0
