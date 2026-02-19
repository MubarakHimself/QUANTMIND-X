#!/bin/bash
# PineScript Validate Tool
# Validates Pine Script syntax and structure
#
# Usage:
#   echo '{"code": "//@version=5\nstrategy(\"test\")"}' | ./pinescript_validate.sh
#   ./pinescript_validate.sh "/path/to/script.pine"
#
# Output: JSON with validation result
# Exit codes: 0 = success, 1 = error

set -euo pipefail

# Function to output JSON error
error_json() {
    echo "{\"error\": \"$1\", \"valid\": false, \"errors\": [], \"warnings\": []}"
    exit 1
}

# Function to validate Pine Script
validate_script() {
    local code="$1"
    local errors=()
    local warnings=()
    
    # Check for version declaration
    if ! echo "$code" | grep -q "//@version="; then
        errors+=("Missing version declaration. Add //@version=5 at the top")
    fi
    
    # Check for strategy or indicator declaration
    if ! echo "$code" | grep -qE "(strategy|indicator)\("; then
        errors+=("Missing strategy() or indicator() declaration")
    fi
    
    # Check for common syntax errors
    # Unmatched parentheses (basic check)
    local open_parens=$(echo "$code" | grep -o "(" | wc -l)
    local close_parens=$(echo "$code" | grep -o ")" | wc -l)
    if [ "$open_parens" -ne "$close_parens" ]; then
        errors+=("Unmatched parentheses detected")
    fi
    
    # Check for undefined variables (basic check)
    # This is a simplified check - real validation would need a proper parser
    
    # Check for deprecated functions
    if echo "$code" | grep -q "security\s*("; then
        warnings+=("Using security() function - ensure proper usage with request.security() for v5")
    fi
    
    # Check for proper plot statements
    if echo "$code" | grep -qE "(strategy|indicator)\(" && ! echo "$code" | grep -q "plot\|plotshape\|plotchar"; then
        warnings+=("No plot statements found - strategy may not display anything")
    fi
    
    # Determine validity
    local valid="true"
    if [ ${#errors[@]} -gt 0 ]; then
        valid="false"
    fi
    
    # Build JSON output
    local errors_json="[]"
    local warnings_json="[]"
    
    if [ ${#errors[@]} -gt 0 ]; then
        errors_json=$(printf '%s\n' "${errors[@]}" | jq -R . | jq -s .)
    fi
    
    if [ ${#warnings[@]} -gt 0 ]; then
        warnings_json=$(printf '%s\n' "${warnings[@]}" | jq -R . | jq -s .)
    fi
    
    cat <<EOF
{
  "valid": $valid,
  "errors": $errors_json,
  "warnings": $warnings_json,
  "checks_performed": [
    "version_declaration",
    "strategy_indicator_declaration",
    "parentheses_balance"
  ]
}
EOF
}

# Main logic
if [ $# -ge 1 ]; then
    # File path provided
    script_path="$1"
    if [ ! -f "$script_path" ]; then
        error_json "File not found: $script_path"
    fi
    code=$(cat "$script_path")
    validate_script "$code"
elif [ ! -t 0 ]; then
    # JSON input from stdin
    input=$(cat)
    code=$(echo "$input" | jq -r '.code // empty')
    
    if [ -z "$code" ]; then
        error_json "Missing 'code' field in input"
    fi
    
    validate_script "$code"
else
    error_json "Usage: $0 '/path/to/script.pine' OR echo '{\"code\": \"...\"}' | $0"
fi

exit 0