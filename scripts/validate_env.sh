#!/bin/bash
# Environment Variable Validator for QuantMindX
# Validates that all required keys from .env.example are set in .env

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

ENV_EXAMPLE="$PROJECT_ROOT/.env.example"
ENV_FILE="$PROJECT_ROOT/.env"

# Check if .env.example exists
if [[ ! -f "$ENV_EXAMPLE" ]]; then
    echo -e "${RED}❌ .env.example not found at $ENV_EXAMPLE${NC}"
    exit 1
fi

# Check if .env exists
if [[ ! -f "$ENV_FILE" ]]; then
    echo -e "${RED}❌ .env file not found at $ENV_FILE${NC}"
    echo -e "${YELLOW}   Run: cp .env.example .env && edit .env with your values${NC}"
    exit 1
fi

# Source .env to get values
set -a
source "$ENV_FILE"
set +a

# Parse keys from .env.example (ignore comments and empty lines)
# Keys are lines matching: KEY=...
MISSING_KEYS=()
while IFS= read -r line; do
    # Skip comments and empty lines
    [[ "$line" =~ ^[[:space:]]*# ]] && continue
    [[ -z "${line// }" ]] && continue
    
    # Extract key (everything before =)
    if [[ "$line" =~ ^([A-Za-z_][A-Za-z0-9_]*)= ]]; then
        KEY="${BASH_REMATCH[1]}"
        
        # Check if key is set in environment (from sourced .env)
        if [[ -z "${!KEY}" ]]; then
            # Check if it's a placeholder value that needs to be changed
            MISSING_KEYS+=("$KEY")
        fi
    fi
done < "$ENV_EXAMPLE"

# Report results
if [[ ${#MISSING_KEYS[@]} -eq 0 ]]; then
    echo -e "${GREEN}✅ All required environment variables are set${NC}"
    exit 0
else
    echo -e "${RED}❌ Missing or unset environment variables:${NC}"
    for key in "${MISSING_KEYS[@]}"; do
        echo -e "   ${RED}• $key${NC}"
    done
    echo ""
    echo -e "${YELLOW}Edit $ENV_FILE to set these values${NC}"
    exit 1
fi