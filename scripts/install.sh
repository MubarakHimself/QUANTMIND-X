#!/bin/bash
# QuantMindX Fresh VPS Installation Script
# Installs all dependencies and sets up the system for production

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

INSTALL_DIR="/opt/quantmindx"

echo -e "${BLUE}╔══════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║       ${GREEN}QuantMindX VPS Installation${BLUE}        ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════╝${NC}"
echo ""

# ============== Step 1: Check Privileges ==============
echo -e "${BLUE}Step 1: Checking privileges...${NC}"

if [[ $EUID -ne 0 ]] && ! sudo -n true 2>/dev/null; then
    echo -e "${YELLOW}This script requires sudo privileges.${NC}"
    echo -e "${YELLOW}Please enter your password if prompted.${NC}"
fi

# ============== Step 2: Install System Dependencies ==============
echo -e "${BLUE}Step 2: Installing system dependencies...${NC}"

# Detect OS
if [[ -f /etc/debian_version ]]; then
    # Debian/Ubuntu
    echo -e "${BLUE}Detected Debian/Ubuntu system${NC}"
    
    sudo apt-get update
    sudo apt-get install -y \
        python3.11 \
        python3.11-venv \
        python3-pip \
        docker.io \
        docker-compose \
        git \
        curl \
        jq \
        ufw
elif [[ -f /etc/redhat-release ]]; then
    # RHEL/CentOS
    echo -e "${BLUE}Detected RHEL/CentOS system${NC}"
    
    sudo yum install -y \
        python3.11 \
        python3-pip \
        docker \
        docker-compose \
        git \
        curl \
        jq \
        firewalld
else
    echo -e "${YELLOW}Warning: Unrecognized OS. Please install dependencies manually:${NC}"
    echo "  - python3.11"
    echo "  - docker"
    echo "  - docker-compose"
    echo "  - git"
    echo "  - curl"
    echo "  - jq"
fi

# Ensure Docker is running
echo -e "${BLUE}Starting Docker service...${NC}"
sudo systemctl start docker
sudo systemctl enable docker

# Add current user to docker group if not already
if ! groups | grep -q docker; then
    echo -e "${BLUE}Adding user to docker group...${NC}"
    sudo usermod -aG docker "$USER"
    echo -e "${YELLOW}Note: You may need to log out and back in for docker group to take effect${NC}"
fi

# ============== Step 3: Clone/Set Up Repository ==============
echo -e "${BLUE}Step 3: Setting up repository...${NC}"

if [[ -d "$INSTALL_DIR" ]]; then
    echo -e "${YELLOW}Installation directory already exists: $INSTALL_DIR${NC}"
    PROJECT_ROOT="$INSTALL_DIR"
else
    # Check if we're running from within the repo
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    if [[ -f "$SCRIPT_DIR/../.git/config" ]]; then
        # Running from within the repo
        PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
        echo -e "${GREEN}Using existing repository at: $PROJECT_ROOT${NC}"
    else
        # Clone the repo
        echo -e "${BLUE}Cloning repository to $INSTALL_DIR...${NC}"
        sudo git clone https://github.com/MubarakHimself/QUANTMIND-X.git "$INSTALL_DIR"
        sudo chown -R "$USER:$USER" "$INSTALL_DIR"
        PROJECT_ROOT="$INSTALL_DIR"
    fi
fi

cd "$PROJECT_ROOT"

# ============== Step 4: Create Python Virtual Environment ==============
echo -e "${BLUE}Step 4: Setting up Python environment...${NC}"

if [[ ! -d "venv" ]]; then
    python3.11 -m venv venv
fi

source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# ============== Step 5: Configure Environment Variables ==============
echo -e "${BLUE}Step 5: Configuring environment...${NC}"

if [[ ! -f ".env" ]]; then
    cp .env.example .env
    
    echo ""
    echo -e "${YELLOW}══════════════════════════════════════════════════════${NC}"
    echo -e "${YELLOW}Please configure the following required environment variables:${NC}"
    echo -e "${YELLOW}══════════════════════════════════════════════════════${NC}"
    echo ""
    
    # Prompt for Grafana Cloud settings
    echo -e "${BLUE}Grafana Cloud Configuration:${NC}"
    read -p "Enter Grafana Prometheus URL (e.g., https://prometheus-prod-XX-prod-XX-XX.grafana.net): " grafana_prom_url
    read -p "Enter Grafana Loki URL (e.g., https://logs-prod-XX.grafana.net): " grafana_loki_url
    read -p "Enter Grafana Instance ID: " grafana_instance_id
    read -p "Enter Grafana API Key: " grafana_api_key
    
    # Prompt for MT5 Bridge token
    echo ""
    echo -e "${BLUE}MT5 Bridge Configuration:${NC}"
    read -p "Enter MT5 Bridge Token (or press Enter to use default): " mt5_token
    mt5_token=${mt5_token:-secret-token-change-me}
    
    # Prompt for Firecrawl API key
    echo ""
    echo -e "${BLUE}Firecrawl Configuration:${NC}"
    read -p "Enter Firecrawl API Key: " firecrawl_key
    
    # Update .env file
    sed -i "s|GRAFANA_PROMETHEUS_URL=.*|GRAFANA_PROMETHEUS_URL=$grafana_prom_url|" .env
    sed -i "s|GRAFANA_LOKI_URL=.*|GRAFANA_LOKI_URL=$grafana_loki_url|" .env
    sed -i "s|GRAFANA_INSTANCE_ID=.*|GRAFANA_INSTANCE_ID=$grafana_instance_id|" .env
    sed -i "s|GRAFANA_API_KEY=.*|GRAFANA_API_KEY=$grafana_api_key|" .env
    sed -i "s|MT5_BRIDGE_TOKEN=.*|MT5_BRIDGE_TOKEN=$mt5_token|" .env
    sed -i "s|FIRECRAWL_API_KEY=.*|FIRECRAWL_API_KEY=$firecrawl_key|" .env
    sed -i "s|QUANTMIND_ENV=.*|QUANTMIND_ENV=production|" .env
    
    echo ""
    echo -e "${GREEN}✅ .env file configured${NC}"
else
    echo -e "${GREEN}.env file already exists, skipping...${NC}"
fi

# ============== Step 6: Validate Environment ==============
echo -e "${BLUE}Step 6: Validating environment configuration...${NC}"

if ! ./scripts/validate_env.sh; then
    echo -e "${RED}❌ Environment validation failed. Please fix the missing variables in .env${NC}"
    exit 1
fi

# ============== Step 7: Create Snapshot Directory ==============
echo -e "${BLUE}Step 7: Creating snapshot directory...${NC}"

sudo mkdir -p /opt/quantmindx/snapshots
sudo chown -R "$USER:$USER" /opt/quantmindx/snapshots

# ============== Step 8: Create Required Directories ==============
echo -e "${BLUE}Step 8: Creating required directories...${NC}"

mkdir -p data/logs
mkdir -p logs

# ============== Step 9: Install CLI Tool ==============
echo -e "${BLUE}Step 9: Installing quantmind CLI...${NC}"

sudo cp scripts/quantmind_cli.sh /usr/local/bin/quantmind
sudo chmod +x /usr/local/bin/quantmind

echo -e "${GREEN}✅ CLI installed: quantmind command available system-wide${NC}"

# ============== Step 10: Start Services ==============
echo -e "${BLUE}Step 10: Starting services...${NC}"

docker compose -f docker-compose.production.yml pull
docker compose -f docker-compose.production.yml up -d

echo -e "${BLUE}Waiting for services to start...${NC}"
sleep 30

# ============== Step 11: Run Health Check ==============
echo -e "${BLUE}Step 11: Running health check...${NC}"

if quantmind health; then
    echo ""
    echo -e "${GREEN}═══════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  QuantMindX Installation Complete!${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════${NC}"
    echo ""
    echo -e "  ${BLUE}API Server:${NC}     http://localhost:8000"
    echo -e "  ${BLUE}API Health:${NC}     http://localhost:8000/health"
    echo -e "  ${BLUE}API Version:${NC}    http://localhost:8000/api/version"
    echo -e "  ${BLUE}MT5 Bridge:${NC}     http://localhost:5005"
    echo -e "  ${BLUE}Prometheus:${NC}     http://localhost:9099"
    echo ""
    echo -e "  ${BLUE}CLI Tool:${NC}       quantmind --help"
    echo ""
    echo -e "${YELLOW}Next steps:${NC}"
    echo "  1. Configure your firewall to allow access to required ports"
    echo "  2. Set up SSL/TLS certificates for production use"
    echo "  3. Configure your Grafana Cloud dashboards"
    echo ""
else
    echo ""
    echo -e "${RED}═══════════════════════════════════════════════${NC}"
    echo -e "${RED}  Installation completed with warnings${NC}"
    echo -e "${RED}═══════════════════════════════════════════════${NC}"
    echo ""
    echo -e "${YELLOW}Some services may not be healthy. Check logs with:${NC}"
    echo "  quantmind logs"
    echo ""
fi