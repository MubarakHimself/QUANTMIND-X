# Claude Code Docker Setup

Isolated Claude Code environment for QuantMindX development.

## Quick Start

```bash
# 1. Copy environment file and add your API key
cp .env.example .env
# Edit .env with your ANTHROPIC_AUTH_TOKEN

# 2. Build the Docker image
docker compose build

# 3. Run Claude Code
docker compose run --rm claude-code

# Or run interactively
docker compose run -it --rm claude-code bash
```

## Directory Structure

```
docker/claude-code/
├── Dockerfile           # Container definition
├── docker-compose.yml   # Compose configuration
├── setup.sh            # Entry point script
├── .env.example        # Environment template
├── config/
│   ├── settings.json   # Claude Code settings
│   └── mcp.json        # MCP server configuration
└── README.md           # This file
```

## Configuration

### Environment Variables

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_AUTH_TOKEN` | Your API token (Z.ai or Anthropic) |
| `ANTHROPIC_BASE_URL` | API endpoint URL |
| `ANTHROPIC_API_KEY` | Direct Anthropic API key |

### Volume Mounts

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| `../../` | `/workspace` | QuantMindX project |
| `./config` | `/config` | Settings (read-only) |
| `claude-data` | `~/.claude` | Persistent data |
| `/var/run/docker.sock` | `/var/run/docker.sock` | Docker access |

## Usage Examples

### Run Claude Code in project directory
```bash
docker compose run --rm claude-code
```

### Run with specific model
```bash
docker compose run --rm claude-code claude --model claude-sonnet-4-20250514
```

### Run with non-interactive mode
```bash
docker compose run --rm claude-code claude -p "Analyze the project structure"
```

### Access container shell
```bash
docker compose run -it --rm claude-code bash
```

### Run as daemon (background)
```bash
docker compose up -d
docker compose exec claude-code claude
```

## MCP Servers

The following MCP servers are configured:

- **context7**: Documentation lookup
- **svelte**: Svelte/SvelteKit assistance
- **chrome_devtools**: Browser automation
- **MCP_DOCKER**: Docker integration

## Differences from Native Installation

| Feature | Native | Docker |
|---------|--------|--------|
| Pencil MCP | ✅ VSCode extension | ❌ Not available |
| Git config | Auto-detected | Mounted read-only |
| Docker access | Full | Via socket mount |
| Isolation | Shared system | Fully isolated |

## Troubleshooting

### Permission denied errors
```bash
# Ensure your UID/GID matches
docker compose build --build-arg USER_UID=$(id -u) --build-arg USER_GID=$(id -g)
```

### Docker socket access
```bash
# Add user to docker group (on host)
sudo usermod -aG docker $USER
```

### Reset persistent data
```bash
docker volume rm quantmind-claude-data
```

## Building from Scratch

```bash
# Clean build
docker compose build --no-cache

# Rebuild with latest Claude Code
docker compose build --pull
```
