# Claude Code Docker Image
# Isolated environment for Claude Code with QuantMindX project

FROM node:20-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    bash \
    python3 \
    python3-pip \
    python3-venv \
    docker.io \
    sudo \
    jq \
    gettext \
    && rm -rf /var/lib/apt/lists/*

# Install Claude Code CLI (as root before switching user)
RUN npm install -g @anthropic-ai/claude-code

# Use existing node user (UID 1000, GID 1000 in node images)
ARG USERNAME=node

# Add sudo access for node user
RUN echo "node ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/node \
    && chmod 0440 /etc/sudoers.d/node

# Set working directory
WORKDIR /workspace

# Copy setup scripts
COPY setup.sh /setup.sh
RUN chmod +x /setup.sh

# Create Claude config directory
RUN mkdir -p /home/node/.claude && chown -R node:node /home/node/.claude

# Set environment
ENV CLAUDE_CONFIG_DIR=/home/node/.claude
ENV ANTHROPIC_API_KEY=""

# Switch to non-root user
USER $USERNAME

# Entry point
ENTRYPOINT ["/setup.sh"]
CMD ["claude"]
