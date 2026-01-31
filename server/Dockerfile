FROM node:20-slim

# Install System Dependencies
# python3/pip -> for yt-dlp and wrapper scripts
# ffmpeg -> for media processing
# git -> often needed for npm dependencies or gemini extensions
# curl -> for healthchecks
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python Tools
# yt-dlp: The best tool for downloading YouTube videos
# Break-system-packages is needed on newer Debian/Ubuntu to pip install globally
RUN pip3 install yt-dlp --break-system-packages

# Install Gemini CLI and Qwen Code CLI Globally
RUN npm install -g @google/gemini-cli @qwen-code/qwen-code

# Setup Application Directory
WORKDIR /app

# Copy Package Files first for caching
COPY server/package.json ./server/
COPY extensions/nprd-extension/package.json ./extensions/nprd-extension/

# Install Node Dependencies
WORKDIR /app/server
RUN npm install

# Copy Source Code
# We copy the 'extensions' folder so the server can access the scripts/skills
WORKDIR /app
COPY extensions ./extensions
COPY server ./server

# Set Working Directory via Env or Just default
WORKDIR /app/server

# Expose API Port
EXPOSE 3000

# Default Command
CMD ["node", "nprd_server.js"]
