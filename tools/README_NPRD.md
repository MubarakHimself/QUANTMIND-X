# NPRD CLI - Usage Guide

## Quick Start

```bash
cd /home/mubarkahimself/Desktop/QUANTMINDX
source venv/bin/activate

# Set API key
export GEMINI_API_KEY="your-key-here"

# Process single video
python tools/nprd_cli.py "https://youtube.com/watch?v=VIDEO_ID"

# Process playlist
python tools/nprd_cli.py "https://youtube.com/playlist?list=PLAYLIST_ID" --playlist
```

## CLI Banner

```
███╗   ██╗██████╗ ██████╗ ██████╗ 
████╗  ██║██╔══██╗██╔══██╗██╔══██╗
██╔██╗ ██║██████╔╝██████╔╝██║  ██║
██║╚██╗██║██╔═══╝ ██╔══██╗██║  ██║
██║ ╚████║██║     ██║  ██║██████╔╝
╚═╝  ╚═══╝╚═╝     ╚═╝  ╚═╝╚═════╝ 

╔═══════════════════════════════════════════════╗
║  General Purpose Multimodal Video Indexer     ║
║  Unbiased extraction • No strategy analysis  ║
║  Version 1.0.0 • Powered by Gemini 1.5 Pro   ║
╚═══════════════════════════════════════════════╝
```

## Options

| Option | Description |
|--------|-------------|
| `--output-dir` | Output directory (default: outputs/videos) |
| `--api-key` | Gemini API key (or set GEMINI_API_KEY env) |
| `--playlist` | Treat input as playlist |
| `--max-videos` | Limit videos from playlist |
| `--max-chunks` | Limit chunks per video |
| `--resume` | Resume interrupted processing |

## Output Structure

```
outputs/videos/
└── Video_Title/
    ├── Video_Title.mp4     # Downloaded video
    ├── chunks/              # Split chunks (if >45min)
    │   ├── Video_Title_chunk_001.mp4
    │   ├── Video_Title_chunk_001.json
    │   └── ...
    ├── Video_Title.json    # Combined analysis
    └── Video_Title.md      # Human-readable summary
```

## Dependencies

```bash
pip install yt-dlp ffmpeg-python google-generativeai click tqdm
```

Also requires: `ffmpeg` system package
