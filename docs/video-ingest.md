# Video Ingest Tool

Dumb tool for downloading and processing YouTube trading videos.

## Usage

```bash
# Download and process with Qwen
python3 scripts/video_ingest_cli.py "https://youtu.be/VIDEO_ID"

# Download only (no LLM)
python3 scripts/video_ingest_cli.py "https://youtu.be/VIDEO_ID" --no-llm

# Specify model
python3 scripts/video_ingest_cli.py "https://youtu.be/VIDEO_ID" --model qwen3-vl-235b-a22b-thinking

# Verbose output
python3 scripts/video_ingest_cli.py "https://youtu.be/VIDEO_ID" --verbose
```

## Output Structure

```
video_in/
├── downloads/
│   └── VIDEO_ID_title/
│       ├── video.mp4
│       ├── audio.mp3
│       ├── captions.vtt
│       ├── metadata.json
│       ├── chunks.json
│       └── analysis.json (if --no-llm not set)
├── processing/
├── completed/
└── failed/
```

## Rate Limiting

Default: 2000 requests/day (Qwen API)
- Uses token bucket algorithm with sliding window
- Thread-safe for concurrent processing

## Smart Chunking

Video duration -> chunks:
- < 10 min: 1 chunk
- 10-30 min: 2 chunks
- 30-60 min: 3 chunks
- 1-2 hours: 4 chunks
- 2-4 hours: 6 chunks
- > 4 hours: 8 chunks

## Models

- Default: Qwen3-VL-235B-A22B-Thinking (with reasoning/thinking capability)
- Handles hours-long videos with full recall
- Large context window for comprehensive analysis
