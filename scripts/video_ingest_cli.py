#!/usr/bin/env python3
"""Video Ingest CLI - Dumb tool for downloading and processing YouTube videos."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.video_ingest.tool import VideoIngestTool


def main():
    parser = argparse.ArgumentParser(
        description="Video Ingest CLI - Download and process YouTube videos"
    )
    parser.add_argument("url", help="YouTube video or playlist URL")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM processing")
    parser.add_argument("--model", default="qwen3-vl-8b", help="Model to use")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    try:
        tool = VideoIngestTool(model=args.model)
        result = tool.process_video(args.url, use_llm=not args.no_llm)
        print(f"\nVideo processed: {result}")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
