"""Video Ingest Tool - Dumb tool for downloading and processing YouTube videos."""

import os
import json
import subprocess
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, List, Dict

# Configuration
VIDEO_IN_ROOT = Path("/home/mubarkahimself/Desktop/QUANTMINDX/video_in")
DEFAULT_MODEL = "qwen3-vl-8b"  # Latest Qwen3-VL model


@dataclass
class VideoMetadata:
    """Metadata for a downloaded video."""
    video_id: str
    title: str
    duration: int  # seconds
    description: str
    uploader: str
    upload_date: str
    view_count: int
    like_count: int
    playlist_id: Optional[str] = None
    playlist_index: Optional[int] = None


class VideoIngestTool:
    """Dumb tool for video ingestion - downloads and organizes video files."""

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self.video_in_root = VIDEO_IN_ROOT
        self.downloads_dir = self.video_in_root / "downloads"
        self.downloads_dir.mkdir(parents=True, exist_ok=True)

    def get_video_info(self, url: str) -> VideoMetadata:
        """Get video metadata without downloading."""
        cmd = [
            "yt-dlp",
            "--dump-json",
            "--no-download",
            "--no-playlist",
            url
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        data = json.loads(result.stdout)

        return VideoMetadata(
            video_id=data.get("id", ""),
            title=data.get("title", ""),
            duration=data.get("duration", 0),
            description=data.get("description", ""),
            uploader=data.get("uploader", ""),
            upload_date=data.get("upload_date", ""),
            view_count=data.get("view_count", 0),
            like_count=data.get("like_count", 0),
            playlist_id=data.get("playlist_id"),
            playlist_index=data.get("playlist_index")
        )

    def download_video(self, url: str, output_dir: Path) -> Path:
        """Download video file to output directory."""
        output_dir.mkdir(parents=True, exist_ok=True)
        output_template = str(output_dir / "%(title)s.%(ext)s")

        cmd = [
            "yt-dlp",
            "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "--output", output_template,
            "--no-playlist",
            url
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        # Find the downloaded file
        files = list(output_dir.glob("*"))
        video_files = [f for f in files if f.suffix in ['.mp4', '.webm']]
        if video_files:
            return video_files[0]
        raise RuntimeError(f"Failed to download video: {result.stderr}")

    def download_audio(self, url: str, output_dir: Path) -> Path:
        """Download audio file to output directory."""
        output_dir.mkdir(parents=True, exist_ok=True)
        output_template = str(output_dir / "audio.%(ext)s")

        cmd = [
            "yt-dlp",
            "-x",  # extract audio
            "--audio-format", "mp3",
            "--output", output_template,
            "--no-playlist",
            url
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        audio_file = output_dir / "audio.mp3"
        if audio_file.exists():
            return audio_file
        raise RuntimeError(f"Failed to download audio: {result.stderr}")

    def download_captions(self, url: str, output_dir: Path) -> Optional[Path]:
        """Download captions/subtitles."""
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "yt-dlp",
            "--write-auto-sub",
            "--sub-lang", "en",
            "--skip-download",
            "--output", str(output_dir / "captions.%(ext)s"),
            "--no-playlist",
            url
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        caption_files = list(output_dir.glob("captions*.vtt"))
        if caption_files:
            return caption_files[0]
        return None

    def calculate_chunks(self, duration_seconds: int) -> List[Dict]:
        """
        Calculate video chunks based on duration.
        Qwen3-VL-8B can handle ~256K tokens, ~1 hour of video well.

        Chunking strategy:
        - < 10 min: 1 chunk (whole video)
        - 10-30 min: 2 chunks (~15 min each)
        - 30-60 min: 3 chunks (~20 min each)
        - 1-2 hours: 4 chunks (~30 min each)
        - 2-4 hours: 6 chunks (~40 min each)
        - > 4 hours: 8 chunks (~60 min each, max)
        """
        if duration_seconds < 600:  # < 10 min
            return [{"start": 0, "end": duration_seconds, "label": "full"}]

        # Calculate optimal chunk count and size
        if duration_seconds < 1800:  # 10-30 min
            chunk_count = 2
        elif duration_seconds < 3600:  # 30-60 min
            chunk_count = 3
        elif duration_seconds < 7200:  # 1-2 hours
            chunk_count = 4
        elif duration_seconds < 14400:  # 2-4 hours
            chunk_count = 6
        else:  # > 4 hours
            chunk_count = 8

        # Divide duration evenly among chunks
        chunk_duration = duration_seconds // chunk_count

        chunks = []
        for i in range(chunk_count):
            start = i * chunk_duration
            # Last chunk ends at duration, others at their chunk boundary
            end = duration_seconds if i == chunk_count - 1 else (i + 1) * chunk_duration
            chunks.append({
                "start": start,
                "end": end,
                "label": f"part_{i+1}"
            })

        return chunks

    def extract_chunk(self, video_path: Path, start_sec: int, end_sec: int, output_path: Path) -> Path:
        """Extract a chunk of video using ffmpeg."""
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-ss", str(start_sec),
            "-to", str(end_sec),
            "-c", "copy",
            str(output_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to extract chunk: {result.stderr}")
        return output_path
