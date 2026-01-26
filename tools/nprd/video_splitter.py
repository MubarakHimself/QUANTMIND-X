"""
Video Splitter - Splits videos >45min into chunks using ffmpeg
"""
import ffmpeg
from pathlib import Path
from typing import List


def get_video_duration(video_path: Path) -> float:
    """Get video duration in seconds."""
    probe = ffmpeg.probe(str(video_path))
    return float(probe['format']['duration'])


def split_video(
    video_path: Path,
    chunk_duration_minutes: int = 45,
    output_dir: Path = None
) -> List[Path]:
    """
    Split video into chunks of specified duration.
    
    Args:
        video_path: Input video file
        chunk_duration_minutes: Max chunk length (default 45)
        output_dir: Where to save chunks (default: same as video)
    
    Returns:
        List of chunk file paths, ordered
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir) if output_dir else video_path.parent / "chunks"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    duration = get_video_duration(video_path)
    chunk_seconds = chunk_duration_minutes * 60
    
    # If video is shorter than chunk duration, just copy
    if duration <= chunk_seconds:
        output_file = output_dir / f"{video_path.stem}_chunk_001.mp4"
        (
            ffmpeg
            .input(str(video_path))
            .output(str(output_file), c='copy')
            .overwrite_output()
            .run(quiet=True)
        )
        return [output_file]
    
    chunks = []
    start = 0
    chunk_num = 1
    
    while start < duration:
        chunk_length = min(chunk_seconds, duration - start)
        output_file = output_dir / f"{video_path.stem}_chunk_{chunk_num:03d}.mp4"
        
        (
            ffmpeg
            .input(str(video_path), ss=start, t=chunk_length)
            .output(str(output_file), c='copy')
            .overwrite_output()
            .run(quiet=True)
        )
        
        chunks.append(output_file)
        start += chunk_length
        chunk_num += 1
    
    return chunks


def format_duration(seconds: float) -> str:
    """Format seconds to HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"
