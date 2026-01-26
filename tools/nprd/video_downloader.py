"""
Video Downloader - Downloads YouTube videos using yt-dlp
"""
import yt_dlp
from pathlib import Path
from typing import Dict, Optional, Callable
import hashlib


def get_video_info(url: str) -> Dict:
    """Get video metadata without downloading."""
    ydl_opts = {'quiet': True, 'no_warnings': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return {
            'id': info.get('id'),
            'title': info.get('title'),
            'duration': info.get('duration', 0),
            'duration_string': info.get('duration_string', '0:00'),
            'uploader': info.get('uploader'),
            'upload_date': info.get('upload_date'),
        }


def download_video(
    url: str,
    output_dir: Path,
    progress_callback: Optional[Callable] = None
) -> Path:
    """
    Download YouTube video.
    
    Args:
        url: YouTube video URL
        output_dir: Where to save video
        progress_callback: Optional callback for progress updates
    
    Returns:
        Path to downloaded .mp4 file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    def progress_hook(d):
        if progress_callback and d['status'] == 'downloading':
            total = d.get('total_bytes') or d.get('total_bytes_estimate', 0)
            downloaded = d.get('downloaded_bytes', 0)
            if total > 0:
                progress_callback(downloaded / total * 100)
    
    # More robust format selection
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': str(output_dir / '%(title)s.%(ext)s'),
        'quiet': False,
        'no_warnings': False,
        'progress_hooks': [progress_hook] if progress_callback else [],
        'merge_output_format': 'mp4',
        # Handle geo-restrictions and age-gates
        'geo_bypass': True,
        'nocheckcertificate': True,
        # Retry on failure
        'retries': 5,
        'fragment_retries': 5,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        # Get the actual downloaded filename
        if info.get('requested_downloads'):
            filename = info['requested_downloads'][0]['filepath']
        else:
            filename = ydl.prepare_filename(info)
        return Path(filename)


def get_video_hash(video_path: Path) -> str:
    """Get SHA256 hash of video file for caching."""
    sha256 = hashlib.sha256()
    with open(video_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()[:16]
