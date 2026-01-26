"""
Playlist Handler - Extracts video list from YouTube playlists
"""
import yt_dlp
from typing import List, Dict


def is_playlist(url: str) -> bool:
    """Check if URL is a YouTube playlist."""
    return "playlist" in url.lower() or "list=" in url


def get_playlist_videos(playlist_url: str) -> Dict:
    """
    Extract all video URLs from playlist.
    
    Returns:
        Dict with 'title', 'id', and 'videos' list
    """
    ydl_opts = {
        'extract_flat': True,
        'quiet': True,
        'no_warnings': True,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(playlist_url, download=False)
        
        if 'entries' not in result:
            raise ValueError("Not a valid playlist")
        
        videos = []
        for entry in result['entries']:
            if entry is None:
                continue
            videos.append({
                'id': entry.get('id'),
                'url': f"https://www.youtube.com/watch?v={entry['id']}",
                'title': entry.get('title', 'Unknown'),
                'duration': entry.get('duration', 0),
            })
        
        return {
            'playlist_id': result.get('id'),
            'playlist_title': result.get('title', 'Unknown Playlist'),
            'total_videos': len(videos),
            'videos': videos
        }
