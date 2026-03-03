"""
Terminal Endpoints for QuantMind IDE.

Handles terminal/shell operations:
- Video ingest processing
"""

import os
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


# Configuration
DATA_DIR = Path(os.getenv("QUANTMIND_DATA_DIR", "data"))
SCRAPED_ARTICLES_DIR = DATA_DIR / "scraped_articles"


class TerminalEndpoint:
    """Terminal endpoint handler for QuantMind IDE.

    Manages terminal/shell operations including video ingest
    processing.
    """

    def __init__(self):
        """Initialize terminal endpoint."""
        SCRAPED_ARTICLES_DIR.mkdir(parents=True, exist_ok=True)
        self._video_handler = None

    def _get_video_handler(self):
        """Lazy load video handler."""
        if self._video_handler is None:
            self._video_handler = VideoIngestAPIHandler()
        return self._video_handler

    def process_video(self, video_path: str, department: str, priority: str = "normal") -> Dict[str, Any]:
        """Process a video file for analysis."""
        return self._get_video_handler().process_video(video_path, department, priority)

    def get_video_status(self, job_id: str) -> Dict[str, Any]:
        """Get video processing status."""
        return self._get_video_handler().get_video_status(job_id)

    def list_video_jobs(self) -> List[Dict[str, Any]]:
        """List all video processing jobs."""
        return self._get_video_handler().list_video_jobs()

    def get_scraped_articles(self) -> List[Dict[str, Any]]:
        """Get list of scraped articles."""
        return self._get_video_handler().get_scraped_articles()


class VideoIngestAPIHandler:
    """Handler for video ingest processing."""

    def __init__(self):
        SCRAPED_ARTICLES_DIR.mkdir(parents=True, exist_ok=True)
        self.jobs_dir = DATA_DIR / "video_jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

    def process_video(self, video_path: str, department: str, priority: str = "normal") -> Dict[str, Any]:
        """Process a video file for analysis."""
        import uuid

        if not os.path.exists(video_path):
            return {
                "success": False,
                "error": f"Video file not found: {video_path}"
            }

        job_id = str(uuid.uuid4())[:8]

        job_data = {
            "job_id": job_id,
            "video_path": video_path,
            "department": department,
            "priority": priority,
            "status": "queued",
            "created_at": datetime.now().isoformat(),
            "result": None,
            "error": None
        }

        job_file = self.jobs_dir / f"{job_id}.json"
        with open(job_file, "w") as f:
            json.dump(job_data, f, indent=2)

        # TODO: Actually process the video in background
        # For now, just return the job info

        return {
            "success": True,
            "job_id": job_id,
            "status": "queued",
            "message": f"Video queued for processing by {department}"
        }

    def get_video_status(self, job_id: str) -> Dict[str, Any]:
        """Get video processing status."""
        job_file = self.jobs_dir / f"{job_id}.json"

        if not job_file.exists():
            return {
                "success": False,
                "error": "Job not found"
            }

        try:
            with open(job_file) as f:
                job_data = json.load(f)
            return {
                "success": True,
                "job": job_data
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def list_video_jobs(self) -> List[Dict[str, Any]]:
        """List all video processing jobs."""
        jobs = []

        for job_file in self.jobs_dir.glob("*.json"):
            try:
                with open(job_file) as f:
                    job_data = json.load(f)
                jobs.append(job_data)
            except:
                pass

        return sorted(jobs, key=lambda x: x.get("created_at", ""), reverse=True)

    def get_scraped_articles(self) -> List[Dict[str, Any]]:
        """Get list of scraped articles."""
        articles = []

        if not SCRAPED_ARTICLES_DIR.exists():
            return articles

        for article_file in SCRAPED_ARTICLES_DIR.glob("*.json"):
            try:
                with open(article_file) as f:
                    data = json.load(f)
                articles.append({
                    "id": article_file.stem,
                    "title": data.get("title", "Untitled"),
                    "url": data.get("url", ""),
                    "scraped_at": data.get("scraped_at", "")
                })
            except:
                pass

        return sorted(articles, key=lambda x: x.get("scraped_at", ""), reverse=True)
