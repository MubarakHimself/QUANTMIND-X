"""
Gemini Client - Video analysis using Google Gemini 1.5 Pro
Uses new google-genai SDK (replaces deprecated google.generativeai)

Install: pip install google-genai
"""
import time
import json
import re
from pathlib import Path
from typing import Dict, Any, Optional

from google import genai


class GeminiVideoAnalyzer:
    """Analyzes videos using Gemini 1.5 Pro."""
    
    def __init__(self, api_key: str):
        """Initialize Gemini client with API key."""
        self.client = genai.Client(api_key=api_key)
        self.uploaded_files = {}
    
    def upload_video(self, video_path: Path, display_name: Optional[str] = None) -> Any:
        """
        Upload video to Gemini File API.
        
        Returns:
            File object for use in generate_content
        """
        video_path = Path(video_path)
        name = display_name or video_path.name
        
        print(f"📤 Uploading {name}...")
        
        # Upload file using new SDK
        video_file = self.client.files.upload(file=str(video_path))
        
        # Wait for processing (files need to be ACTIVE before use)
        while video_file.state.name == "PROCESSING":
            print("   ⏳ Processing video...")
            time.sleep(5)
            video_file = self.client.files.get(name=video_file.name)
        
        if video_file.state.name == "FAILED":
            raise ValueError(f"Video processing failed: {video_file.name}")
        
        self.uploaded_files[str(video_path)] = video_file
        print(f"✅ Upload complete: {video_file.name}")
        return video_file
    
    def analyze_video(self, video_file: Any, prompt: str) -> Dict[str, Any]:
        """
        Send video + prompt to Gemini, get JSON response.
        
        Args:
            video_file: File object from upload_video()
            prompt: The analysis prompt
        """
        print("🤖 Analyzing video with Gemini 1.5 Pro...")
        
        # Generate content with file and prompt
        # Using gemini-2.0-flash for video (or gemini-1.5-pro-latest)
        response = self.client.models.generate_content(
            model='gemini-2.0-flash',
            contents=[video_file, prompt]
        )
        
        return self._parse_response(response.text)
    
    def _parse_response(self, text: str) -> Dict[str, Any]:
        """Parse JSON from Gemini response."""
        # Remove markdown code fences if present
        text = re.sub(r'```json\s*\n?', '', text)
        text = re.sub(r'```\s*\n?', '', text)
        text = text.strip()
        
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            # Try to fix common issues
            text = re.sub(r',\s*}', '}', text)  # Remove trailing commas
            text = re.sub(r',\s*]', ']', text)
            try:
                return json.loads(text)
            except:
                # Return raw text in a structure if JSON fails
                print(f"⚠️  Could not parse JSON, returning raw text")
                return {
                    "video_id": "unknown",
                    "meta": {"title": "Parse Error", "raw_response": True},
                    "timeline": [],
                    "raw_text": text[:5000]
                }
    
    def get_cached_file(self, video_path: Path) -> Optional[Any]:
        """Get cached file if video was already uploaded."""
        return self.uploaded_files.get(str(video_path))
    
    def delete_file(self, video_file: Any):
        """Delete uploaded file to free up quota."""
        try:
            self.client.files.delete(name=video_file.name)
            print(f"🗑️  Deleted: {video_file.name}")
        except Exception as e:
            print(f"⚠️  Could not delete file: {e}")
