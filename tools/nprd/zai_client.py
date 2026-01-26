"""
ZAI Client - Video analysis using ZhipuAI GLM-4.5V
Uses zai-sdk package

Install: pip install zai-sdk
API Key: Get from https://z.ai/manage-apikey/apikey-list
"""
import time
import json
import re
import base64
from pathlib import Path
from typing import Dict, Any, Optional

from zai import ZaiClient


class ZaiVideoAnalyzer:
    """Analyzes videos using ZhipuAI GLM-4.5V."""
    
    def __init__(self, api_key: str, model_name: str = "glm-4.5v"):
        """Initialize ZAI client with API key and model."""
        self.client = ZaiClient(api_key=api_key)
        self.model = model_name
    
    def encode_video_base64(self, video_path: Path) -> str:
        """
        Encode video to base64 for API.
        Note: Large videos may need to be split or uploaded differently.
        """
        video_path = Path(video_path)
        with open(video_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    def analyze_video(self, video_path: Path, prompt: str) -> Dict[str, Any]:
        """
        Analyze video with GLM-4.5V.
        
        For large videos, this encodes as base64 and sends via API.
        """
        video_path = Path(video_path)
        
        print(f"📤 Encoding {video_path.name} for ZAI...")
        
        # Check file size
        file_size_mb = video_path.stat().st_size / (1024 * 1024)
        if file_size_mb > 100:
            print(f"⚠️  Video is {file_size_mb:.1f}MB - large files may take longer")
        
        # Encode video
        video_b64 = self.encode_video_base64(video_path)
        
        # Determine the mime type
        suffix = video_path.suffix.lower()
        mime_type = {
            '.mp4': 'video/mp4',
            '.mov': 'video/quicktime',
            '.m4v': 'video/x-m4v',
            '.avi': 'video/x-msvideo',
            '.webm': 'video/webm'
        }.get(suffix, 'video/mp4')
        
        print("🤖 Analyzing video with GLM-4.5V...")
        
        # Create video data URL
        video_url = f"data:{mime_type};base64,{video_b64}"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video_url",
                                "video_url": {
                                    "url": video_url
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ],
                thinking={"type": "enabled"},
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            return self._parse_response(content)
            
        except Exception as e:
            error_str = str(e)
            if "video_url" in error_str.lower():
                # Try with image_url type instead (some APIs use this)
                print("⚠️  Trying alternative content type...")
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": video_url
                                    }
                                },
                                {
                                    "type": "text",
                                    "text": prompt
                                }
                            ]
                        }
                    ],
                    thinking={"type": "enabled"},
                    response_format={"type": "json_object"}
                )
                
                content = response.choices[0].message.content
                return self._parse_response(content)
            else:
                raise
    
    def _parse_response(self, text: str) -> Dict[str, Any]:
        """Parse JSON from ZAI response."""
        if not text:
            return {"error": "Empty response"}
        
        # Clean up thinking tags if present
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        text = re.sub(r'<\|begin_of_box\|>.*?<\|end_of_box\|>', '', text, flags=re.DOTALL)
        
        # Remove markdown code fences if present
        text = re.sub(r'```json\s*\n?', '', text)
        text = re.sub(r'```\s*\n?', '', text)
        text = text.strip()
        
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            # Try to fix common issues
            text = re.sub(r',\s*}', '}', text)
            text = re.sub(r',\s*]', ']', text)
            try:
                return json.loads(text)
            except:
                print(f"⚠️  Could not parse JSON, returning raw text")
                return {
                    "video_id": "unknown",
                    "meta": {"title": "Parse Error", "raw_response": True},
                    "timeline": [],
                    "raw_text": text[:5000]
                }
