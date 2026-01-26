#!/usr/bin/env python3
"""
OpenRouter Video Analyzer Client for NPRD.
Supports multiple video-capable models via OpenRouter API.
Uses OpenAI-compatible endpoint with base_url replacement.
"""

import base64
import json
from pathlib import Path
from typing import Optional

import httpx

# Available video models on OpenRouter
VIDEO_MODELS = {
    "1": {
        "id": "nvidia/nemotron-nano-12b-v2-vl:free",
        "name": "NVIDIA Nemotron Nano 12B VL (FREE)",
        "input_price": 0,
        "output_price": 0,
    },
    "2": {
        "id": "allenai/molmo-2-8b:free",
        "name": "AllenAI Molmo2 8B (FREE)",
        "input_price": 0,
        "output_price": 0,
    },
    "3": {
        "id": "google/gemini-2.0-flash-lite-001",
        "name": "Google Gemini 2.0 Flash Lite ($0.075/1M)",
        "input_price": 0.075,
        "output_price": 0.30,
    },
    "4": {
        "id": "google/gemini-2.0-flash-001",
        "name": "Google Gemini 2.0 Flash ($0.10/1M)",
        "input_price": 0.10,
        "output_price": 0.40,
    },
    "5": {
        "id": "bytedance-seed/seed-1.6-flash",
        "name": "ByteDance Seed 1.6 Flash ($0.075/1M)",
        "input_price": 0.075,
        "output_price": 0.30,
    },
    "6": {
        "id": "z-ai/glm-4.6v",
        "name": "Z.AI GLM-4.6V ($0.30/1M)",
        "input_price": 0.30,
        "output_price": 0.90,
    },
    "7": {
        "id": "google/gemini-2.5-flash-lite",
        "name": "Google Gemini 2.5 Flash Lite ($0.10/1M)",
        "input_price": 0.10,
        "output_price": 0.40,
    },
}

class OpenRouterVideoAnalyzer:
    """Analyzes videos using OpenRouter API with model selection."""
    
    BASE_URL = "https://openrouter.ai/api/v1"
    
    def __init__(self, api_key: str, model_id: str = "google/gemini-2.0-flash-001"):
        """
        Initialize OpenRouter client.
        
        Args:
            api_key: OpenRouter API key
            model_id: Full model ID (e.g., 'google/gemini-2.0-flash-001')
        """
        self.api_key = api_key
        self.model = model_id
        self.client = httpx.Client(timeout=300.0)  # 5 min timeout for large videos
        
    def validate_model(self) -> bool:
        """Validate that the model exists on OpenRouter."""
        try:
            response = self.client.get(
                f"{self.BASE_URL}/models",
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
            if response.status_code == 200:
                models = response.json().get('data', [])
                model_ids = [m.get('id') for m in models]
                return self.model in model_ids
            return False
        except Exception as e:
            print(f"⚠️ Model validation failed: {e}")
            return True  # Proceed anyway, let API handle it
    
    def encode_video_base64(self, video_path: Path) -> str:
        """Encode video file to base64."""
        file_size_mb = video_path.stat().st_size / (1024 * 1024)
        print(f"📤 Encoding {video_path.name} for OpenRouter...")
        if file_size_mb > 100:
            print(f"⚠️  Video is {file_size_mb:.1f}MB - large files may take longer")
        
        with open(video_path, 'rb') as f:
            return base64.standard_b64encode(f.read()).decode('utf-8')
    
    def analyze_video(self, video_path: Path, prompt: str) -> dict:
        """
        Analyze a video using OpenRouter.
        
        Args:
            video_path: Path to video file
            prompt: Analysis prompt
            
        Returns:
            Parsed JSON response or dict with raw response
        """
        video_b64 = self.encode_video_base64(video_path)
        
        # Determine MIME type
        suffix = video_path.suffix.lower()
        mime_types = {
            '.mp4': 'video/mp4',
            '.webm': 'video/webm',
            '.mov': 'video/quicktime',
            '.avi': 'video/x-msvideo',
        }
        mime_type = mime_types.get(suffix, 'video/mp4')
        
        print(f"🤖 Analyzing video with {self.model}...")
        
        # Build request payload (OpenAI-compatible format with video)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video_url",
                            "video_url": {
                                "url": f"data:{mime_type};base64,{video_b64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            "max_tokens": 16000,
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/quantmindx",
            "X-Title": "NPRD Video Indexer",
        }
        
        response = self.client.post(
            f"{self.BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
        )
        
        if response.status_code != 200:
            error_text = response.text[:500]
            raise Exception(f"OpenRouter API error {response.status_code}: {error_text}")
        
        result = response.json()
        
        # Extract content
        content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
        
        # Try to parse as JSON
        try:
            # Look for JSON in the response
            if '```json' in content:
                json_str = content.split('```json')[1].split('```')[0].strip()
            elif '{' in content:
                start = content.find('{')
                end = content.rfind('}') + 1
                json_str = content[start:end]
            else:
                return {"raw_response": content, "meta": {"model": self.model}}
            
            parsed = json.loads(json_str)
            if 'meta' not in parsed:
                parsed['meta'] = {}
            parsed['meta']['model'] = self.model
            return parsed
            
        except json.JSONDecodeError:
            return {"raw_response": content, "meta": {"model": self.model}}
    
    @staticmethod
    def list_models():
        """Print available video models."""
        print("\n📋 Available Video Models on OpenRouter:")
        print("-" * 60)
        for key, model in VIDEO_MODELS.items():
            price_str = "FREE" if model['input_price'] == 0 else f"${model['input_price']}/1M"
            print(f"   {key}. {model['name']}")
        print("-" * 60)
    
    @staticmethod
    def get_model_id(choice: str) -> Optional[str]:
        """Get model ID from user choice number."""
        if choice in VIDEO_MODELS:
            return VIDEO_MODELS[choice]['id']
        # Check if it's a direct model ID
        if '/' in choice:
            return choice
        return None
