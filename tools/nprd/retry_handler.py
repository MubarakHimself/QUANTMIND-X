"""
Retry Handler - Exponential backoff with chunk-level recovery
"""
import time
import random
import json
from pathlib import Path
from typing import Callable, Any, Optional, List
from functools import wraps


def retry_with_backoff(
    func: Callable,
    max_retries: int = 3,
    base_delay: float = 2.0,
    max_delay: float = 60.0
) -> Any:
    """
    Retry function with exponential backoff + jitter.
    
    Delays: 2s, 4s, 8s (with random jitter ±1s)
    """
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            last_exception = e
            if attempt == max_retries - 1:
                raise
            
            delay = min(base_delay * (2 ** attempt), max_delay)
            jitter = random.uniform(-1, 1)
            wait_time = max(1, delay + jitter)
            
            print(f"⚠️  Attempt {attempt + 1} failed: {str(e)[:100]}")
            print(f"⏳ Retrying in {wait_time:.1f}s...")
            time.sleep(wait_time)
    
    raise last_exception


class ChunkRecoveryManager:
    """Manages chunk-level processing with crash recovery."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.failed_log = self.output_dir / "failed_chunks.json"
    
    def is_processed(self, chunk_path: Path) -> bool:
        """Check if chunk was already processed."""
        result_file = self.output_dir / f"{chunk_path.stem}.json"
        return result_file.exists()
    
    def save_result(self, chunk_path: Path, result: dict) -> Path:
        """Save chunk result immediately (crash-safe)."""
        result_file = self.output_dir / f"{chunk_path.stem}.json"
        result_file.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        return result_file
    
    def load_result(self, chunk_path: Path) -> Optional[dict]:
        """Load existing result for chunk."""
        result_file = self.output_dir / f"{chunk_path.stem}.json"
        if result_file.exists():
            return json.loads(result_file.read_text())
        return None
    
    def log_failure(self, chunk_path: Path, error: str):
        """Log failed chunk for later retry."""
        failures = []
        if self.failed_log.exists():
            failures = json.loads(self.failed_log.read_text())
        
        failures.append({
            "chunk": str(chunk_path),
            "error": error,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        })
        
        self.failed_log.write_text(json.dumps(failures, indent=2))
    
    def get_failures(self) -> List[dict]:
        """Get list of failed chunks."""
        if self.failed_log.exists():
            return json.loads(self.failed_log.read_text())
        return []
