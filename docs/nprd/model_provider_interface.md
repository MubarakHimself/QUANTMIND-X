# ModelProvider Interface

## Overview

The `ModelProvider` interface defines the contract for AI model providers (Gemini CLI, Qwen-VL) that analyze video content to generate timeline outputs with transcripts and visual descriptions.

## Interface Definition

```python
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List
from src.nprd.models import TimelineOutput, RateLimit

class ModelProvider(ABC):
    """Abstract interface for multimodal AI model providers."""
    
    @abstractmethod
    def analyze(self, frames: List[Path], audio: Path, prompt: str) -> TimelineOutput:
        """Analyze video content using multimodal AI."""
        pass
    
    @abstractmethod
    def get_rate_limit(self) -> RateLimit:
        """Get current rate limit status for this provider."""
        pass
```

## Methods

### `analyze(frames, audio, prompt) -> TimelineOutput`

Analyzes video content using multimodal AI to generate a timeline with transcripts and visual descriptions.

**Parameters:**
- `frames` (List[Path]): List of paths to extracted frame images (JPEG format)
- `audio` (Path): Path to extracted audio file (MP3 format)
- `prompt` (str): Analysis prompt instructing the model on extraction methodology

**Returns:**
- `TimelineOutput`: Timeline containing clips with transcripts and descriptions

**Raises:**
- `ProviderError`: If analysis fails (authentication, rate limit, etc.)
- `NetworkError`: If network communication fails
- `ValidationError`: If input data is invalid

**Example:**
```python
frames = [Path("frame_0.jpg"), Path("frame_30.jpg"), Path("frame_60.jpg")]
audio = Path("audio.mp3")
prompt = "Extract verbatim transcripts and objective visual descriptions"

timeline = provider.analyze(frames, audio, prompt)
print(f"Generated {len(timeline.timeline)} clips")
```

### `get_rate_limit() -> RateLimit`

Returns the current rate limit status for this provider.

**Returns:**
- `RateLimit`: Object with `requests_per_day`, `requests_used`, and `window_start`

**Example:**
```python
rate_limit = provider.get_rate_limit()
print(f"Remaining requests: {rate_limit.get_remaining()}")
if rate_limit.is_exceeded():
    print("Rate limit exceeded, switching to alternative provider")
```

## Implementations

### GeminiCLIProvider

Uses Google's Gemini CLI with YOLO mode (bypass permissions).

**Features:**
- Subscription-based (unlimited requests)
- YOLO mode for automated processing
- High-quality multimodal analysis
- Graceful authentication error handling
- Network error detection and reporting
- JSON output parsing with fallback strategies

**Requirements:**
- Gemini CLI installed: `npm install -g @google/gemini-cli`
- API key configured: `gemini auth` or `GEMINI_API_KEY` env var
- YOLO mode enabled for automated processing

**Configuration:**
```python
from src.nprd.providers import GeminiCLIProvider

# Default configuration (YOLO mode enabled)
provider = GeminiCLIProvider()

# Custom configuration
provider = GeminiCLIProvider(
    yolo_mode=True,      # Bypass permission prompts
    api_key="your-key"   # Optional, uses env var if not provided
)
```

**Command Structure:**
The provider builds Gemini CLI commands with the following structure:
```bash
gemini run --yolo "prompt" -f audio.mp3 -f frame_0.jpg -f frame_1.jpg --format json
```

**Error Handling:**
- `AuthenticationError`: Raised when API key is invalid or missing
- `NetworkError`: Raised when network communication fails
- `ValidationError`: Raised when input files are missing or invalid
- `ProviderError`: Raised for other provider-specific errors

**Example Usage:**
```python
from src.nprd.providers import GeminiCLIProvider, AuthenticationError
from pathlib import Path

provider = GeminiCLIProvider(yolo_mode=True)

try:
    frames = [Path("frame_0.jpg"), Path("frame_30.jpg")]
    audio = Path("audio.mp3")
    prompt = "Extract verbatim transcripts and objective visual descriptions"
    
    timeline = provider.analyze(frames, audio, prompt)
    print(f"Successfully analyzed {len(timeline.timeline)} clips")
    
except AuthenticationError as e:
    print(f"Authentication failed: {e}")
    print("Please run 'gemini auth' or set GEMINI_API_KEY")
    
except NetworkError as e:
    print(f"Network error: {e}")
    
except ValidationError as e:
    print(f"Invalid input: {e}")
```

**Rate Limit Tracking:**
```python
provider = GeminiCLIProvider()

# Check rate limit (unlimited for subscription)
rate_limit = provider.get_rate_limit()
print(f"Requests per day: {rate_limit.requests_per_day}")  # None (unlimited)
print(f"Requests used: {rate_limit.requests_used}")

# Rate limit increments automatically after each analyze() call
timeline = provider.analyze(frames, audio, prompt)
print(f"Requests used: {provider.get_rate_limit().requests_used}")  # Incremented
```

**JSON Output Parsing:**
The provider handles multiple JSON output formats from Gemini CLI:
- Structured timeline with `timeline` key
- Alternative formats with `clips` or `segments` keys
- Fallback to creating clips from unstructured text

**Implementation Status:** ✅ Complete (Task 3.2)

### QwenVLProvider

Uses Qwen-VL API in headless mode (no GUI).

**Features:**
- Free tier: 2000 requests/day
- Headless mode for server environments
- OpenAI-compatible API

**Configuration:**
```python
provider = QwenVLProvider(api_key="sk-...", headless=True)
```

**Implementation Status:** 🔄 Pending (Task 3.3)

## Design Philosophy

The ModelProvider interface follows the NPRD "dumb extraction" philosophy:

1. **No Interpretation**: Extract only observable facts
2. **Verbatim Transcripts**: Word-for-word, no summarization
3. **Objective Descriptions**: Observable visual content only
4. **Domain-Agnostic**: No trading knowledge or strategy inference

## Rate Limiting

The interface includes rate limit tracking to:
- Prevent quota exhaustion
- Enable automatic provider switching
- Track usage across time windows

**Rate Limit Model:**
```python
@dataclass
class RateLimit:
    requests_per_day: Optional[int]  # None = unlimited
    requests_used: int = 0
    window_start: Optional[datetime] = None
```

## Testing

The interface includes comprehensive unit tests:

```bash
# Run all provider tests
python -m pytest tests/nprd/test_providers.py -v

# Run specific test class
python -m pytest tests/nprd/test_providers.py::TestModelProviderInterface -v
```

**Test Coverage:**
- Abstract interface validation
- Method signature verification
- Return type validation
- Rate limit integration
- Contract compliance

## Usage Example

```python
from src.nprd import ModelProvider, RateLimit
from pathlib import Path

# Create provider (implementation-specific)
provider = GeminiCLIProvider(yolo_mode=True)

# Check rate limit before processing
rate_limit = provider.get_rate_limit()
if rate_limit.is_exceeded():
    print("Switching to alternative provider")
    provider = QwenVLProvider(api_key="sk-...")

# Analyze video content
frames = [Path(f"frame_{i}.jpg") for i in range(10)]
audio = Path("audio.mp3")
prompt = "Extract verbatim transcripts and objective visual descriptions"

timeline = provider.analyze(frames, audio, prompt)

# Process results
for clip in timeline.timeline:
    print(f"Clip {clip.clip_id}: {clip.timestamp_start} - {clip.timestamp_end}")
    print(f"Transcript: {clip.transcript}")
    print(f"Visual: {clip.visual_description}")
```

## Next Steps

1. ~~Implement `GeminiCLIProvider` (Task 3.2)~~ ✅ Complete
2. Implement `QwenVLProvider` (Task 3.3)
3. Add provider-specific error handling ✅ Complete for Gemini
4. Integrate with job queue system
5. Add automatic provider fallback logic

## References

- Requirements: 18.1, 18.3, 18.8 (Gemini CLI YOLO Mode)
- Requirements: 7.1 (Rate Limiting for API Calls)
- Design Document: Section 1.5 (Model Provider Interface)
- Tasks: 3.1 (Create abstract ModelProvider interface) ✅
- Tasks: 3.2 (Implement GeminiCLIProvider with YOLO mode) ✅
