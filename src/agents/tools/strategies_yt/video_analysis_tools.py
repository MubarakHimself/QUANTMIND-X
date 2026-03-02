"""
Video Analysis Tools

Tools for analyzing trading strategy videos and extracting strategy elements.
Integrates with the strategies-yt pipeline to process educational content
and extract structured trading requirements.

These tools work with the video-to-TRD pipeline:
1. Analyze video for strategy elements
2. Extract indicators and their settings
3. Extract entry/exit rules
4. Extract risk management parameters
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from enum import Enum
import re

# Import video_ingest processor
try:
    from src.video_ingest.processor import VideoIngestProcessor
    from src.video_ingest.models import JobOptions
    VIDEO_INGEST_AVAILABLE = True
except ImportError:
    VIDEO_INGEST_AVAILABLE = False
    logger.warning("video_ingest module not available, using mock analysis")

logger = logging.getLogger(__name__)


# =============================================================================
# PATH CONSTANTS
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
STRATEGIES_YT_DIR = PROJECT_ROOT / "strategies-yt"
PROMPTS_DIR = STRATEGIES_YT_DIR / "prompts"


# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

class ClipType(Enum):
    """Types of clips detected in video analysis."""
    STRATEGY_OVERVIEW = "strategy_overview"
    INDICATOR_EXPLANATION = "indicator_explanation"
    ENTRY_RULE = "entry_rule"
    EXIT_RULE = "exit_rule"
    RISK_MANAGEMENT = "risk_management"
    LIVE_EXAMPLE = "live_example"
    CHART_ANALYSIS = "chart_analysis"


@dataclass
class VideoClip:
    """
    Represents a clip extracted from video analysis.

    Contains visual descriptions, transcript, OCR content, and metadata.
    """
    clip_id: int
    timestamp_start: str  # Format: "MM:SS"
    timestamp_end: str
    speaker: str
    visual_description: str
    transcript: str
    ocr_content: List[str] = field(default_factory=list)
    slide_detected: bool = False
    clip_type: ClipType = ClipType.STRATEGY_OVERVIEW
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['clip_type'] = self.clip_type.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VideoClip":
        if 'clip_type' in data and isinstance(data['clip_type'], str):
            data['clip_type'] = ClipType(data['clip_type'])
        return cls(**data)


@dataclass
class VideoAnalysisResult:
    """
    Complete video analysis result.

    Contains all clips, metadata, and extracted strategy elements.
    """
    video_id: str
    title: str
    total_duration: str
    speakers: Dict[str, str]
    timeline: List[VideoClip] = field(default_factory=list)
    extracted_elements: Dict[str, Any] = field(default_factory=dict)
    confidence_score: float = 0.0
    analyzed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VideoAnalysisResult":
        timeline = [
            VideoClip.from_dict(clip) if isinstance(clip, dict) else clip
            for clip in data.pop('timeline', [])
        ]
        return cls(timeline=timeline, **data)


@dataclass
class IndicatorExtraction:
    """Extracted indicator with settings."""
    name: str
    type: str  # trend, momentum, volatility, volume
    parameters: Dict[str, Any] = field(default_factory=dict)
    usage: str = ""
    chart_examples: List[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class RuleExtraction:
    """Extracted trading rule (entry or exit)."""
    rule_type: str  # entry, exit, filter
    name: str
    conditions: List[str] = field(default_factory=list)
    order_type: str = "market"  # market, limit, stop
    confirmation_required: bool = False
    timeframe: str = ""
    confidence: float = 0.0


@dataclass
class RiskParameterExtraction:
    """Extracted risk management parameters."""
    stop_loss_type: str = "fixed"  # fixed, trailing, atr, swing
    stop_loss_value: float = 0.0
    take_profit_type: str = "fixed"  # fixed, risk_reward, fibonacci
    take_profit_value: float = 0.0
    max_risk_percent: float = 2.0
    max_daily_loss: float = 5.0
    position_sizing_method: str = "kelly"
    kelly_fraction: float = 0.25
    confidence: float = 0.0


# =============================================================================
# VIDEO ANALYSIS FUNCTIONS
# =============================================================================

async def analyze_trading_video(
    video_id: str,
    video_url: Optional[str] = None,
    analysis_depth: str = "standard"
) -> Dict[str, Any]:
    """
    Analyze trading video and extract strategy elements.

    Uses the video_ingest processor to process YouTube videos with Gemini CLI
    or Qwen CLI for AI-powered analysis.

    Args:
        video_id: YouTube video ID or identifier
        video_url: Optional direct URL to video
        analysis_depth: "quick", "standard", or "deep"

    Returns:
        Dictionary containing:
        - success: Analysis status
        - analysis: VideoAnalysisResult with timeline and elements
        - summary: Quick summary of findings
    """
    logger.info(f"Analyzing trading video: {video_id} (depth: {analysis_depth})")

    # Build the full URL if only ID is provided
    if not video_url:
        video_url = f"https://www.youtube.com/watch?v={video_id}"

    try:
        # Use video_ingest processor if available
        if VIDEO_INGEST_AVAILABLE:
            logger.info(f"Using video_ingest processor for {video_url}")

            # Create processor with options based on analysis depth
            options = JobOptions()

            # Adjust frame interval based on analysis depth
            if analysis_depth == "quick":
                options.frame_interval = 60  # Every 60 seconds
            elif analysis_depth == "deep":
                options.frame_interval = 10  # Every 10 seconds
            else:  # standard
                options.frame_interval = 30  # Every 30 seconds

            # Process the video
            processor = VideoIngestProcessor()
            result = processor.process(url=video_url, job_id=f"agent_{video_id}", options=options)

            # Convert ProcessingResult to VideoAnalysisResult format
            timeline = result.timeline

            # Map timeline clips to VideoClip format
            video_clips = []
            for i, clip in enumerate(timeline.timeline):
                # Determine clip type from description
                clip_type = ClipType.OTHER
                desc_lower = (clip.description or "").lower()
                if any(word in desc_lower for word in ["entry", "buy", "sell", "signal"]):
                    clip_type = ClipType.ENTRY_RULE
                elif any(word in desc_lower for word in ["exit", "stop", "take profit", "tp"]):
                    clip_type = ClipType.EXIT_RULE
                elif any(word in desc_lower for word in ["indicator", "ma", "ema", "rsi", "macd"]):
                    clip_type = ClipType.INDICATOR_EXPLANATION
                elif any(word in desc_lower for word in ["risk", "position", "lot", "size"]):
                    clip_type = ClipType.RISK_MANAGEMENT

                video_clips.append(VideoClip(
                    clip_id=i + 1,
                    timestamp_start=clip.timestamp_start,
                    timestamp_end=clip.timestamp_end,
                    speaker=clip.speaker or "SPEAKER_01",
                    visual_description=clip.description or "",
                    transcript=clip.transcript or "",
                    ocr_content=clip.ocr_content or [],
                    clip_type=clip_type,
                    confidence=clip.confidence_score or 0.8
                ))

            # Format duration
            total_seconds = timeline.duration_seconds
            minutes, seconds = divmod(total_seconds, 60)
            hours, minutes = divmod(minutes, 60)
            if hours > 0:
                total_duration = f"{hours}:{minutes:02d}:{seconds:02d}"
            else:
                total_duration = f"{minutes}:{seconds:02d}"

            # Create VideoAnalysisResult
            analysis = VideoAnalysisResult(
                video_id=video_id,
                title=timeline.title or f"Video {video_id}",
                total_duration=total_duration,
                speakers={"SPEAKER_01": "Speaker"},
                timeline=video_clips,
                confidence_score=0.85  # Default since provider doesn't return score
            )
            analysis.extracted_elements = await _extract_all_elements(analysis.timeline)
            summary = _generate_analysis_summary(analysis)

            return {
                "success": True,
                "analysis": analysis.to_dict(),
                "summary": summary,
                "video_id": video_id,
                "provider_used": result.provider_used,
                "processing_time": result.processing_time_seconds,
                "analyzed_at": datetime.now(timezone.utc).isoformat()
            }
        else:
            # Fallback to mock if video_ingest not available
            logger.warning("video_ingest not available, using mock analysis")
            raise Exception("video_ingest module not available")

    except Exception as e:
        logger.error(f"Failed to analyze video: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "video_id": video_id
        }


async def analyze_playlist(
    playlist_url: str,
    max_videos: int = 10,
    analysis_depth: str = "standard"
) -> Dict[str, Any]:
    """
    Analyze a YouTube playlist and extract strategy elements from multiple videos.

    Uses the video_ingest processor to batch process playlist videos with Gemini CLI
    or Qwen CLI for AI-powered analysis.

    Args:
        playlist_url: YouTube playlist URL
        max_videos: Maximum number of videos to process (default: 10)
        analysis_depth: "quick", "standard", or "deep"

    Returns:
        Dictionary containing:
        - success: Analysis status
        - videos: Array of video analysis results
        - summary: Summary of all strategies found
        - total_processed: Number of videos processed
    """
    logger.info(f"Analyzing playlist: {playlist_url} (max: {max_videos}, depth: {analysis_depth})")

    try:
        if not VIDEO_INGEST_AVAILABLE:
            raise Exception("video_ingest module not available")

        # Create processor
        processor = VideoIngestProcessor()
        options = JobOptions()

        # Adjust based on analysis depth
        if analysis_depth == "quick":
            options.frame_interval = 60
        elif analysis_depth == "deep":
            options.frame_interval = 10
        else:
            options.frame_interval = 30

        # Process playlist
        results = processor.process_playlist(
            playlist_url=playlist_url,
            max_videos=max_videos,
            options=options
        )

        # Convert each result to VideoAnalysisResult format
        videos = []
        all_indicators = []
        all_entries = []
        all_exits = []
        all_risk = []

        for i, result in enumerate(results):
            timeline = result.timeline

            # Map clips
            video_clips = []
            for j, clip in enumerate(timeline.timeline):
                clip_type = ClipType.OTHER
                desc_lower = (clip.description or "").lower()
                if any(word in desc_lower for word in ["entry", "buy", "sell", "signal"]):
                    clip_type = ClipType.ENTRY_RULE
                    all_entries.append(clip.transcript or "")
                elif any(word in desc_lower for word in ["exit", "stop", "take profit", "tp"]):
                    clip_type = ClipType.EXIT_RULE
                    all_exits.append(clip.transcript or "")
                elif any(word in desc_lower for word in ["indicator", "ma", "ema", "rsi", "macd"]):
                    clip_type = ClipType.INDICATOR_EXPLANATION
                    all_indicators.append(clip.description or "")
                elif any(word in desc_lower for word in ["risk", "position", "lot", "size"]):
                    clip_type = ClipType.RISK_MANAGEMENT
                    all_risk.append(clip.transcript or "")

                video_clips.append(VideoClip(
                    clip_id=j + 1,
                    timestamp_start=clip.timestamp_start,
                    timestamp_end=clip.timestamp_end,
                    speaker=clip.speaker or "SPEAKER_01",
                    visual_description=clip.description or "",
                    transcript=clip.transcript or "",
                    ocr_content=clip.ocr_content or [],
                    clip_type=clip_type,
                    confidence=clip.confidence_score or 0.8
                ))

            total_seconds = timeline.duration_seconds
            minutes, seconds = divmod(total_seconds, 60)
            hours, minutes = divmod(minutes, 60)
            total_duration = f"{hours}:{minutes:02d}:{seconds:02d}" if hours > 0 else f"{minutes}:{seconds:02d}"

            analysis = VideoAnalysisResult(
                video_id=f"video_{i}",
                title=timeline.title or f"Video {i}",
                total_duration=total_duration,
                speakers={"SPEAKER_01": "Speaker"},
                timeline=video_clips,
                confidence_score=0.85
            )
            analysis.extracted_elements = await _extract_all_elements(analysis.timeline)

            videos.append({
                "index": i,
                "video_id": f"video_{i}",
                "title": timeline.title,
                "analysis": analysis.to_dict(),
                "provider_used": result.provider_used,
                "processing_time": result.processing_time_seconds
            })

        # Generate summary
        summary = f"Processed {len(videos)} videos from playlist. "
        if all_indicators:
            summary += f"Found {len(all_indicators)} indicator explanations. "
        if all_entries:
            summary += f"Found {len(all_entries)} entry rules. "
        if all_exits:
            summary += f"Found {len(all_exits)} exit rules. "
        if all_risk:
            summary += f"Found {len(all_risk)} risk management rules. "

        return {
            "success": True,
            "playlist_url": playlist_url,
            "videos": videos,
            "summary": summary,
            "total_processed": len(videos),
            "analyzed_at": datetime.now(timezone.utc).isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to analyze playlist: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "playlist_url": playlist_url
        }


async def extract_indicators(
    video_id: str,
    timeline: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Extract indicator settings from video analysis.

    Args:
        video_id: Video identifier
        timeline: Optional pre-extracted timeline clips

    Returns:
        Dictionary with extracted indicators and settings
    """
    logger.info(f"Extracting indicators from: {video_id}")

    try:
        # Common trading indicators with parameter patterns
        indicator_patterns = {
            "EMA": {
                "patterns": [r"EMA\s*(\d+)", r"Exponential.*?(\d+)"],
                "params": ["period"]
            },
            "SMA": {
                "patterns": [r"SMA\s*(\d+)", r"Simple.*?(\d+)"],
                "params": ["period"]
            },
            "RSI": {
                "patterns": [r"RSI\s*(\d+)", r"Relative Strength.*?(\d+)"],
                "params": ["period", "overbought", "oversold"]
            },
            "MACD": {
                "patterns": [r"MACD"],
                "params": ["fast", "slow", "signal"]
            },
            "ATR": {
                "patterns": [r"ATR\s*(\d+)", r"Average True Range.*?(\d+)"],
                "params": ["period", "multiplier"]
            },
            "Bollinger Bands": {
                "patterns": [r"BB\s*(\d+)", r"Bollinger.*?(\d+)"],
                "params": ["period", "std_dev"]
            },
            "Stochastic": {
                "patterns": [r"Stoch"],
                "params": ["k_period", "d_period", "smooth"]
            }
        }

        extracted_indicators: List[IndicatorExtraction] = []

        # Process timeline or load from video
        clips = None
        if timeline:
            clips = [VideoClip.from_dict(c) if isinstance(c, dict) else c for c in timeline]
        else:
            # Load from video analysis
            result = await analyze_trading_video(video_id)
            if result.get("success"):
                analysis = VideoAnalysisResult.from_dict(result["analysis"])
                clips = analysis.timeline

        if not clips:
            return {
                "success": False,
                "error": "No timeline data available",
                "indicators": []
            }

        # Extract indicators from clips
        for clip in clips:
            if clip.clip_type in [ClipType.INDICATOR_EXPLANATION, ClipType.CHART_ANALYSIS]:
                # Combine all text sources
                text_sources = [clip.transcript] + clip.ocr_content
                combined_text = " ".join(text_sources).lower()

                # Check for indicator patterns
                for indicator_name, patterns in indicator_patterns.items():
                    for pattern in patterns['patterns']:
                        matches = re.findall(pattern, combined_text, re.IGNORECASE)
                        if matches:
                            # Extract or infer parameters
                            params = {}
                            if indicator_name in ["EMA", "SMA", "RSI", "ATR", "Bollinger Bands"]:
                                try:
                                    params["period"] = int(matches[0])
                                except (ValueError, IndexError):
                                    params["period"] = 20 if indicator_name != "RSI" else 14

                            # Check if already extracted
                            existing = next((i for i in extracted_indicators if i.name == indicator_name), None)
                            if existing:
                                # Merge parameters
                                existing.parameters.update(params)
                                existing.confidence = max(existing.confidence, clip.confidence)
                            else:
                                extracted_indicators.append(IndicatorExtraction(
                                    name=indicator_name,
                                    type=_classify_indicator_type(indicator_name),
                                    parameters=params,
                                    usage=clip.transcript[:200],
                                    confidence=clip.confidence
                                ))

        return {
            "success": True,
            "indicators": [asdict(i) for i in extracted_indicators],
            "count": len(extracted_indicators),
            "video_id": video_id
        }

    except Exception as e:
        logger.error(f"Failed to extract indicators: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "indicators": []
        }


async def extract_entry_rules(
    video_id: str,
    timeline: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Extract entry conditions from video analysis.

    Args:
        video_id: Video identifier
        timeline: Optional pre-extracted timeline clips

    Returns:
        Dictionary with extracted entry rules
    """
    logger.info(f"Extracting entry rules from: {video_id}")

    try:
        clips = None
        if timeline:
            clips = [VideoClip.from_dict(c) if isinstance(c, dict) else c for c in timeline]
        else:
            result = await analyze_trading_video(video_id)
            if result.get("success"):
                analysis = VideoAnalysisResult.from_dict(result["analysis"])
                clips = analysis.timeline

        if not clips:
            return {
                "success": False,
                "error": "No timeline data available",
                "entry_rules": []
            }

        entry_rules: List[RuleExtraction] = []

        # Entry rule keywords
        entry_keywords = [
            "entry", "enter", "buy", "sell", "long", "short",
            "setup", "signal", "trigger", "condition"
        ]

        for clip in clips:
            if clip.clip_type == ClipType.ENTRY_RULE:
                text = clip.transcript.lower()

                # Check if this clip discusses entry rules
                if any(kw in text for kw in entry_keywords):
                    # Extract conditions from transcript
                    conditions = _extract_conditions_from_text(clip.transcript)

                    # Determine order type
                    order_type = "market"
                    if "limit" in text:
                        order_type = "limit"
                    elif "stop" in text:
                        order_type = "stop"

                    # Check for confirmation requirements
                    confirmation = any(kw in text for kw in [
                        "confirm", "wait for", "must", "require"
                    ])

                    # Extract timeframe
                    timeframe = _extract_timeframe(text)

                    rule = RuleExtraction(
                        rule_type="entry",
                        name=f"Entry Rule {len(entry_rules) + 1}",
                        conditions=conditions,
                        order_type=order_type,
                        confirmation_required=confirmation,
                        timeframe=timeframe,
                        confidence=clip.confidence
                    )

                    entry_rules.append(rule)

        return {
            "success": True,
            "entry_rules": [asdict(r) for r in entry_rules],
            "count": len(entry_rules),
            "video_id": video_id
        }

    except Exception as e:
        logger.error(f"Failed to extract entry rules: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "entry_rules": []
        }


async def extract_exit_rules(
    video_id: str,
    timeline: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Extract exit conditions from video analysis.

    Args:
        video_id: Video identifier
        timeline: Optional pre-extracted timeline clips

    Returns:
        Dictionary with extracted exit rules
    """
    logger.info(f"Extracting exit rules from: {video_id}")

    try:
        clips = None
        if timeline:
            clips = [VideoClip.from_dict(c) if isinstance(c, dict) else c for c in timeline]
        else:
            result = await analyze_trading_video(video_id)
            if result.get("success"):
                analysis = VideoAnalysisResult.from_dict(result["analysis"])
                clips = analysis.timeline

        if not clips:
            return {
                "success": False,
                "error": "No timeline data available",
                "exit_rules": []
            }

        exit_rules: List[RuleExtraction] = []

        # Exit rule keywords
        exit_keywords = [
            "exit", "close", "take profit", "tp", "stop loss", "sl",
            "trailing", "breakeven", "target"
        ]

        for clip in clips:
            if clip.clip_type == ClipType.EXIT_RULE:
                text = clip.transcript.lower()

                # Check if this clip discusses exit rules
                if any(kw in text for kw in exit_keywords):
                    # Extract conditions
                    conditions = _extract_conditions_from_text(clip.transcript)

                    # Extract stop loss and take profit values
                    sl_value = _extract_pip_value(text, ["stop loss", "sl", "stop"])
                    tp_value = _extract_pip_value(text, ["take profit", "tp", "target"])

                    rule = RuleExtraction(
                        rule_type="exit",
                        name=f"Exit Rule {len(exit_rules) + 1}",
                        conditions=conditions,
                        confidence=clip.confidence
                    )

                    # Add SL/TP as conditions if found
                    if sl_value > 0:
                        rule.conditions.append(f"Stop Loss: {sl_value} pips")
                    if tp_value > 0:
                        rule.conditions.append(f"Take Profit: {tp_value} pips")

                    exit_rules.append(rule)

        return {
            "success": True,
            "exit_rules": [asdict(r) for r in exit_rules],
            "count": len(exit_rules),
            "video_id": video_id
        }

    except Exception as e:
        logger.error(f"Failed to extract exit rules: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "exit_rules": []
        }


async def extract_risk_parameters(
    video_id: str,
    timeline: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Extract risk management parameters from video analysis.

    Args:
        video_id: Video identifier
        timeline: Optional pre-extracted timeline clips

    Returns:
        Dictionary with extracted risk parameters
    """
    logger.info(f"Extracting risk parameters from: {video_id}")

    try:
        clips = None
        if timeline:
            clips = [VideoClip.from_dict(c) if isinstance(c, dict) else c for c in timeline]
        else:
            result = await analyze_trading_video(video_id)
            if result.get("success"):
                analysis = VideoAnalysisResult.from_dict(result["analysis"])
                clips = analysis.timeline

        if not clips:
            return {
                "success": False,
                "error": "No timeline data available",
                "risk_parameters": {}
            }

        risk_params = RiskParameterExtraction()

        # Analyze all clips for risk mentions
        all_text = " ".join([
            clip.transcript + " " + " ".join(clip.ocr_content)
            for clip in clips
        ]).lower()

        # Extract stop loss
        sl_keywords = ["stop loss", "sl", "stop"]
        for kw in sl_keywords:
            value = _extract_pip_value(all_text, [kw])
            if value > 0:
                risk_params.stop_loss_value = value
                if "atr" in all_text:
                    risk_params.stop_loss_type = "atr"
                elif "trailing" in all_text:
                    risk_params.stop_loss_type = "trailing"
                break

        # Extract take profit
        tp_keywords = ["take profit", "tp", "target", "profit target"]
        for kw in tp_keywords:
            value = _extract_pip_value(all_text, [kw])
            if value > 0:
                risk_params.take_profit_value = value
                if "risk:reward" in all_text or "rr" in all_text:
                    risk_params.take_profit_type = "risk_reward"
                elif "fibonacci" in all_text or "fib" in all_text:
                    risk_params.take_profit_type = "fibonacci"
                break

        # Extract position sizing
        if "kelly" in all_text:
            risk_params.position_sizing_method = "kelly"
            # Try to extract Kelly fraction
            kelly_match = re.search(r'kelly\s*(?:fraction\s*)?(\d+(?:\.\d+)?)', all_text)
            if kelly_match:
                risk_params.kelly_fraction = float(kelly_match.group(1)) / 100 if float(kelly_match.group(1)) > 1 else float(kelly_match.group(1))
        elif "fixed" in all_text:
            risk_params.position_sizing_method = "fixed"
        elif "percent" in all_text or "%" in all_text:
            risk_params.position_sizing_method = "percent"

        # Extract risk percentages
        risk_match = re.search(r'risk\s*(?:per\s*trade)?\s*(\d+(?:\.\d+)?)\s*%', all_text)
        if risk_match:
            risk_params.max_risk_percent = float(risk_match.group(1))

        daily_match = re.search(r'(?:daily\s*loss|max\s*daily)\s*(\d+(?:\.\d+)?)\s*%', all_text)
        if daily_match:
            risk_params.max_daily_loss = float(daily_match.group(1))

        # Calculate confidence based on findings
        confidence = 0.5
        if risk_params.stop_loss_value > 0:
            confidence += 0.2
        if risk_params.take_profit_value > 0:
            confidence += 0.2
        if risk_params.kelly_fraction > 0:
            confidence += 0.1
        risk_params.confidence = min(confidence, 1.0)

        return {
            "success": True,
            "risk_parameters": asdict(risk_params),
            "video_id": video_id
        }

    except Exception as e:
        logger.error(f"Failed to extract risk parameters: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "risk_parameters": {}
        }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

async def _extract_all_elements(timeline: List[VideoClip]) -> Dict[str, Any]:
    """Extract all strategy elements from timeline."""
    elements = {
        "indicators": [],
        "entry_rules": [],
        "exit_rules": [],
        "risk_parameters": {},
        "timeframes": set(),
        "sessions": set(),
        "symbols": set()
    }

    for clip in timeline:
        # Extract timeframes
        tf = _extract_timeframe(clip.transcript.lower())
        if tf:
            elements["timeframes"].add(tf)

        # Extract sessions
        for session in ["london", "new_york", "asian", "overlap"]:
            if session in clip.transcript.lower():
                elements["sessions"].add(session.upper())

        # Extract symbols
        symbol_match = re.search(r'\b[A-Z]{6}\b', clip.transcript)
        if symbol_match:
            elements["symbols"].add(symbol_match.group(0))

    return {
        **elements,
        "timeframes": list(elements["timeframes"]),
        "sessions": list(elements["sessions"]),
        "symbols": list(elements["symbols"])
    }


def _classify_indicator_type(indicator_name: str) -> str:
    """Classify indicator by type."""
    trend_indicators = ["EMA", "SMA", "MACD"]
    momentum_indicators = ["RSI", "Stochastic"]
    volatility_indicators = ["ATR", "Bollinger Bands"]

    name = indicator_name.upper()
    if any(ind in name for ind in trend_indicators):
        return "trend"
    elif any(ind in name for ind in momentum_indicators):
        return "momentum"
    elif any(ind in name for ind in volatility_indicators):
        return "volatility"
    return "other"


def _extract_conditions_from_text(text: str) -> List[str]:
    """Extract trading conditions from transcript text."""
    conditions = []

    # Split by common separators
    sentences = re.split(r'[.;,\n]', text)

    # Filter for condition-like sentences
    condition_keywords = [
        "when", "if", "once", "after", "before", "wait for",
        "must", "should", "only", "confirm", "verify"
    ]

    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 10 and any(kw in sentence.lower() for kw in condition_keywords):
            conditions.append(sentence)

    return conditions[:10]  # Limit to top 10 conditions


def _extract_timeframe(text: str) -> str:
    """Extract timeframe from text."""
    timeframe_patterns = [
        r'M(\d+)', r'H(\d+)', r'D(\d+)', r'W(\d+)',
        r'(\d+)\s*min', r'(\d+)\s*hour', r'(\d+)\s*day'
    ]

    for pattern in timeframe_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1)
            if 'min' in pattern.lower():
                return f"M{value}"
            elif 'hour' in pattern.lower():
                return f"H{value}"
            elif 'day' in pattern.lower():
                return f"D{value}"
            return match.group(0)

    return ""


def _extract_pip_value(text: str, keywords: List[str]) -> float:
    """Extract pip value from text given keywords."""
    for kw in keywords:
        # Look for pattern like "50 pips", "50 pip", "50pips"
        pattern = rf'{kw}\s*(?:of\s*)?(?:\d+\s*(?:to\s*)?)?(\d+(?:\.\d+)?)\s*pips?'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except (ValueError, IndexError):
                pass

        # Also look for pattern before keyword
        pattern = r'(\d+(?:\.\d+)?)\s*pips?\s*(?:of\s*)?' + re.escape(kw)
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except (ValueError, IndexError):
                pass

    return 0.0


def _generate_analysis_summary(analysis: VideoAnalysisResult) -> Dict[str, Any]:
    """Generate quick summary of analysis results."""
    clip_types = {}
    for clip in analysis.timeline:
        clip_type = clip.clip_type.value
        clip_types[clip_type] = clip_types.get(clip_type, 0) + 1

    return {
        "total_clips": len(analysis.timeline),
        "clip_types": clip_types,
        "duration": analysis.total_duration,
        "speakers": list(analysis.speakers.values()),
        "confidence": analysis.confidence_score
    }


# =============================================================================
# TOOL REGISTRY
# =============================================================================

VIDEO_ANALYSIS_TOOLS = {
    "analyze_trading_video": {
        "function": analyze_trading_video,
        "description": "Analyze trading video and extract strategy elements using AI",
        "parameters": {
            "video_id": {"type": "string", "required": True},
            "video_url": {"type": "string", "required": False},
            "analysis_depth": {"type": "string", "required": False, "default": "standard"}
        }
    },
    "analyze_playlist": {
        "function": analyze_playlist,
        "description": "Analyze YouTube playlist and extract strategy elements from multiple videos",
        "parameters": {
            "playlist_url": {"type": "string", "required": True},
            "max_videos": {"type": "integer", "required": False, "default": 10},
            "analysis_depth": {"type": "string", "required": False, "default": "standard"}
        }
    },
    "extract_indicators": {
        "function": extract_indicators,
        "description": "Extract indicator settings from video analysis",
        "parameters": {
            "video_id": {"type": "string", "required": True},
            "timeline": {"type": "array", "required": False}
        }
    },
    "extract_entry_rules": {
        "function": extract_entry_rules,
        "description": "Extract entry conditions from video analysis",
        "parameters": {
            "video_id": {"type": "string", "required": True},
            "timeline": {"type": "array", "required": False}
        }
    },
    "extract_exit_rules": {
        "function": extract_exit_rules,
        "description": "Extract exit conditions from video analysis",
        "parameters": {
            "video_id": {"type": "string", "required": True},
            "timeline": {"type": "array", "required": False}
        }
    },
    "extract_risk_parameters": {
        "function": extract_risk_parameters,
        "description": "Extract risk management rules from video analysis",
        "parameters": {
            "video_id": {"type": "string", "required": True},
            "timeline": {"type": "array", "required": False}
        }
    }
}


def get_video_analysis_tool(name: str) -> Optional[Dict[str, Any]]:
    """Get a video analysis tool by name."""
    return VIDEO_ANALYSIS_TOOLS.get(name)


def list_video_analysis_tools() -> List[str]:
    """List all available video analysis tools."""
    return list(VIDEO_ANALYSIS_TOOLS.keys())
