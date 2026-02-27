"""
Strategies-YT Hooks for video analysis and TRD generation.

These hooks allow integration with external services like:
- Gemini CLI for enhanced video analysis
- QuenCode for code generation
"""

from typing import Dict, Any, Optional


class StrategiesYTHooks:
    """
    Hooks for Strategies-YT tool integration.

    Provides extension points for:
    - Pre-processing video analysis
    - Post-processing TRD generation
    """

    async def pre_video_analysis(self, video_url: str) -> Dict[str, Any]:
        """
        Hook called before video analysis begins.

        Can be used to:
        - Integrate with Gemini CLI for enhanced analysis
        - Fetch video metadata
        - Validate video URL

        Args:
            video_url: URL of the video to analyze

        Returns:
            Dictionary with analysis context or modifications
        """
        # Default implementation: pass through
        return {
            "video_url": video_url,
            "enhanced_analysis": None,
        }

    async def post_trd_generation(self, trd: Dict[str, Any]) -> Dict[str, Any]:
        """
        Hook called after TRD (Trading Requirements Document) is generated.

        Can be used to:
        - Send to external services for enhancement
        - Validate TRD structure
        - Add additional metadata

        Args:
            trd: Generated Trading Requirements Document

        Returns:
            Enhanced or modified TRD
        """
        # Default implementation: pass through
        return trd

    async def on_strategy_extracted(
        self,
        strategy_data: Dict[str, Any],
        source: str
    ) -> Dict[str, Any]:
        """
        Hook called when a strategy is extracted from video/PDF.

        Args:
            strategy_data: Extracted strategy information
            source: Source type (video, pdf, text)

        Returns:
            Enhanced strategy data
        """
        # Default implementation: pass through
        return strategy_data


# Global hook instance
_strategies_yt_hooks: Optional[StrategiesYTHooks] = None


def get_strategies_yt_hooks() -> StrategiesYTHooks:
    """
    Get the global Strategies-YT hooks instance.

    Returns:
        StrategiesYTHooks instance
    """
    global _strategies_yt_hooks
    if _strategies_yt_hooks is None:
        _strategies_yt_hooks = StrategiesYTHooks()
    return _strategies_yt_hooks
