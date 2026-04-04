"""
Fortnight Accumulator
=====================

Fortnight Accumulation step (18:00 GMT).
Updates 14-day rolling DPR data and writes fortnight performance file
to cold storage.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

from src.router.dpr_scoring_engine import DprScore

logger = logging.getLogger(__name__)


@dataclass
class FortnightResult:
    """Result of Fortnight Accumulation step."""
    fortnight_stats: Dict[str, Any]
    file_path: str
    bots_scored: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fortnight_stats": self.fortnight_stats,
            "file_path": self.file_path,
            "bots_scored": self.bots_scored,
        }


class FortnightAccumulator:
    """
    Fortnight Accumulation step (18:00 GMT).

    Updates 14-day rolling DPR data and writes fortnight performance file
    to cold storage.
    """

    FORTNIGHT_DAYS = 14

    def __init__(self):
        # In-memory storage for 14-day rolling DPR data
        # Would be persisted to database in production
        self._rolling_dpr: Dict[str, List[Dict[str, Any]]] = {}
        logger.info("FortnightAccumulator initialized")

    async def run(self, dpr_scores: List[DprScore]) -> FortnightResult:
        """
        Accumulate DPR data and write to cold storage.

        Args:
            dpr_scores: List of DprScore objects from DPR Update step

        Returns:
            FortnightResult with statistics and file path
        """
        # Update rolling DPR data
        await self._update_rolling_dpr(dpr_scores)

        # Compute fortnight statistics
        fortnight_stats = await self._compute_fortnight_stats(dpr_scores)

        # Write to cold storage
        file_path = await self._write_fortnight_file(fortnight_stats)

        result = FortnightResult(
            fortnight_stats=fortnight_stats,
            file_path=file_path,
            bots_scored=len(dpr_scores),
        )

        logger.info(
            f"Fortnight Accumulation complete: {len(dpr_scores)} bots scored, "
            f"file written to {file_path}"
        )

        return result

    async def _update_rolling_dpr(self, dpr_scores: List[DprScore]) -> None:
        """
        Update rolling DPR data with current session scores.

        Args:
            dpr_scores: Current session DPR scores
        """
        now = datetime.now(timezone.utc)
        date_str = now.strftime("%Y-%m-%d")

        for score in dpr_scores:
            if score.bot_id not in self._rolling_dpr:
                self._rolling_dpr[score.bot_id] = []

            # Add current score to rolling data
            self._rolling_dpr[score.bot_id].append({
                "date": date_str,
                "composite_score": score.composite_score,
                "tier": score.tier,
                "session_specialist": score.session_specialist,
                "session_concern": score.session_concern,
            })

            # Keep only last FORTNIGHT_DAYS entries
            self._rolling_dpr[score.bot_id] = \
                self._rolling_dpr[score.bot_id][-self.FORTNIGHT_DAYS:]

    async def _compute_fortnight_stats(
        self,
        dpr_scores: List[DprScore]
    ) -> Dict[str, Any]:
        """
        Compute fortnight statistics from rolling DPR data.

        Args:
            dpr_scores: Current session DPR scores

        Returns:
            Dictionary with fortnight statistics
        """
        now = datetime.now(timezone.utc)
        date_str = now.strftime("%Y-%m-%d")

        bot_stats = {}
        for bot_id, history in self._rolling_dpr.items():
            if not history:
                continue

            # Calculate average DPR over fortnight
            scores = [h["composite_score"] for h in history]
            avg_score = sum(scores) / len(scores) if scores else 0.0

            # Count specialist/ concern occurrences
            specialist_count = sum(1 for h in history if h.get("session_specialist", False))
            concern_count = sum(1 for h in history if h.get("session_concern", False))

            # Tier distribution
            tier_counts = {"T1": 0, "T2": 0, "T3": 0}
            for h in history:
                tier = h.get("tier", "T3")
                if tier in tier_counts:
                    tier_counts[tier] += 1

            bot_stats[bot_id] = {
                "avg_composite_score": round(avg_score, 2),
                "sessions_in_fortnight": len(history),
                "specialist_sessions": specialist_count,
                "concern_sessions": concern_count,
                "tier_distribution": tier_counts,
                "current_tier": history[-1].get("tier", "T3") if history else "T3",
            }

        return {
            "date": date_str,
            "fortnight_days": self.FORTNIGHT_DAYS,
            "bots_tracked": len(bot_stats),
            "bot_stats": bot_stats,
            "generated_at": now.isoformat(),
        }

    async def _write_fortnight_file(self, stats: Dict[str, Any]) -> str:
        """
        Write fortnight performance file to cold storage.

        Args:
            stats: Fortnight statistics

        Returns:
            File path where data was written
        """
        from src.router.cold_storage_writer import get_cold_storage_writer

        writer = get_cold_storage_writer()
        now = datetime.now(timezone.utc)
        date_str = now.strftime("%Y-%m-%d")
        file_key = f"fortnight/{date_str}_fortnight_performance.json"

        await writer.write(file_key, stats)

        logger.info(f"Fortnight performance file written: {file_key}")
        return file_key


# ============= Singleton Factory =============
_fortnight_accumulator: Optional[FortnightAccumulator] = None


def get_fortnight_accumulator() -> FortnightAccumulator:
    """Get singleton instance of FortnightAccumulator."""
    global _fortnight_accumulator
    if _fortnight_accumulator is None:
        _fortnight_accumulator = FortnightAccumulator()
    return _fortnight_accumulator
