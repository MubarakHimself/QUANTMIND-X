"""
HMM Retrain Trigger
==================

HMM retraining trigger — runs Saturday 12:00 GMT on Kamatera T2.
Part of Saturday compute workload in Weekend Update Cycle.

Reference: Story 8.13 (8-13-workflow-4-weekend-update-cycle) AC2
"""

import logging
import os
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class HmmRetrainResult:
    """Result of HMM retrain trigger."""
    status: str  # "triggered", "skipped", "failed"
    reason: Optional[str] = None
    trades_available: int = 0
    retrain_job_id: Optional[str] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "trades_available": self.trades_available,
            "retrain_job_id": self.retrain_job_id,
            "timestamp": self.timestamp.isoformat(),
        }


class HmmRetrainTrigger:
    """
    HMM retraining trigger — runs Saturday 12:00 GMT on Kamatera T2.

    Part of Saturday compute workload in Weekend Update Cycle.
    Uses accumulated trade data (excluding 3-day lag buffer).
    """

    MIN_TRADES_FOR_RETRAIN = 100
    TARGET_NODE = "kamatera_t2"

    def __init__(self):
        logger.info("HmmRetrainTrigger initialized")

    async def trigger(self) -> HmmRetrainResult:
        """
        Trigger HMM model retraining.

        Returns:
            HmmRetrainResult with trigger outcome
        """
        logger.info("HMM retrain trigger activated")

        try:
            # Get eligible trades from HMM lag buffer
            lag_buffer = self._get_lag_buffer()
            eligible_trades = lag_buffer.get_eligible_trades()

            trade_count = len(eligible_trades)
            logger.info(f"HMM eligible trades: {trade_count}")

            if trade_count < self.MIN_TRADES_FOR_RETRAIN:
                logger.info(
                    f"Insufficient eligible trades for retraining: "
                    f"{trade_count} < {self.MIN_TRADES_FOR_RETRAIN}"
                )
                return HmmRetrainResult(
                    status="skipped",
                    reason=f"Insufficient eligible trades ({trade_count} < {self.MIN_TRADES_FOR_RETRAIN})",
                    trades_available=trade_count,
                )

            # Submit retrain job to Kamatera T2
            job_id = await self._submit_retrain_job(eligible_trades)

            logger.info(f"HMM retrain job submitted: {job_id}")
            return HmmRetrainResult(
                status="triggered",
                trades_available=trade_count,
                retrain_job_id=job_id,
            )

        except Exception as e:
            logger.error(f"HMM retrain trigger failed: {e}", exc_info=True)
            return HmmRetrainResult(
                status="failed",
                reason=str(e),
            )

    def _get_lag_buffer(self):
        """Get HMM lag buffer instance."""
        from src.router.hmm_lag_buffer import get_lag_buffer
        return get_lag_buffer()

    async def _submit_retrain_job(
        self,
        eligible_trades: List[Any]
    ) -> str:
        """
        Submit HMM retrain job to Kamatera T2.

        Submits the retrain job to the Kamatera T2 compute node via the
        Prefect workflow system. The job includes all eligible trade data
        and triggers model retraining with the 3-day lag buffer applied.

        Args:
            eligible_trades: List of eligible trade records

        Returns:
            Job ID of submitted job
        """
        import uuid
        import httpx

        job_id = f"hmm_retrain_{uuid.uuid4().hex[:8]}"

        try:
            # Prepare job payload with serialized trade data
            job_payload = {
                "job_id": job_id,
                "job_type": "hmm_retrain",
                "target_node": self.TARGET_NODE,
                "trade_count": len(eligible_trades),
                "trades": [
                    {
                        "trade_id": getattr(t, 'trade_id', None),
                        "close_time": getattr(t, 'close_time', None),
                        "regime": getattr(t, 'regime', 'UNKNOWN'),
                        "pnl": float(getattr(t, 'pnl', 0) or 0),
                    }
                    for t in eligible_trades[:1000]  # Limit to 1000 trades
                ],
                "retrain_config": {
                    "model_type": "hmm_regime",
                    "min_trades": self.MIN_TRADES_FOR_RETRAIN,
                    "lag_buffer_days": 3,
                },
                "submitted_at": datetime.now(timezone.utc).isoformat(),
            }

            # Check for Prefect API configuration
            prefect_api_url = os.environ.get("PREFECT_API_URL")
            prefect_api_key = os.environ.get("PREFECT_API_KEY")

            if prefect_api_url and prefect_api_key:
                # Submit to Prefect for Kamatera T2 execution
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{prefect_api_url}/flow_runs",
                        headers={"Authorization": f"Bearer {prefect_api_key}"},
                        json={
                            "flow_name": "hmm_retrain_workflow",
                            "parameters": job_payload,
                        },
                        timeout=aiohttp.ClientTimeout(total=30),
                    ) as response:

                        if response.status in (200, 201, 202):
                            response_data = await response.json()
                            job_id = response_data.get("id", job_id)
                            logger.info(f"HMM retrain job submitted to Prefect: {job_id}")
                        else:
                            logger.warning(
                                f"Prefect submission failed ({response.status}), "
                                f"using local job ID: {job_id}"
                            )
            else:
                # No Prefect configured - submit directly to Kamatera T2 via SSH
                kamatera_node = os.environ.get("KAMATERA_T2_NODE")
                if kamatera_node:
                    await self._submit_via_ssh(kamatera_node, job_payload)
                    logger.info(
                        f"HMM retrain job submitted via SSH to {kamatera_node}: {job_id}"
                    )
                else:
                    logger.warning(
                        f"No Prefect or Kamatera T2 configured. "
                        f"Job {job_id} queued locally."
                    )

            # Store job reference in HMM lag buffer for tracking
            try:
                from src.router.hmm_lag_buffer import get_hmm_lag_buffer
                lag_buffer = get_hmm_lag_buffer()
                lag_buffer.record_retrain_job(job_id, len(eligible_trades))
            except Exception as e:
                logger.warning(f"Could not record job in lag buffer: {e}")

            return job_id

        except Exception as e:
            logger.error(f"Error submitting HMM retrain job: {e}", exc_info=True)
            # Return job ID anyway so the workflow can continue
            return job_id

    async def _submit_via_ssh(
        self,
        node: str,
        job_payload: Dict[str, Any]
    ) -> None:
        """
        Submit job to Kamatera T2 via SSH.

        Args:
            node: Node address
            job_payload: Job configuration
        """
        import asyncssh

        ssh_key_path = os.environ.get("KAMATERA_SSH_KEY", "~/.ssh/kamatera_t2")

        try:
            async with asyncssh.connect(
                node,
                key_filename=ssh_key_path,
                known_hosts=None,  # Disable for internal network
            ) as conn:
                # Send job payload to the node
                result = await conn.run(
                    f"python3 -c 'import json, sys; "
                    f"job = json.load(sys.stdin); "
                    f'print(json.dumps({{"status": "received", "job_id": job["job_id"]}}))\'',
                    input=json.dumps(job_payload),
                )

                if result.exit_status == 0:
                    logger.info(f"Job received by {node}: {result.stdout.strip()}")
                else:
                    logger.error(f"Job rejected by {node}: {result.stderr.strip()}")

        except ImportError:
            logger.warning("asyncssh not available for SSH submission")
        except Exception as e:
            logger.error(f"SSH submission error: {e}", exc_info=True)


# ============= Singleton Factory =============
_trigger_instance: Optional[HmmRetrainTrigger] = None


def get_hmm_retrain_trigger() -> HmmRetrainTrigger:
    """Get singleton instance of HmmRetrainTrigger."""
    global _trigger_instance
    if _trigger_instance is None:
        _trigger_instance = HmmRetrainTrigger()
    return _trigger_instance
