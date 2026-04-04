"""
Prefect Trigger Tool — Agent → Prefect Bridge

Allows agents (FloorManager, ResearchHead) to trigger Prefect flow deployments
when they complete work. Used by the autonomous overnight pipeline.
"""

import logging
from typing import Any, Dict, Optional

import httpx

from flows.config import PREFECT_API_URL

logger = logging.getLogger(__name__)


class PrefectTriggerTools:
    """Tools for triggering Prefect flow deployments from agents."""

    def __init__(self, prefect_api_url: str = PREFECT_API_URL):
        self.prefect_api_url = prefect_api_url

    async def trigger_prefect_flow(
        self,
        flow_name: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Trigger a registered Prefect flow deployment via Prefect API.

        Args:
            flow_name: Name of the deployment (e.g., 'weekend-compute')
            parameters: Optional dict of parameters to pass to the flow

        Returns:
            Dictionary with success status and either data or error:
            - success: bool
            - data: Dict with run_id, status, flow_name if successful
            - error: str error message if failed
        """
        if parameters is None:
            parameters = {}

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.prefect_api_url}/deployments/name/{flow_name}/create_flow_run",
                    json={"parameters": parameters}
                )

                if response.status_code == 200:
                    data = response.json()
                    run_id = data.get("id", "unknown")
                    logger.info(f"Prefect flow '{flow_name}' triggered: run_id={run_id}")
                    return {
                        "success": True,
                        "data": {
                            "run_id": run_id,
                            "status": "SCHEDULED",
                            "flow_name": flow_name
                        }
                    }
                elif response.status_code == 404:
                    logger.warning(f"Prefect deployment '{flow_name}' not found")
                    return {
                        "success": False,
                        "error": f"Deployment '{flow_name}' not found. Available deployments: weekend-compute, hmm-retraining, research-synthesis"
                    }
                else:
                    logger.error(f"Prefect API error: {response.status_code} {response.text}")
                    return {
                        "success": False,
                        "error": f"Prefect API error: {response.status_code}"
                    }

        except httpx.TimeoutException:
            logger.error(f"Prefect API timeout for flow '{flow_name}'")
            return {"success": False, "error": "Prefect API timeout"}
        except Exception as e:
            logger.error(f"Prefect trigger error: {e}")
            return {"success": False, "error": str(e)}
