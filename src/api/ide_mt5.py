"""
QuantMind IDE MT5 Endpoints

API endpoints for MT5 terminal integration.
"""

import logging
import platform
import subprocess
from fastapi import APIRouter, HTTPException
from typing import Optional, List

from src.api.ide_models import MT5ScanRequest, MT5LaunchRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ide/mt5", tags=["mt5"])


@router.get("/status")
async def get_mt5_status():
    """Get MT5 terminal status."""
    # Mock status - in production, check actual MT5 terminal
    return {
        "connected": False,
        "terminal_path": None,
        "version": None,
        "accounts": [],
    }


@router.post("/scan")
async def scan_mt5_terminals(request: MT5ScanRequest):
    """Scan for MT5 terminals."""
    # Mock scan results
    terminals = []

    if platform.system() == "Windows":
        # Common Windows MT5 installation paths
        default_paths = [
            "C:\\Program Files\\MetaTrader 5",
            "C:\\Program Files (x86)\\MetaTrader 5",
        ]
        custom_paths = request.custom_paths or []

        for path in default_paths + custom_paths:
            terminal_exe = f"{path}\\terminal64.exe"
            terminals.append({
                "path": path,
                "terminal_exe": terminal_exe,
                "exists": False,
            })

    return {
        "terminals": terminals,
        "count": len(terminals),
    }


@router.post("/launch")
async def launch_mt5(request: MT5LaunchRequest):
    """Launch MT5 terminal."""
    try:
        if platform.system() == "Windows":
            cmd = [request.terminal_path]

            if request.login:
                cmd.append(f"--login={request.login}")
            if request.password:
                cmd.append(f"--password={request.password}")
            if request.server:
                cmd.append(f"--server={request.server}")

            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            return {
                "success": True,
                "message": f"Launched MT5 terminal: {request.terminal_path}"
            }
        else:
            raise HTTPException(400, "MT5 launch is only supported on Windows")

    except Exception as e:
        logger.error(f"Error launching MT5: {e}")
        raise HTTPException(500, str(e))
