"""
QuantMind IDE MT5 Endpoints

API endpoints for MT5 terminal integration.
"""

import logging
import os
import platform
import subprocess
from fastapi import APIRouter, HTTPException
from pathlib import Path
from typing import List, Dict, Any

from src.api.ide_models import MT5ScanRequest, MT5LaunchRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ide/mt5", tags=["mt5"])


def _platform_common_paths() -> List[str]:
    if platform.system() == "Windows":
        username = os.getenv("USERNAME", os.getenv("USER", "user"))
        return [
            r"C:\Program Files\MetaTrader 5\terminal64.exe",
            r"C:\Program Files (x86)\MetaTrader 5\terminal.exe",
            rf"C:\Users\{username}\AppData\Roaming\MetaQuotes\Terminal\terminal64.exe",
        ]
    if platform.system() == "Linux":
        return [
            os.path.expanduser("~/.wine/drive_c/Program Files/MetaTrader 5/terminal.exe"),
            "/opt/metatrader5/terminal.exe",
            "/usr/bin/metatrader5",
        ]
    if platform.system() == "Darwin":
        return [
            "/Applications/MetaTrader 5.app/Contents/MacOS/terminal",
            "/Applications/MetaTrader 5 Terminal.app/Contents/MacOS/terminal",
        ]
    return []


def _scan_installations(custom_paths: List[str] | None = None) -> List[Dict[str, Any]]:
    installations: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for raw_path in [*_platform_common_paths(), *(custom_paths or [])]:
        expanded = os.path.expanduser(raw_path)
        if expanded in seen:
            continue
        seen.add(expanded)

        path = Path(expanded)
        if path.is_file():
            stat = path.stat()
            installations.append(
                {
                    "path": str(path),
                    "exists": True,
                    "size_bytes": stat.st_size,
                    "modified": stat.st_mtime,
                }
            )
            continue

        if path.is_dir():
            for candidate_name in ("terminal64.exe", "terminal.exe", "terminal"):
                candidate = path / candidate_name
                if candidate.is_file():
                    stat = candidate.stat()
                    installations.append(
                        {
                            "path": str(candidate),
                            "exists": True,
                            "size_bytes": stat.st_size,
                            "modified": stat.st_mtime,
                        }
                    )
                    break

    return installations


def _mt5_runtime_status() -> Dict[str, Any]:
    host_platform = platform.system()
    try:
        import MetaTrader5 as mt5
    except ImportError:
        if host_platform == "Linux":
            error_message = (
                "MetaTrader5 package not installed on this Linux host; "
                "local MT5 runtime is unavailable here. Use the Windows VPS bridge for live MT5 access."
            )
        else:
            error_message = "MetaTrader5 package not installed"
        return {
            "connected": False,
            "terminal_path": None,
            "version": None,
            "accounts": [],
            "error": error_message,
        }

    initialized = False
    try:
        initialized = mt5.initialize()
        if not initialized:
            return {
                "connected": False,
                "terminal_path": None,
                "version": None,
                "accounts": [],
                "error": str(mt5.last_error()),
            }

        terminal_info = mt5.terminal_info()
        version = mt5.version()
        account_info = mt5.account_info()
        accounts: List[Dict[str, Any]] = []
        if account_info is not None and hasattr(account_info, "_asdict"):
            account_dict = account_info._asdict()
            accounts.append(
                {
                    "login": account_dict.get("login"),
                    "server": account_dict.get("server"),
                    "currency": account_dict.get("currency"),
                    "balance": account_dict.get("balance"),
                    "equity": account_dict.get("equity"),
                    "margin": account_dict.get("margin"),
                    "free_margin": account_dict.get("margin_free"),
                }
            )

        terminal_dict = terminal_info._asdict() if terminal_info and hasattr(terminal_info, "_asdict") else {}
        return {
            "connected": True,
            "terminal_path": terminal_dict.get("path") or terminal_dict.get("data_path"),
            "version": list(version) if version else None,
            "accounts": accounts,
            "error": None,
        }
    except Exception as e:
        logger.error(f"Error checking MT5 status: {e}")
        return {
            "connected": False,
            "terminal_path": None,
            "version": None,
            "accounts": [],
            "error": str(e),
        }
    finally:
        if initialized:
            try:
                mt5.shutdown()
            except Exception:
                pass


@router.get("/status")
async def get_mt5_status():
    """Get MT5 terminal status."""
    return _mt5_runtime_status()


@router.post("/scan")
async def scan_mt5_terminals(request: MT5ScanRequest):
    """Scan for MT5 terminals."""
    terminals = _scan_installations(request.custom_paths)
    return {
        "terminals": terminals,
        "count": len(terminals),
        "platform": platform.system(),
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
            raise HTTPException(
                400,
                "MT5 launch is only supported on Windows. "
                "On Linux, use the remote Windows VPS/MT5 bridge instead of a local terminal launch.",
            )

    except Exception as e:
        logger.error(f"Error launching MT5: {e}")
        raise HTTPException(500, str(e))
