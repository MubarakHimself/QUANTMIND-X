"""
Session Endpoints for QuantMind IDE.

Handles session-related operations:
- Strategy management (create, list, get)
- Broker accounts
- Live trading control
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from .models import (
    StrategyStatus,
    StrategyFolder,
    StrategyDetail,
    BotControl,
    MT5ScanRequest,
    MT5LaunchRequest,
)

logger = logging.getLogger(__name__)


# Configuration
DATA_DIR = Path(os.getenv("QUANTMIND_DATA_DIR", "data"))
STRATEGIES_DIR = DATA_DIR / "strategies"


class SessionEndpoint:
    """Session endpoint handler for QuantMind IDE.

    Manages session-related operations including strategy folders,
    broker accounts, and live trading control.
    """

    def __init__(self):
        """Initialize session endpoint."""
        STRATEGIES_DIR.mkdir(parents=True, exist_ok=True)
        self._strategy_handler = None
        self._broker_handler = None
        self._trading_handler = None

    def _get_strategy_handler(self):
        """Lazy load strategy handler."""
        if self._strategy_handler is None:
            self._strategy_handler = StrategyAPIHandler()
        return self._strategy_handler

    def _get_broker_handler(self):
        """Lazy load broker handler."""
        if self._broker_handler is None:
            self._broker_handler = BrokerAccountsAPIHandler()
        return self._broker_handler

    def _get_trading_handler(self):
        """Lazy load trading handler."""
        if self._trading_handler is None:
            self._trading_handler = LiveTradingAPIHandler()
        return self._trading_handler

    def list_strategies(self) -> List[StrategyFolder]:
        """List all strategy folders."""
        return self._get_strategy_handler().list_strategies()

    def get_strategy(self, strategy_id: str) -> Optional[StrategyDetail]:
        """Get detailed strategy folder contents."""
        return self._get_strategy_handler().get_strategy(strategy_id)

    def create_strategy_folder(self, name: str) -> str:
        """Create a new strategy folder."""
        return self._get_strategy_handler().create_strategy_folder(name)

    def scan_mt5_accounts(self, request: MT5ScanRequest) -> Dict[str, Any]:
        """Scan for MT5 accounts."""
        return self._get_broker_handler().scan_mt5_accounts(request)

    def launch_mt5_terminal(self, request: MT5LaunchRequest) -> Dict[str, Any]:
        """Launch MT5 terminal."""
        return self._get_broker_handler().launch_mt5_terminal(request)

    def control_bot(self, bot_id: str, action: str) -> Dict[str, Any]:
        """Control bot (start/stop)."""
        return self._get_trading_handler().control_bot(bot_id, action)

    def clone_bot(self, bot_id: str, new_name: str) -> Dict[str, Any]:
        """Clone a trading bot."""
        return self._get_trading_handler().clone_bot(bot_id, new_name)


class StrategyAPIHandler:
    """Handler for strategy folder operations."""

    def __init__(self):
        STRATEGIES_DIR.mkdir(parents=True, exist_ok=True)

    def list_strategies(self) -> List[StrategyFolder]:
        """List all strategy folders."""
        strategies = []

        if not STRATEGIES_DIR.exists():
            return strategies

        for folder in STRATEGIES_DIR.iterdir():
            if folder.is_dir() and not folder.name.startswith('.'):
                status_file = folder / "status.json"
                status = StrategyStatus.PENDING
                created_at = datetime.fromtimestamp(folder.stat().st_ctime).isoformat()

                if status_file.exists():
                    try:
                        with open(status_file) as f:
                            data = json.load(f)
                            status = StrategyStatus(data.get("status", "pending"))
                    except:
                        pass

                strategies.append(StrategyFolder(
                    id=folder.name,
                    name=folder.name.replace("_", " "),
                    status=status,
                    created_at=created_at,
                    has_video_ingest=(folder / "video_ingest").exists(),
                    has_trd=(folder / "trd").exists(),
                    has_ea=(folder / "ea").exists(),
                    has_backtest=(folder / "backtest").exists()
                ))

        return sorted(strategies, key=lambda x: x.created_at, reverse=True)

    def get_strategy(self, strategy_id: str) -> Optional[StrategyDetail]:
        """Get detailed strategy folder contents."""
        folder = STRATEGIES_DIR / strategy_id

        if not folder.exists():
            return None

        status_file = folder / "status.json"
        status = StrategyStatus.PENDING

        if status_file.exists():
            try:
                with open(status_file) as f:
                    data = json.load(f)
                    status = StrategyStatus(data.get("status", "pending"))
            except:
                pass

        # Gather folder contents
        video_ingest_data = None
        if (folder / "video_ingest").exists():
            video_ingest_data = {
                "files": [f.name for f in (folder / "video_ingest").iterdir() if f.is_file()]
            }
            metadata_file = folder / "video_ingest" / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file) as f:
                    video_ingest_data["metadata"] = json.load(f)

        trd_data = None
        if (folder / "trd").exists():
            trd_data = {
                "files": [f.name for f in (folder / "trd").iterdir() if f.is_file()]
            }

        ea_data = None
        if (folder / "ea").exists():
            ea_data = {
                "files": [f.name for f in (folder / "ea").iterdir() if f.is_file()]
            }

        backtests = []
        if (folder / "backtest").exists():
            for bt_file in (folder / "backtest").iterdir():
                if bt_file.suffix in ['.html', '.json']:
                    backtests.append({
                        "name": bt_file.name,
                        "path": str(bt_file),
                        "mode": bt_file.stem.split('_')[0] if '_' in bt_file.stem else "mode_a"
                    })

        return StrategyDetail(
            id=strategy_id,
            name=strategy_id.replace("_", " "),
            status=status,
            created_at=datetime.fromtimestamp(folder.stat().st_ctime).isoformat(),
            video_ingest=video_ingest_data,
            trd=trd_data,
            ea=ea_data,
            backtests=backtests
        )

    def create_strategy_folder(self, name: str) -> str:
        """Create a new strategy folder."""
        folder_name = name.replace(" ", "_")
        folder = STRATEGIES_DIR / folder_name

        # Create folder structure
        (folder / "video_ingest").mkdir(parents=True, exist_ok=True)
        (folder / "trd").mkdir(exist_ok=True)
        (folder / "ea").mkdir(exist_ok=True)
        (folder / "backtest").mkdir(exist_ok=True)

        # Create status file
        with open(folder / "status.json", "w") as f:
            json.dump({
                "status": "pending",
                "created_at": datetime.now().isoformat(),
                "tags": []
            }, f)

        return folder_name


class BrokerAccountsAPIHandler:
    """Handler for MT5 broker accounts."""

    def __init__(self):
        self.mt5_detected = False
        try:
            import MetaTrader5 as mt5
            if mt5.initialize():
                self.mt5_detected = True
                mt5.shutdown()
        except ImportError:
            pass

    def scan_mt5_accounts(self, request: MT5ScanRequest) -> Dict[str, Any]:
        """Scan for MT5 accounts."""
        if not self.mt5_detected:
            return {
                "success": False,
                "error": "MetaTrader5 not installed",
                "accounts": []
            }

        try:
            import MetaTrader5 as mt5
            if not mt5.initialize():
                return {
                    "success": False,
                    "error": "Failed to initialize MT5",
                    "accounts": []
                }

            accounts = mt5.accounts_get()
            account_list = []

            for acc in accounts:
                account_list.append({
                    "login": acc.login,
                    "server": acc.server,
                    "currency": acc.currency,
                    "balance": acc.balance,
                    "equity": acc.equity,
                    "margin": acc.margin,
                    "free_margin": acc.margin_free,
                })

            mt5.shutdown()
            return {
                "success": True,
                "accounts": account_list
            }
        except Exception as e:
            logger.error(f"MT5 scan error: {e}")
            return {
                "success": False,
                "error": str(e),
                "accounts": []
            }

    def launch_mt5_terminal(self, request: MT5LaunchRequest) -> Dict[str, Any]:
        """Launch MT5 terminal."""
        import subprocess
        import platform

        terminal_path = request.terminal_path or "C:\\Program Files\\MetaTrader 5\\terminal64.exe"
        login = request.login or ""
        password = request.password or ""
        server = request.server or ""

        try:
            if platform.system() == "Windows":
                cmd = [terminal_path]
                if login:
                    cmd.extend([f"/login:{login}", f"/password:{password}", f"/server:{server}"])
                subprocess.Popen(cmd, detached=True)
                return {"success": True, "message": "MT5 launched"}
            else:
                return {"success": False, "error": "MT5 only supported on Windows"}
        except Exception as e:
            return {"success": False, "error": str(e)}


class LiveTradingAPIHandler:
    """Handler for live trading operations."""

    def __init__(self):
        self.bots_dir = DATA_DIR / "live_bots"
        self.bots_dir.mkdir(parents=True, exist_ok=True)

    def control_bot(self, bot_id: str, action: str) -> Dict[str, Any]:
        """Control bot (start/stop)."""
        bot_file = self.bots_dir / f"{bot_id}.json"

        if not bot_file.exists():
            return {"success": False, "error": "Bot not found"}

        try:
            with open(bot_file) as f:
                bot_data = json.load(f)

            if action == "start":
                bot_data["status"] = "running"
                bot_data["started_at"] = datetime.now().isoformat()
            elif action == "stop":
                bot_data["status"] = "stopped"
                bot_data["stopped_at"] = datetime.now().isoformat()
            else:
                return {"success": False, "error": f"Unknown action: {action}"}

            with open(bot_file, "w") as f:
                json.dump(bot_data, f, indent=2)

            return {"success": True, "bot": bot_data}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def clone_bot(self, bot_id: str, new_name: str) -> Dict[str, Any]:
        """Clone a trading bot."""
        bot_file = self.bots_dir / f"{bot_id}.json"
        new_bot_file = self.bots_dir / f"{new_name}.json"

        if not bot_file.exists():
            return {"success": False, "error": "Bot not found"}

        if new_bot_file.exists():
            return {"success": False, "error": "Bot name already exists"}

        try:
            with open(bot_file) as f:
                bot_data = json.load(f)

            bot_data["id"] = new_name
            bot_data["name"] = new_name
            bot_data["status"] = "pending"
            bot_data["created_at"] = datetime.now().isoformat()

            with open(new_bot_file, "w") as f:
                json.dump(bot_data, f, indent=2)

            return {"success": True, "bot": bot_data}
        except Exception as e:
            return {"success": False, "error": str(e)}
