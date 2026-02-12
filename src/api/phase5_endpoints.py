"""
Phase 5 API Endpoints

Task Group 8: API Layer Integration - Phase 5

Provides RESTful API endpoints for:
- Database export (CSV/JSON)
- MT5 scanner and launcher
- Broker connection with actual MT5 integration

Architecture:
- Uses Pydantic for request/response validation
- Integrates with existing database, MT5, and broker modules
- Supports cross-platform MT5 detection and launching
"""

import logging
import csv
import io
import os
import platform
import subprocess
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models for API Requests/Responses
# =============================================================================

class DatabaseExportRequest(BaseModel):
    """Request model for database export."""
    format: str = Field(default="csv", description="Export format: csv or json")
    limit: Optional[int] = Field(default=None, description="Maximum rows to export")


class DatabaseExportResponse(BaseModel):
    """Response model for database export."""
    success: bool
    table_name: str
    format: str
    row_count: int
    total_rows: int
    columns: List[str]
    content: str
    size_bytes: int
    error: Optional[str] = None


class MT5ScanRequest(BaseModel):
    """Request model for MT5 scan."""
    custom_paths: Optional[List[str]] = Field(default=None, description="Custom paths to scan")


class MT5ScanResponse(BaseModel):
    """Response model for MT5 scan."""
    success: bool
    platform: str
    found: bool
    installations: List[Dict[str, Any]]
    searched_paths: List[str]
    recommendations: Dict[str, str]
    web_terminal_available: bool
    web_terminal_url: str


class MT5LaunchRequest(BaseModel):
    """Request model for MT5 launch."""
    terminal_path: str = Field(..., description="Path to MT5 terminal executable")
    login: Optional[int] = Field(default=None, description="Account login number")
    password: Optional[str] = Field(default=None, description="Account password")
    server: Optional[str] = Field(default=None, description="Broker server name")


class MT5LaunchResponse(BaseModel):
    """Response model for MT5 launch."""
    success: bool
    terminal_path: str
    login: Optional[int]
    server: Optional[str]
    platform: str
    message: str
    timestamp: str
    error: Optional[str] = None


class BrokerConnectRequest(BaseModel):
    """Request model for broker connection."""
    broker_id: str = Field(..., description="Broker identifier")
    login: int = Field(..., description="Account login number")
    password: str = Field(..., description="Account password")
    server: str = Field(..., description="Broker server name")


class BrokerConnectResponse(BaseModel):
    """Response model for broker connection."""
    success: bool
    broker_id: str
    login: int
    server: str
    connected: bool
    account_info: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: str


# =============================================================================
# API Endpoint Handlers
# =============================================================================

class DatabaseExportAPIHandler:
    """
    Handles database export API endpoints.

    Integrates with:
    - src/database/duckdb_connection.py for database access
    - src/api/analytics_db.py for analytics data
    """

    def __init__(self):
        """Initialize database export handler."""
        self.data_dir = Path(os.getenv("QUANTMIND_DATA_DIR", "data"))

    def list_tables(self) -> Dict[str, Any]:
        """List all available database tables."""
        tables = {
            "strategies": str(self.data_dir / "strategies"),
            "assets": str(self.data_dir / "shared_assets"),
            "knowledge": str(self.data_dir / "knowledge"),
            "backtests": str(self.data_dir / "backtests"),
        }

        # Add DuckDB tables if available
        try:
            from src.database.duckdb_connection import DuckDBConnection
            db = DuckDBConnection()
            connection = db.get_connection()

            # Get list of tables
            result = connection.execute("SHOW TABLES").fetchall()
            duckdb_tables = [row[0] for row in result]

            tables["duckdb"] = {
                "analytics_db": str(self.data_dir / "analytics.duckdb"),
                "tables": duckdb_tables
            }

        except Exception as e:
            logger.debug(f"DuckDB not available: {e}")
            tables["duckdb"] = {"tables": []}

        return {
            "available_tables": tables,
            "total_count": len(tables)
        }

    def export_table(self, table_name: str, export_format: str = "csv",
                     limit: Optional[int] = None) -> DatabaseExportResponse:
        """
        Export database table to CSV or JSON.

        Args:
            table_name: Name of the table to export
            export_format: Format to export (csv or json)
            limit: Maximum number of rows to export

        Returns:
            DatabaseExportResponse with exported data
        """
        try:
            # Try to export from DuckDB first
            from src.database.duckdb_connection import DuckDBConnection

            db = DuckDBConnection()
            connection = db.get_connection()

            # Check if table exists
            try:
                count_result = connection.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
                total_rows = count_result[0] if count_result else 0

                if total_rows == 0:
                    return DatabaseExportResponse(
                        success=False,
                        table_name=table_name,
                        format=export_format,
                        row_count=0,
                        total_rows=0,
                        columns=[],
                        content="",
                        size_bytes=0,
                        error=f"Table {table_name} is empty"
                    )
            except Exception as e:
                return DatabaseExportResponse(
                    success=False,
                    table_name=table_name,
                    format=export_format,
                    row_count=0,
                    total_rows=0,
                    columns=[],
                    content="",
                    size_bytes=0,
                    error=f"Table {table_name} does not exist: {str(e)}"
                )

            # Build query with limit
            limit_clause = f"LIMIT {limit}" if limit else ""
            query = f"SELECT * FROM {table_name} {limit_clause}"

            # Fetch data
            result = connection.execute(query).fetchall()
            columns = [desc[0] for desc in connection.description]

            # Convert to list of dicts
            data = [dict(zip(columns, row)) for row in result]

            if export_format.lower() == "csv":
                # Generate CSV
                output = io.StringIO()
                writer = csv.DictWriter(output, fieldnames=columns)
                writer.writeheader()
                writer.writerows(data)
                content = output.getvalue()

                return DatabaseExportResponse(
                    success=True,
                    table_name=table_name,
                    format="csv",
                    row_count=len(data),
                    total_rows=total_rows,
                    columns=columns,
                    content=content,
                    size_bytes=len(content.encode('utf-8'))
                )

            else:  # JSON format
                import json

                content = json.dumps(data, indent=2, default=str)

                return DatabaseExportResponse(
                    success=True,
                    table_name=table_name,
                    format="json",
                    row_count=len(data),
                    total_rows=total_rows,
                    columns=columns,
                    content=content,
                    size_bytes=len(content.encode('utf-8'))
                )

        except ImportError:
            return DatabaseExportResponse(
                success=False,
                table_name=table_name,
                format=export_format,
                row_count=0,
                total_rows=0,
                columns=[],
                content="",
                size_bytes=0,
                error="DuckDB not available"
            )
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return DatabaseExportResponse(
                success=False,
                table_name=table_name,
                format=export_format,
                row_count=0,
                total_rows=0,
                columns=[],
                content="",
                size_bytes=0,
                error=str(e)
            )


class MT5ScannerAPIHandler:
    """
    Handles MT5 scanner and launcher API endpoints.

    Features:
    - Cross-platform MT5 detection (Windows, Linux/macOS)
    - Automated path scanning
    - MT5 desktop application launching
    - Platform-specific recommendations
    """

    def __init__(self):
        """Initialize MT5 scanner handler."""
        self.platform = platform.system()
        self.common_paths = self._get_common_paths()

    def _get_common_paths(self) -> List[str]:
        """Get common MT5 installation paths based on platform."""
        username = os.getenv('USER', os.getenv('USERNAME', 'user'))

        paths = {
            'Windows': [
                r'C:\Program Files\MetaTrader 5\terminal64.exe',
                r'C:\Program Files (x86)\MetaTrader 5\terminal.exe',
                rf'C:\Users\{username}\AppData\Roaming\MetaQuotes\Terminal\{username}\terminal64.exe',
            ],
            'Linux': [
                os.path.expanduser('~/.wine/drive_c/Program Files/MetaTrader 5/terminal.exe'),
                '/opt/metatrader5/terminal.exe',
                '/usr/bin/metatrader5',
            ],
            'Darwin': [
                '/Applications/MetaTrader 5.app/Contents/MacOS/terminal',
                '/Applications/MetaTrader 5 Terminal.app/Contents/MacOS/terminal',
            ]
        }

        return paths.get(self.platform, [])

    def scan_for_mt5(self, custom_paths: Optional[List[str]] = None) -> MT5ScanResponse:
        """
        Scan for MT5 installation on the system.

        Args:
            custom_paths: Optional list of custom paths to scan

        Returns:
            MT5ScanResponse with scan results
        """
        found_installations = []

        # Scan common paths
        paths_to_scan = self.common_paths.copy()
        if custom_paths:
            paths_to_scan.extend(custom_paths)

        for path in paths_to_scan:
            # Expand user path
            expanded_path = os.path.expanduser(path)

            if os.path.isfile(expanded_path):
                try:
                    # Get file stats
                    stat_info = os.stat(expanded_path)
                    found_installations.append({
                        "path": expanded_path,
                        "type": "file",
                        "size_bytes": stat_info.st_size,
                        "modified": datetime.fromtimestamp(stat_info.st_mtime).isoformat(),
                        "platform": self.platform
                    })
                except Exception as e:
                    logger.debug(f"Could not stat {path}: {e}")

            elif os.path.isdir(expanded_path):
                # Look for terminal.exe/terminal in directory
                terminal_exe = os.path.join(
                    expanded_path,
                    'terminal64.exe' if self.platform == 'Windows' else 'terminal'
                )
                if os.path.isfile(terminal_exe):
                    try:
                        stat_info = os.stat(terminal_exe)
                        found_installations.append({
                            "path": terminal_exe,
                            "type": "directory_installation",
                            "size_bytes": stat_info.st_size,
                            "modified": datetime.fromtimestamp(stat_info.st_mtime).isoformat(),
                            "platform": self.platform
                        })
                    except Exception as e:
                        logger.debug(f"Could not stat terminal: {e}")

        # Get platform recommendations
        recommendations = self._get_platform_recommendations()

        return MT5ScanResponse(
            success=True,
            platform=self.platform,
            found=len(found_installations) > 0,
            installations=found_installations,
            searched_paths=paths_to_scan,
            recommendations=recommendations,
            web_terminal_available=True,
            web_terminal_url="https://web.metaquotes.net/en/terminal"
        )

    def _get_platform_recommendations(self) -> Dict[str, str]:
        """Get platform-specific recommendations for MT5."""
        if self.platform == 'Windows':
            return {
                "status": "fully_supported",
                "message": "MT5 Desktop is fully supported on Windows. Direct MT5 integration available.",
                "actions": ["Scan for existing installations", "Download from metaquotes.net", "Use Web Terminal"]
            }
        elif self.platform == 'Linux':
            return {
                "status": "partial_support",
                "message": "MT5 can run on Linux via Wine. Web Terminal recommended for easier setup.",
                "actions": ["Install via Wine", "Use Web Terminal", "Consider Windows VM for full features"]
            }
        elif self.platform == 'Darwin':
            return {
                "status": "partial_support",
                "message": "MT5 Desktop is supported on macOS. Some features may be limited.",
                "actions": ["Scan for existing installations", "Download for macOS", "Use Web Terminal"]
            }
        else:
            return {
                "status": "unknown_platform",
                "message": f"Platform {self.platform} not recognized. Web Terminal recommended.",
                "actions": ["Use Web Terminal"]
            }

    def launch_mt5(self, terminal_path: str, login: Optional[int] = None,
                   password: Optional[str] = None, server: Optional[str] = None) -> MT5LaunchResponse:
        """
        Launch MT5 desktop application.

        Args:
            terminal_path: Path to MT5 terminal executable
            login: Optional account login number
            password: Optional account password
            server: Optional broker server name

        Returns:
            MT5LaunchResponse with launch result
        """
        try:
            # Verify path exists
            if not os.path.isfile(terminal_path):
                return MT5LaunchResponse(
                    success=False,
                    terminal_path=terminal_path,
                    login=login,
                    server=server,
                    platform=self.platform,
                    message=f"Terminal not found at: {terminal_path}",
                    timestamp=datetime.now().isoformat(),
                    error=f"Terminal not found at: {terminal_path}"
                )

            # Build command arguments
            args = [terminal_path]

            # Add login parameters if provided
            if login:
                args.append(f"/login:{login}")
                if password:
                    args.append(f"/password:{password}")
                if server:
                    args.append(f"/server:{server}")

            # Launch process
            if self.platform == 'Windows':
                # Use start command to detach from terminal
                subprocess.Popen(
                    ['start', '', terminal_path] + args[1:],
                    shell=True,
                    creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
                )
            else:
                # Unix-like systems
                subprocess.Popen(
                    args,
                    start_new_session=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )

            logger.info(f"Launched MT5: {terminal_path}")

            return MT5LaunchResponse(
                success=True,
                terminal_path=terminal_path,
                login=login,
                server=server,
                platform=self.platform,
                message="MT5 terminal launched successfully",
                timestamp=datetime.now().isoformat()
            )

        except Exception as e:
            logger.error(f"Failed to launch MT5: {e}")
            return MT5LaunchResponse(
                success=False,
                terminal_path=terminal_path,
                login=login,
                server=server,
                platform=self.platform,
                message="Failed to launch MT5 terminal",
                timestamp=datetime.now().isoformat(),
                error=str(e)
            )


class BrokerConnectionAPIHandler:
    """
    Handles broker connection API endpoints with actual MT5 integration.

    Integrates with:
    - MetaTrader5 Python package for terminal connection
    - src/router/broker_registry.py for broker management
    """

    def __init__(self):
        """Initialize broker connection handler."""
        self._connected = False
        self._mt5 = None

    def connect_broker(self, request: BrokerConnectRequest) -> BrokerConnectResponse:
        """
        Connect to MT5 broker with actual MT5 integration.

        Args:
            request: Broker connection request

        Returns:
            BrokerConnectResponse with connection result
        """
        try:
            # Try to import MetaTrader5 package
            try:
                import MetaTrader5 as mt5
                self._mt5 = mt5
            except ImportError:
                return BrokerConnectResponse(
                    success=False,
                    broker_id=request.broker_id,
                    login=request.login,
                    server=request.server,
                    connected=False,
                    account_info=None,
                    error="MetaTrader5 package not installed. Install with: pip install MetaTrader5",
                    timestamp=datetime.now().isoformat()
                )

            # Initialize MT5
            if not mt5.initialize():
                error_code = mt5.last_error()
                return BrokerConnectResponse(
                    success=False,
                    broker_id=request.broker_id,
                    login=request.login,
                    server=request.server,
                    connected=False,
                    account_info=None,
                    error=f"MT5 initialize failed: {error_code}",
                    timestamp=datetime.now().isoformat()
                )

            # Login to account
            if not mt5.login(request.login, request.password, request.server):
                error_code = mt5.last_error()
                mt5.shutdown()
                return BrokerConnectResponse(
                    success=False,
                    broker_id=request.broker_id,
                    login=request.login,
                    server=request.server,
                    connected=False,
                    account_info=None,
                    error=f"MT5 login failed: {error_code}",
                    timestamp=datetime.now().isoformat()
                )

            # Get account information
            account_info = mt5.account_info()._asdict()

            # Get terminal info
            terminal_info = mt5.terminal_info()._asdict() if mt5.terminal_info() else {}

            # Shutdown MT5
            mt5.shutdown()

            self._connected = True

            logger.info(f"Connected to broker {request.broker_id}: {request.login}@{request.server}")

            return BrokerConnectResponse(
                success=True,
                broker_id=request.broker_id,
                login=request.login,
                server=request.server,
                connected=True,
                account_info={
                    "login": account_info.get("login"),
                    "server": account_info.get("server"),
                    "balance": account_info.get("balance"),
                    "equity": account_info.get("equity"),
                    "margin": account_info.get("margin"),
                    "free_margin": account_info.get("margin_free"),
                    "margin_level": account_info.get("margin_level"),
                    "currency": account_info.get("currency"),
                    "company": account_info.get("company"),
                    "name": account_info.get("name"),
                },
                timestamp=datetime.now().isoformat()
            )

        except Exception as e:
            logger.error(f"Broker connection failed: {e}")
            return BrokerConnectResponse(
                success=False,
                broker_id=request.broker_id,
                login=request.login,
                server=request.server,
                connected=False,
                account_info=None,
                error=str(e),
                timestamp=datetime.now().isoformat()
            )

    def disconnect_broker(self) -> Dict[str, Any]:
        """Disconnect from MT5 broker."""
        try:
            if self._mt5 and self._connected:
                self._mt5.shutdown()
                self._connected = False

            return {
                "success": True,
                "message": "Disconnected from broker",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Disconnect failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


# =============================================================================
# FastAPI Application Factory
# =============================================================================

def create_phase5_api_app():
    """
    Create FastAPI application with Phase 5 endpoints.

    Example usage:
        from fastapi import FastAPI
        app = create_phase5_api_app()
    """
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import JSONResponse, StreamingResponse
        from fastapi.middleware.cors import CORSMiddleware
    except ImportError:
        logger.warning("FastAPI not available. Install with: pip install fastapi uvicorn")
        return None

    app = FastAPI(
        title="QuantMindX Phase 5 API",
        description="Database export, MT5 scanner, and broker connection endpoints",
        version="1.0.0"
    )

    # CORS for Tauri
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["tauri://localhost", "http://localhost:1420", "http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )

    # Initialize handlers
    db_export_handler = DatabaseExportAPIHandler()
    mt5_scanner_handler = MT5ScannerAPIHandler()
    broker_connection_handler = BrokerConnectionAPIHandler()

    # --------------------------------------------------------------------------
    # Database Export Endpoints
    # --------------------------------------------------------------------------

    @app.get("/api/database/tables")
    async def list_database_tables():
        """List all available database tables."""
        return db_export_handler.list_tables()

    @app.get("/api/database/export/{table_name}")
    async def export_database_table(
        table_name: str,
        format: str = "csv",
        limit: Optional[int] = None
    ):
        """
        Export database table to CSV or JSON.

        Args:
            table_name: Name of the table to export
            format: Export format (csv or json)
            limit: Maximum number of rows to export
        """
        result = db_export_handler.export_table(table_name, format, limit)

        if not result.success:
            raise HTTPException(400, result.error)

        # Return content directly for download
        media_type = "text/csv" if result.format == "csv" else "application/json"
        filename = f"{table_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{result.format}"

        return StreamingResponse(
            io.BytesIO(result.content.encode('utf-8')),
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )

    # --------------------------------------------------------------------------
    # MT5 Scanner Endpoints
    # --------------------------------------------------------------------------

    @app.post("/api/mt5/scan", response_model=MT5ScanResponse)
    async def scan_mt5_installation(request: MT5ScanRequest):
        """Scan for MT5 installation on the system."""
        return mt5_scanner_handler.scan_for_mt5(request.custom_paths)

    @app.get("/api/mt5/scan", response_model=MT5ScanResponse)
    async def scan_mt5_installation_get():
        """Scan for MT5 installation (GET method)."""
        return mt5_scanner_handler.scan_for_mt5()

    @app.post("/api/mt5/launch", response_model=MT5LaunchResponse)
    async def launch_mt5_terminal(request: MT5LaunchRequest):
        """Launch MT5 desktop application."""
        return mt5_scanner_handler.launch_mt5(
            terminal_path=request.terminal_path,
            login=request.login,
            password=request.password,
            server=request.server
        )

    # --------------------------------------------------------------------------
    # Broker Connection Endpoints
    # --------------------------------------------------------------------------

    @app.post("/api/trading/broker/connect", response_model=BrokerConnectResponse)
    async def connect_to_broker(request: BrokerConnectRequest):
        """Connect to MT5 broker with actual MT5 integration."""
        return broker_connection_handler.connect_broker(request)

    @app.post("/api/trading/broker/disconnect")
    async def disconnect_from_broker():
        """Disconnect from MT5 broker."""
        return broker_connection_handler.disconnect_broker()

    # --------------------------------------------------------------------------
    # Health Check
    # --------------------------------------------------------------------------

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service": "QuantMindX Phase 5 API",
            "features": {
                "database_export": True,
                "mt5_scanner": True,
                "mt5_launcher": True,
                "broker_connection": True
            }
        }

    return app


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Request/Response Models
    'DatabaseExportRequest',
    'DatabaseExportResponse',
    'MT5ScanRequest',
    'MT5ScanResponse',
    'MT5LaunchRequest',
    'MT5LaunchResponse',
    'BrokerConnectRequest',
    'BrokerConnectResponse',
    # API Handlers
    'DatabaseExportAPIHandler',
    'MT5ScannerAPIHandler',
    'BrokerConnectionAPIHandler',
    # Application Factory
    'create_phase5_api_app',
]
