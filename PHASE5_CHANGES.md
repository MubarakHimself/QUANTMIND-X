# Phase 5 - Backend API Endpoints Implementation Summary

## Overview

This document summarizes the changes made to implement Phase 5 - Backend API Endpoints for the QuantMindX trading system.

## Files Modified

### 1. `/home/mubarkahimself/Desktop/QUANTMINDX/src/api/ide_endpoints.py`
**Status**: Modified imports only

**Changes**:
- Added imports for `csv`, `io`, `platform`, and `subprocess` modules
- Added imports for `sys` module
- Prepared file for additional handler classes (DatabaseAPIHandler, MT5APIHandler)

### 2. `/home/mubarkahimself/Desktop/QUANTMINDX/src/api/trading_endpoints.py`
**Status**: Modified

**Changes**:
- Added new Pydantic models for broker connection:
  - `BrokerConnectRequest` - Request model for broker connection
  - `BrokerConnectResponse` - Response model with account info
- Added new `BrokerConnectionHandler` class:
  - `connect_broker()` - Connects to MT5 using MetaTrader5 package
  - `disconnect_broker()` - Disconnects from MT5
  - Returns account information including balance, equity, margin, etc.
- Added new API endpoints:
  - `POST /api/v1/trading/broker/connect` - Connect to MT5 broker
  - `POST /api/v1/trading/broker/disconnect` - Disconnect from broker
- Updated module exports to include new models and handlers

### 3. `/home/mubarkahimself/Desktop/QUANTMINDX/quantmind-ide/src/lib/services/mt5Scanner.ts`
**Status**: Modified

**Changes**:
- Updated `scanForMT5()` method:
  - Now calls backend API endpoint `POST http://localhost:8000/api/mt5/scan`
  - Returns actual scan results from backend including installation path
  - Handles errors gracefully with fallback
- Updated `launchMT5Desktop()` method:
  - Now calls backend API endpoint `POST http://localhost:8000/api/mt5/launch`
  - Passes login, password, and server configuration
  - Returns actual launch status from backend

### 4. `/home/mubarkahimself/Desktop/QUANTMINDX/src/api/phase5_endpoints.py` (NEW FILE)
**Status**: Created

**Purpose**: Standalone Phase 5 API endpoints file containing:

**Classes**:
1. `DatabaseExportAPIHandler`
   - `list_tables()` - Lists all available database tables
   - `export_table()` - Exports table to CSV or JSON format

2. `MT5ScannerAPIHandler`
   - `scan_for_mt5()` - Scans for MT5 installation on Windows/Linux/macOS
   - `launch_mt5()` - Launches MT5 desktop application
   - `_get_common_paths()` - Platform-specific MT5 paths
   - `_get_platform_recommendations()` - Platform-specific recommendations

3. `BrokerConnectionAPIHandler`
   - `connect_broker()` - Connects to MT5 with actual integration
   - `disconnect_broker()` - Disconnects from MT5

**API Endpoints**:
- `GET /api/database/tables` - List available database tables
- `GET /api/database/export/{table_name}` - Export table to CSV/JSON
- `POST /api/mt5/scan` - Scan for MT5 installation
- `GET /api/mt5/scan` - Scan for MT5 installation (GET method)
- `POST /api/mt5/launch` - Launch MT5 desktop application
- `POST /api/trading/broker/connect` - Connect to MT5 broker
- `POST /api/trading/broker/disconnect` - Disconnect from broker
- `GET /health` - Health check endpoint

## API Endpoint Details

### Database Export Endpoint

**Endpoint**: `GET /api/database/export/{table_name}`

**Query Parameters**:
- `format` - Export format (csv or json, default: csv)
- `limit` - Maximum rows to export (optional)

**Response**: Streaming file download with appropriate headers

**Example**:
```bash
curl -X GET "http://localhost:8000/api/database/export/backtests?format=csv&limit=1000" \
  -o backtests_export.csv
```

### MT5 Scanner Endpoint

**Endpoint**: `POST /api/mt5/scan`

**Request Body**:
```json
{
  "custom_paths": ["/custom/path/to/mt5"]
}
```

**Response**:
```json
{
  "success": true,
  "platform": "Linux",
  "found": true,
  "installations": [
    {
      "path": "/path/to/terminal64.exe",
      "type": "file",
      "size_bytes": 12345678,
      "modified": "2026-02-10T12:00:00",
      "platform": "Linux"
    }
  ],
  "searched_paths": [...],
  "recommendations": {...},
  "web_terminal_available": true,
  "web_terminal_url": "https://web.metaquotes.net/en/terminal"
}
```

### MT5 Launch Endpoint

**Endpoint**: `POST /api/mt5/launch`

**Request Body**:
```json
{
  "terminal_path": "/path/to/terminal64.exe",
  "login": 12345678,
  "password": "your_password",
  "server": "BrokerName-Server"
}
```

**Response**:
```json
{
  "success": true,
  "terminal_path": "/path/to/terminal64.exe",
  "login": 12345678,
  "server": "BrokerName-Server",
  "platform": "Linux",
  "message": "MT5 terminal launched successfully",
  "timestamp": "2026-02-10T12:00:00"
}
```

### Broker Connection Endpoint

**Endpoint**: `POST /api/v1/trading/broker/connect`

**Request Body**:
```json
{
  "broker_id": "my_broker",
  "login": 12345678,
  "password": "your_password",
  "server": "BrokerName-Server"
}
```

**Response**:
```json
{
  "success": true,
  "broker_id": "my_broker",
  "login": 12345678,
  "server": "BrokerName-Server",
  "connected": true,
  "account_info": {
    "login": 12345678,
    "server": "BrokerName-Server",
    "balance": 10000.0,
    "equity": 10245.80,
    "margin": 150.25,
    "free_margin": 10095.55,
    "margin_level": 6813.33,
    "currency": "USD",
    "company": "Broker Name",
    "name": "Account Name"
  },
  "timestamp": "2026-02-10T12:00:00"
}
```

## Testing the Endpoints

### Test Database Export
```bash
# List available tables
curl http://localhost:8000/api/database/tables

# Export table to CSV
curl "http://localhost:8000/api/database/export/backtests?format=csv" -o backtests.csv

# Export table to JSON
curl "http://localhost:8000/api/database/export/backtests?format=json" -o backtests.json

# Export with limit
curl "http://localhost:8000/api/database/export/backtests?format=csv&limit=100" -o backtests.csv
```

### Test MT5 Scanner
```bash
# Scan for MT5 installation
curl -X POST http://localhost:8000/api/mt5/scan \
  -H "Content-Type: application/json" \
  -d '{"custom_paths": ["/custom/path"]}'

# Or use GET method
curl http://localhost:8000/api/mt5/scan
```

### Test MT5 Launcher
```bash
# Launch MT5 with credentials
curl -X POST http://localhost:8000/api/mt5/launch \
  -H "Content-Type: application/json" \
  -d '{
    "terminal_path": "C:\\Program Files\\MetaTrader 5\\terminal64.exe",
    "login": 12345678,
    "password": "your_password",
    "server": "BrokerName-Server"
  }'
```

### Test Broker Connection
```bash
# Connect to broker
curl -X POST http://localhost:8000/api/v1/trading/broker/connect \
  -H "Content-Type: application/json" \
  -d '{
    "broker_id": "my_broker",
    "login": 12345678,
    "password": "your_password",
    "server": "BrokerName-Server"
  }'

# Disconnect from broker
curl -X POST http://localhost:8000/api/v1/trading/broker/disconnect
```

## Integration with Frontend

The frontend `mt5Scanner.ts` service now calls the backend APIs:

1. **Scanning**: Calls `/api/mt5/scan` to detect MT5 installation
2. **Launching**: Calls `/api/mt5/launch` to start MT5 with credentials
3. **Connecting**: Calls `/api/v1/trading/broker/connect` for actual MT5 connection

## Dependencies

Required Python packages:
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `pydantic` - Data validation
- `MetaTrader5` - MT5 Python integration (for broker connection)

Install with:
```bash
pip install fastapi uvicorn pydantic MetaTrader5
```

## Running the API Server

### Option 1: Standalone Phase 5 API
```bash
cd /home/mubarkahimself/Desktop/QUANTMINDX
python -m uvicorn src.api.phase5_endpoints:create_phase5_api_app --reload --host 0.0.0.0 --port 8000
```

### Option 2: Integrated with Main API
The endpoints are already integrated into the main trading API at `/api/v1/trading/broker/connect`.

## Security Notes

1. **Passwords**: Never log passwords in production
2. **MT5 Credentials**: Store securely using environment variables or secure vault
3. **CORS**: CORS is configured for Tauri (localhost:1420) and development (localhost:5173)
4. **Error Messages**: Generic error messages to avoid information leakage

## Platform Support

### Windows
- Full MT5 Desktop support
- Direct executable launching
- Full broker connection support

### Linux
- MT5 via Wine
- Web Terminal recommended
- Limited MT5 package support

### macOS
- MT5 Desktop supported
- Some features may be limited
- Web Terminal available

## Future Enhancements

1. Add WebSocket support for real-time MT5 price data
2. Add MT5 order placement endpoints
3. Add position management endpoints
4. Add historical data download from MT5
5. Add symbol info retrieval from MT5

## Summary

Phase 5 implementation successfully adds:
- Database export functionality (CSV/JSON)
- Cross-platform MT5 scanner
- MT5 desktop launcher
- Actual MT5 broker connection with account info retrieval

All endpoints are RESTful, use Pydantic for validation, and integrate with existing backend infrastructure.
