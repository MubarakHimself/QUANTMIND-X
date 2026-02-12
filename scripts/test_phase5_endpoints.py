#!/usr/bin/env python3
"""
Test script for Phase 5 API Endpoints

Tests:
1. Database export endpoint
2. MT5 scanner endpoint
3. MT5 launcher endpoint
4. Broker connection endpoint

Usage:
    python scripts/test_phase5_endpoints.py
"""

import sys
import requests
import json
from typing import Dict, Any

# API base URL
BASE_URL = "http://localhost:8000"


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_result(name: str, result: Dict[str, Any]):
    """Print test result."""
    status = "✓ PASS" if result.get("success") else "✗ FAIL"
    print(f"\n{status} - {name}")
    if "error" in result and not result.get("success"):
        print(f"  Error: {result['error']}")
    else:
        print(f"  Details: {json.dumps(result, indent=2, default=str)[:200]}...")


def test_database_list_tables():
    """Test listing database tables."""
    try:
        response = requests.get(f"{BASE_URL}/api/database/tables", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return {"success": True, "tables": data.get("available_tables", {})}
        return {"success": False, "error": f"HTTP {response.status_code}"}
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": str(e)}


def test_database_export():
    """Test exporting a database table."""
    try:
        # Try to export a table (will fail if table doesn't exist, but endpoint should work)
        response = requests.get(
            f"{BASE_URL}/api/database/export/test_table?format=json&limit=10",
            timeout=5
        )
        # Even if table doesn't exist, we should get a proper error response
        return {"success": response.status_code in [200, 400], "status": response.status_code}
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": str(e)}


def test_mt5_scan():
    """Test MT5 scanner endpoint."""
    try:
        response = requests.get(f"{BASE_URL}/api/mt5/scan", timeout=10)
        if response.status_code == 200:
            data = response.json()
            return {
                "success": True,
                "platform": data.get("platform"),
                "found": data.get("found"),
                "installations": len(data.get("installations", []))
            }
        return {"success": False, "error": f"HTTP {response.status_code}"}
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": str(e)}


def test_mt5_scan_post():
    """Test MT5 scanner with POST method and custom paths."""
    try:
        response = requests.post(
            f"{BASE_URL}/api/mt5/scan",
            json={"custom_paths": []},
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            return {
                "success": True,
                "platform": data.get("platform"),
                "web_terminal": data.get("web_terminal_available")
            }
        return {"success": False, "error": f"HTTP {response.status_code}"}
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": str(e)}


def test_mt5_launch():
    """Test MT5 launcher endpoint (without actually launching)."""
    try:
        # Test with invalid path to avoid actually launching
        response = requests.post(
            f"{BASE_URL}/api/mt5/launch",
            json={
                "terminal_path": "/nonexistent/path/to/terminal.exe"
            },
            timeout=5
        )
        # Should fail gracefully with proper error message
        data = response.json()
        if response.status_code == 200:
            return {
                "success": "success" in data and not data["success"],
                "error_handled": data.get("error") is not None
            }
        return {"success": False, "error": f"HTTP {response.status_code}"}
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": str(e)}


def test_broker_connect():
    """Test broker connection endpoint (with dummy credentials)."""
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/trading/broker/connect",
            json={
                "broker_id": "test_broker",
                "login": 123456,
                "password": "test_password",
                "server": "Test-Server"
            },
            timeout=10
        )
        # Should fail with proper error (no MT5 installed or invalid credentials)
        data = response.json()
        if response.status_code == 200:
            return {
                "success": True,
                "connected": data.get("connected", False),
                "has_response": "success" in data
            }
        return {"success": False, "error": f"HTTP {response.status_code}"}
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": str(e)}


def test_health_check():
    """Test health check endpoint."""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return {
                "success": data.get("status") == "healthy",
                "service": data.get("service")
            }
        return {"success": False, "error": f"HTTP {response.status_code}"}
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": str(e)}


def main():
    """Run all tests."""
    print_section("Phase 5 API Endpoint Tests")
    print(f"\nTesting API at: {BASE_URL}")
    print("\nNote: Some tests are expected to return errors (e.g., invalid credentials)")
    print("      We're testing that endpoints handle errors gracefully.")

    results = []

    # Health check
    print_section("Health Check")
    result = test_health_check()
    results.append(("Health Check", result))
    print_result("Health Check", result)

    # Database tests
    print_section("Database Export Endpoints")
    result = test_database_list_tables()
    results.append(("Database - List Tables", result))
    print_result("List Database Tables", result)

    result = test_database_export()
    results.append(("Database - Export", result))
    print_result("Export Database Table", result)

    # MT5 Scanner tests
    print_section("MT5 Scanner Endpoints")
    result = test_mt5_scan()
    results.append(("MT5 - Scan (GET)", result))
    print_result("MT5 Scan (GET)", result)

    result = test_mt5_scan_post()
    results.append(("MT5 - Scan (POST)", result))
    print_result("MT5 Scan (POST)", result)

    # MT5 Launcher tests
    print_section("MT5 Launcher Endpoints")
    result = test_mt5_launch()
    results.append(("MT5 - Launch", result))
    print_result("MT5 Launch (with invalid path)", result)

    # Broker Connection tests
    print_section("Broker Connection Endpoints")
    result = test_broker_connect()
    results.append(("Broker - Connect", result))
    print_result("Broker Connect (with dummy credentials)", result)

    # Summary
    print_section("Test Summary")
    passed = sum(1 for _, r in results if r.get("success"))
    total = len(results)

    print(f"\nResults: {passed}/{total} tests passed")
    print("\nDetailed Results:")
    for name, result in results:
        status = "✓" if result.get("success") else "✗"
        print(f"  {status} {name}")

    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
