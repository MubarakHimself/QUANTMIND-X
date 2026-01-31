#!/usr/bin/env python3
"""
Configuration validation script for Docker setup
Validates Dockerfile, entrypoint.sh, requirements.txt, and docker-compose.yml
"""

import os
import re
import sys
from typing import List, Tuple


def validate_dockerfile() -> Tuple[bool, List[str]]:
    """Validate Dockerfile configuration"""
    errors = []

    dockerfile_path = "/home/mubarkahimself/Desktop/QUANTMINDX/docker/strategy-agent/Dockerfile"

    if not os.path.exists(dockerfile_path):
        errors.append("Dockerfile does not exist")
        return False, errors

    with open(dockerfile_path, 'r') as f:
        content = f.read()

    # Check base image
    if "FROM python:3.10-slim" not in content:
        errors.append("Dockerfile does not use python:3.10-slim base image")

    # Check for non-root user creation
    if "useradd -r -g trader -u 1001" not in content:
        errors.append("Dockerfile does not create non-root user 'trader' with UID 1001")

    # Check for USER directive
    if "USER trader" not in content:
        errors.append("Dockerfile does not switch to non-root user")

    # Check for read-only filesystem comment
    if "read-only" not in content.lower():
        errors.append("Dockerfile does not mention read-only filesystem")

    # Check for healthcheck
    if "HEALTHCHECK" not in content:
        errors.append("Dockerfile does not have HEALTHCHECK directive")

    # Check for security hardening
    security_features = [
        "no-new-privileges",
        "cap_drop",
        "cap_add"
    ]
    for feature in security_features:
        if feature not in content.lower():
            errors.append(f"Dockerfile missing security feature: {feature}")

    return len(errors) == 0, errors


def validate_entrypoint() -> Tuple[bool, List[str]]:
    """Validate entrypoint.sh configuration"""
    errors = []

    entrypoint_path = "/home/mubarkahimself/Desktop/QUANTMINDX/docker/strategy-agent/entrypoint.sh"

    if not os.path.exists(entrypoint_path):
        errors.append("entrypoint.sh does not exist")
        return False, errors

    # Check if executable
    if not os.access(entrypoint_path, os.X_OK):
        errors.append("entrypoint.sh is not executable")

    with open(entrypoint_path, 'r') as f:
        content = f.read()

    # Check for shebang
    if not content.startswith("#!/bin/bash"):
        errors.append("entrypoint.sh does not have #!/bin/bash shebang")

    # Check for security functions
    required_functions = [
        "check_security",
        "init_mt5",
        "start_heartbeat_publisher",
        "start_trade_event_publisher",
        "start_agent",
        "cleanup"
    ]

    for func in required_functions:
        if f"{func}()" not in content:
            errors.append(f"entrypoint.sh missing function: {func}")

    # Check for non-root check
    if "id -u" not in content:
        errors.append("entrypoint.sh does not check for root user")

    # Check for MT5 initialization
    if "MetaTrader5" not in content:
        errors.append("entrypoint.sh does not initialize MT5")

    # Check for heartbeat interval (60s)
    if "HEARTBEAT_INTERVAL=60" not in content:
        errors.append("entrypoint.sh does not set heartbeat interval to 60s")

    # Check for signal trapping (graceful shutdown)
    if "trap" not in content or "SIGTERM" not in content:
        errors.append("entrypoint.sh does not trap signals for graceful shutdown")

    return len(errors) == 0, errors


def validate_requirements() -> Tuple[bool, List[str]]:
    """Validate requirements.txt"""
    errors = []

    requirements_path = "/home/mubarkahimself/Desktop/QUANTMINDX/docker/strategy-agent/requirements.txt"

    if not os.path.exists(requirements_path):
        errors.append("requirements.txt does not exist")
        return False, errors

    with open(requirements_path, 'r') as f:
        content = f.read()

    # Check for required packages
    required_packages = [
        "pydantic",
        "redis",
        "fastmcp",
        "MetaTrader5",
        "pytest",
        "bandit"
    ]

    for package in required_packages:
        if package.lower() not in content.lower():
            errors.append(f"requirements.txt missing package: {package}")

    return len(errors) == 0, errors


def validate_docker_compose() -> Tuple[bool, List[str]]:
    """Validate docker-compose.yml"""
    errors = []

    compose_path = "/home/mubarkahimself/Desktop/QUANTMINDX/docker/strategy-agent/docker-compose.yml"

    if not os.path.exists(compose_path):
        errors.append("docker-compose.yml does not exist")
        return False, errors

    with open(compose_path, 'r') as f:
        content = f.read()

    # Check for version
    if "version:" not in content:
        errors.append("docker-compose.yml missing version")

    # Check for services
    if "services:" not in content:
        errors.append("docker-compose.yml missing services section")

    # Check for redis service
    if "redis:" not in content:
        errors.append("docker-compose.yml missing redis service")

    # Check for strategy-agent service
    if "strategy-agent:" not in content:
        errors.append("docker-compose.yml missing strategy-agent service")

    # Check for security options
    if "no-new-privileges:true" not in content:
        errors.append("docker-compose.yml missing no-new-privileges security option")

    # Check for read-only filesystem
    if "read_only: true" not in content:
        errors.append("docker-compose.yml missing read-only filesystem")

    # Check for resource limits
    if "memory:" not in content:
        errors.append("docker-compose.yml missing memory limit")

    if "cpus:" not in content:
        errors.append("docker-compose.yml missing CPU limit")

    if "pids_limit:" not in content:
        errors.append("docker-compose.yml missing PIDs limit")

    # Check for healthcheck
    if "healthcheck:" not in content:
        errors.append("docker-compose.yml missing healthcheck")

    # Check for networks
    if "networks:" not in content:
        errors.append("docker-compose.yml missing networks section")

    return len(errors) == 0, errors


def validate_tests() -> Tuple[bool, List[str]]:
    """Validate test files"""
    errors = []

    test_dir = "/home/mubarkahimself/Desktop/QUANTMINDX/docker/strategy-agent/tests"

    if not os.path.exists(test_dir):
        errors.append("tests directory does not exist")
        return False, errors

    # Check for test file
    test_file = os.path.join(test_dir, "test_docker_security.py")
    if not os.path.exists(test_file):
        errors.append("test_docker_security.py does not exist")
    else:
        with open(test_file, 'r') as f:
            test_content = f.read()

        # Count number of tests (should be 2-8+)
        test_count = len(re.findall(r'def test_\d+_', test_content))
        if test_count < 2:
            errors.append(f"Not enough tests (found {test_count}, need at least 2)")

    # Check for conftest.py
    conftest_file = os.path.join(test_dir, "conftest.py")
    if not os.path.exists(conftest_file):
        errors.append("conftest.py does not exist")

    # Check for README
    readme_file = os.path.join(test_dir, "README.md")
    if not os.path.exists(readme_file):
        errors.append("tests/README.md does not exist")

    return len(errors) == 0, errors


def main():
    """Run all validations"""
    print("=" * 60)
    print("QuantMindX Docker Configuration Validation")
    print("=" * 60)
    print()

    all_valid = True
    validations = [
        ("Dockerfile", validate_dockerfile),
        ("entrypoint.sh", validate_entrypoint),
        ("requirements.txt", validate_requirements),
        ("docker-compose.yml", validate_docker_compose),
        ("Tests", validate_tests),
    ]

    for name, validate_func in validations:
        print(f"Validating {name}...")
        valid, errors = validate_func()

        if valid:
            print(f"  ✓ {name} is valid")
        else:
            print(f"  ✗ {name} has errors:")
            for error in errors:
                print(f"    - {error}")
            all_valid = False

        print()

    print("=" * 60)
    if all_valid:
        print("✓ All validations passed!")
        print("=" * 60)
        return 0
    else:
        print("✗ Some validations failed")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
