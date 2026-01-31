# Docker Tests for QuantMindX Strategy Agent

This directory contains comprehensive tests for the Docker container setup of the QuantMindX Strategy Agent.

## Test Coverage

### 1. Build Tests (Test 01-02)
- `test_01_container_builds_successfully`: Verifies Docker image builds without errors
- `test_02_base_image_is_correct`: Validates base image is python:3.10-slim

### 2. Security Tests (Test 03-06)
- `test_03_container_runs_as_non_root_user`: Ensures container runs as UID 1001 (not root)
- `test_04_container_has_read_only_root_filesystem`: Verifies root filesystem is read-only
- `test_05_container_has_restricted_capabilities`: Checks ALL capabilities are dropped
- `test_06_container_has_no_new_privileges`: Validates no-new-privileges security option

### 3. Resource Limit Tests (Test 07-09)
- `test_07_container_has_memory_limit`: Verifies 512MB memory limit
- `test_08_container_has_cpu_limit`: Validates 1.0 CPU limit
- `test_09_container_has_pids_limit`: Checks 100 PIDs limit (fork bomb protection)

### 4. Dependency Tests (Test 10-12)
- `test_10_python_packages_installed`: Verifies pydantic, redis, fastmcp, structlog
- `test_11_metatrader5_module_available`: Checks MetaTrader5 module is importable
- `test_12_entrypoint_script_exists_and_executable`: Validates entrypoint.sh

### 5. Health Check Tests (Test 13)
- `test_13_healthcheck_configured`: Verifies healthcheck is configured

### 6. Entrypoint Tests (Test 14-15)
- `test_14_entrypoint_validates_non_root_user`: Checks user validation
- `test_15_entrypoint_starts_background_services`: Verifies heartbeat and trade event publishers start

### 7. Docker Compose Tests (Test 16-18)
- `test_16_docker_compose_file_exists`: Checks docker-compose.yml exists
- `test_17_docker_compose_has_security_options`: Validates security options in compose
- `test_18_docker_compose_has_resource_limits`: Validates resource limits in compose

## Running Tests

### Run all tests:
```bash
cd /home/mubarkahimself/Desktop/QUANTMINDX/docker/strategy-agent
pytest tests/ -v
```

### Run specific test category:
```bash
# Security tests only
pytest tests/test_docker_security.py::TestDockerSecurity -v

# Build tests only
pytest tests/test_docker_security.py::TestDockerBuild -v

# Resource limit tests only
pytest tests/test_docker_security.py::TestDockerResources -v
```

### Run with detailed output:
```bash
pytest tests/ -v -s
```

### Run specific test:
```bash
pytest tests/test_docker_security.py::TestDockerSecurity::test_03_container_runs_as_non_root_user -v
```

## Requirements

- Docker daemon running
- pytest installed: `pip install pytest`
- Test container cleanup happens automatically

## Security Features Tested

1. **Non-root user**: Container runs as UID 1001 (trader user)
2. **Read-only filesystem**: Root filesystem is mounted read-only
3. **Capability dropping**: ALL capabilities dropped, only NET_BIND_SERVICE added
4. **No new privileges**: no-new-privileges security option enabled
5. **Resource limits**: Memory (512MB), CPU (1.0), PIDs (100)
6. **Fork bomb protection**: PIDs limit prevents process proliferation

## Expected Test Results

All 18 tests should pass:
- 2 build tests
- 4 security tests
- 3 resource limit tests
- 3 dependency tests
- 1 health check test
- 2 entrypoint tests
- 3 docker-compose tests

## Troubleshooting

### Docker not available:
```bash
# Start Docker daemon
sudo systemctl start docker
sudo systemctl status docker
```

### Permission denied:
```bash
# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker
```

### Build fails:
```bash
# Check Dockerfile syntax
docker build -t test --no-cache .
```

### Container won't start:
```bash
# Check container logs
docker logs test-strategy-agent
```
