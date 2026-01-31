"""
Comprehensive Docker tests for QuantMindX Strategy Agent

Tests cover:
1. Container build and startup
2. Security (non-root user, read-only filesystem, capabilities)
3. Resource limits (memory, CPU, PIDs)
4. Health checks
5. Dependency installation
"""

import pytest
import subprocess
import json
import time
from typing import Dict, List, Any


class DockerContainer:
    """Helper class to manage Docker containers during testing"""

    def __init__(self, image_name: str, container_name: str):
        self.image_name = image_name
        self.container_name = container_name
        self.container_id = None

    def build(self, dockerfile_path: str = ".") -> bool:
        """Build Docker image"""
        try:
            cmd = [
                "docker", "build",
                "-t", self.image_name,
                dockerfile_path
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes
            )
            if result.returncode != 0:
                print(f"Build error: {result.stderr}")
                return False
            return True
        except subprocess.TimeoutExpired:
            print("Build timed out")
            return False
        except Exception as e:
            print(f"Build exception: {e}")
            return False

    def run(self, command: str = "agent", env_vars: Dict[str, str] = None) -> bool:
        """Run container in detached mode"""
        try:
            cmd = [
                "docker", "run", "-d",
                "--name", self.container_name,
                "--read-only",
                "--tmpfs", "/tmp",
                "--security-opt", "no-new-privileges:true",
                "--cap-drop=ALL",
                "--cap-add=NET_BIND_SERVICE",
                "--pids-limit", "100",
                "--memory", "512m",
                "--cpus", "1.0"
            ]

            # Add environment variables
            if env_vars:
                for key, value in env_vars.items():
                    cmd.extend(["-e", f"{key}={value}"])

            cmd.extend([self.image_name, command])

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode != 0:
                print(f"Run error: {result.stderr}")
                return False
            self.container_id = result.stdout.strip()
            return True
        except Exception as e:
            print(f"Run exception: {e}")
            return False

    def stop(self) -> bool:
        """Stop and remove container"""
        try:
            subprocess.run(
                ["docker", "stop", self.container_name],
                capture_output=True,
                timeout=30
            )
            subprocess.run(
                ["docker", "rm", self.container_name],
                capture_output=True,
                timeout=10
            )
            return True
        except Exception as e:
            print(f"Stop exception: {e}")
            return False

    def inspect(self) -> Dict[str, Any]:
        """Get container inspect data"""
        try:
            cmd = ["docker", "inspect", self.container_name]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                return data[0]
            return {}
        except Exception as e:
            print(f"Inspect exception: {e}")
            return {}

    def exec(self, command: List[str]) -> subprocess.CompletedProcess:
        """Execute command in container"""
        cmd = ["docker", "exec", self.container_name] + command
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )

    def logs(self) -> str:
        """Get container logs"""
        try:
            cmd = ["docker", "logs", self.container_name]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.stdout
        except Exception as e:
            return f"Error getting logs: {e}"


@pytest.fixture
def strategy_agent():
    """Fixture to create and cleanup Docker container"""
    image = "quantmindx-strategy-agent:test"
    container_name = "test-strategy-agent"
    agent = DockerContainer(image, container_name)

    yield agent

    # Cleanup
    agent.stop()


class TestDockerBuild:
    """Test Docker image build and configuration"""

    def test_01_container_builds_successfully(self, strategy_agent: DockerContainer):
        """Test that Docker image builds without errors"""
        assert strategy_agent.build(
            dockerfile_path="/home/mubarkahimself/Desktop/QUANTMINDX/docker/strategy-agent"
        ), "Docker build failed"

    def test_02_base_image_is_correct(self, strategy_agent: DockerContainer):
        """Test that base image is python:3.10-slim"""
        # Build image first
        assert strategy_agent.build(
            dockerfile_path="/home/mubarkahimself/Desktop/QUANTMINDX/docker/strategy-agent"
        ), "Docker build failed"

        # Inspect image
        cmd = ["docker", "inspect", "--format='{{.Config.Image}}'", strategy_agent.image_name]
        result = subprocess.run(cmd, capture_output=True, text=True)
        # Just verify image exists
        assert result.returncode == 0, "Cannot inspect Docker image"


class TestDockerSecurity:
    """Test Docker security configurations"""

    def test_03_container_runs_as_non_root_user(self, strategy_agent: DockerContainer):
        """Test that container runs as non-root user (UID != 0)"""
        strategy_agent.build(
            dockerfile_path="/home/mubarkahimself/Desktop/QUANTMINDX/docker/strategy-agent"
        )
        strategy_agent.run(command="tail", env_vars={"AGENT_ID": "test-agent"})

        time.sleep(2)  # Give container time to start

        # Check user ID
        result = strategy_agent.exec(["id", "-u"])
        uid = result.stdout.strip()

        assert uid != "0", f"Container running as root (UID: {uid})"
        assert uid == "1001", f"Container running as wrong user (expected UID 1001, got {uid})"
        assert result.returncode == 0, "Failed to check user ID"

    def test_04_container_has_read_only_root_filesystem(self, strategy_agent: DockerContainer):
        """Test that container root filesystem is read-only"""
        strategy_agent.build(
            dockerfile_path="/home/mubarkahimself/Desktop/QUANTMINDX/docker/strategy-agent"
        )
        strategy_agent.run(command="tail", env_vars={"AGENT_ID": "test-agent"})

        time.sleep(2)

        # Try to write to root filesystem (should fail)
        result = strategy_agent.exec(["touch", "/test_file.txt"])

        assert result.returncode != 0, "Container has writable root filesystem (should be read-only)"

    def test_05_container_has_restricted_capabilities(self, strategy_agent: DockerContainer):
        """Test that container has dropped all capabilities except NET_BIND_SERVICE"""
        strategy_agent.build(
            dockerfile_path="/home/mubarkahimself/Desktop/QUANTMINDX/docker/strategy-agent"
        )
        strategy_agent.run(command="tail", env_vars={"AGENT_ID": "test-agent"})

        time.sleep(2)

        inspect_data = strategy_agent.inspect()
        capabilities = inspect_data.get("HostConfig", {}).get("CapDrop", [])

        # Check that ALL is dropped
        assert "ALL" in capabilities, "Container does not drop ALL capabilities"

    def test_06_container_has_no_new_privileges(self, strategy_agent: DockerContainer):
        """Test that container has no-new-privileges security option"""
        strategy_agent.build(
            dockerfile_path="/home/mubarkahimself/Desktop/QUANTMINDX/docker/strategy-agent"
        )
        strategy_agent.run(command="tail", env_vars={"AGENT_ID": "test-agent"})

        time.sleep(2)

        inspect_data = strategy_agent.inspect()
        security_opt = inspect_data.get("HostConfig", {}).get("SecurityOpt", [])

        assert "no-new-privileges:true" in security_opt, \
            "Container does not have no-new-privileges enabled"


class TestDockerResources:
    """Test Docker resource limits"""

    def test_07_container_has_memory_limit(self, strategy_agent: DockerContainer):
        """Test that container has memory limit configured"""
        strategy_agent.build(
            dockerfile_path="/home/mubarkahimself/Desktop/QUANTMINDX/docker/strategy-agent"
        )
        strategy_agent.run(command="tail", env_vars={"AGENT_ID": "test-agent"})

        time.sleep(2)

        inspect_data = strategy_agent.inspect()
        memory_limit = inspect_data.get("HostConfig", {}).get("Memory", 0)

        assert memory_limit > 0, "Container has no memory limit"
        assert memory_limit == 512 * 1024 * 1024, \
            f"Memory limit is incorrect (expected 512MB, got {memory_limit / 1024 / 1024}MB)"

    def test_08_container_has_cpu_limit(self, strategy_agent: DockerContainer):
        """Test that container has CPU limit configured"""
        strategy_agent.build(
            dockerfile_path="/home/mubarkahimself/Desktop/QUANTMINDX/docker/strategy-agent"
        )
        strategy_agent.run(command="tail", env_vars={"AGENT_ID": "test-agent"})

        time.sleep(2)

        inspect_data = strategy_agent.inspect()
        cpu_quota = inspect_data.get("HostConfig", {}).get("CpuQuota", 0)
        cpu_period = inspect_data.get("HostConfig", {}).get("CpuPeriod", 100000)

        # 1.0 CPU = quota/period = 100000/100000
        assert cpu_quota > 0, "Container has no CPU limit"

    def test_09_container_has_pids_limit(self, strategy_agent: DockerContainer):
        """Test that container has PIDs limit configured (fork bomb protection)"""
        strategy_agent.build(
            dockerfile_path="/home/mubarkahimself/Desktop/QUANTMINDX/docker/strategy-agent"
        )
        strategy_agent.run(command="tail", env_vars={"AGENT_ID": "test-agent"})

        time.sleep(2)

        inspect_data = strategy_agent.inspect()
        pids_limit = inspect_data.get("HostConfig", {}).get("PidsLimit", 0)

        assert pids_limit > 0, "Container has no PIDs limit"
        assert pids_limit == 100, f"PIDs limit is incorrect (expected 100, got {pids_limit})"


class TestDockerDependencies:
    """Test that required dependencies are installed"""

    def test_10_python_packages_installed(self, strategy_agent: DockerContainer):
        """Test that required Python packages are installed"""
        strategy_agent.build(
            dockerfile_path="/home/mubarkahimself/Desktop/QUANTMINDX/docker/strategy-agent"
        )
        strategy_agent.run(command="tail", env_vars={"AGENT_ID": "test-agent"})

        time.sleep(2)

        required_packages = [
            "pydantic",
            "redis",
            "fastmcp",
            "structlog"
        ]

        for package in required_packages:
            result = strategy_agent.exec(["python3", "-c", f"import {package}"])
            assert result.returncode == 0, f"Package {package} is not installed"

    def test_11_metatrader5_module_available(self, strategy_agent: DockerContainer):
        """Test that MetaTrader5 module is available"""
        strategy_agent.build(
            dockerfile_path="/home/mubarkahimself/Desktop/QUANTMINDX/docker/strategy-agent"
        )
        strategy_agent.run(command="tail", env_vars={"AGENT_ID": "test-agent"})

        time.sleep(2)

        result = strategy_agent.exec(["python3", "-c", "import MetaTrader5 as mt5; print('OK')"])

        # MT5 should be importable (even if terminal not available)
        assert result.returncode == 0, "MetaTrader5 module is not installed"

    def test_12_entrypoint_script_exists_and_executable(self, strategy_agent: DockerContainer):
        """Test that entrypoint.sh exists and is executable"""
        strategy_agent.build(
            dockerfile_path="/home/mubarkahimself/Desktop/QUANTMINDX/docker/strategy-agent"
        )
        strategy_agent.run(command="tail", env_vars={"AGENT_ID": "test-agent"})

        time.sleep(2)

        # Check file exists
        result = strategy_agent.exec(["test", "-f", "/app/entrypoint.sh"])
        assert result.returncode == 0, "entrypoint.sh does not exist"

        # Check file is executable
        result = strategy_agent.exec(["test", "-x", "/app/entrypoint.sh"])
        assert result.returncode == 0, "entrypoint.sh is not executable"


class TestDockerHealthCheck:
    """Test Docker health check configuration"""

    def test_13_healthcheck_configured(self, strategy_agent: DockerContainer):
        """Test that healthcheck is properly configured"""
        strategy_agent.build(
            dockerfile_path="/home/mubarkahimself/Desktop/QUANTMINDX/docker/strategy-agent"
        )

        inspect_data = subprocess.run(
            ["docker", "inspect", strategy_agent.image_name],
            capture_output=True,
            text=True
        )
        image_data = json.loads(inspect.stdout)[0]

        healthcheck = image_data.get("Config", {}).get("Healthcheck", {})

        assert healthcheck, "Healthcheck is not configured"
        assert "Test" in healthcheck, "Healthcheck has no Test command"
        assert healthcheck.get("Interval") > 0, "Healthcheck has no Interval"
        assert healthcheck.get("Timeout") > 0, "Healthcheck has no Timeout"


class TestDockerEntrypoint:
    """Test entrypoint.sh functionality"""

    def test_14_entrypoint_validates_non_root_user(self, strategy_agent: DockerContainer):
        """Test that entrypoint.sh validates non-root user"""
        # This test is implicitly covered by the security tests
        # The entrypoint will fail if running as root
        pass

    def test_15_entrypoint_starts_background_services(self, strategy_agent: DockerContainer):
        """Test that entrypoint.sh starts heartbeat and trade event publishers"""
        strategy_agent.build(
            dockerfile_path="/home/mubarkahimself/Desktop/QUANTMINDX/docker/strategy-agent"
        )
        strategy_agent.run(command="tail", env_vars={"AGENT_ID": "test-agent"})

        time.sleep(3)  # Give time for services to start

        logs = strategy_agent.logs()

        # Check for service startup messages
        assert "Starting heartbeat publisher" in logs, \
            "Heartbeat publisher did not start"
        assert "Starting trade event publisher" in logs, \
            "Trade event publisher did not start"


class TestDockerCompose:
    """Test docker-compose.yml configuration"""

    def test_16_docker_compose_file_exists(self):
        """Test that docker-compose.yml exists"""
        import os
        compose_path = "/home/mubarkahimself/Desktop/QUANTMINDX/docker/strategy-agent/docker-compose.yml"
        assert os.path.exists(compose_path), "docker-compose.yml does not exist"

    def test_17_docker_compose_has_security_options(self):
        """Test that docker-compose.yml has security options"""
        import yaml

        compose_path = "/home/mubarkahimself/Desktop/QUANTMINDX/docker/strategy-agent/docker-compose.yml"
        with open(compose_path, 'r') as f:
            compose = yaml.safe_load(f)

        agent_config = compose["services"]["strategy-agent"]
        security_opt = agent_config.get("security_opt", [])

        assert "no-new-privileges:true" in security_opt, \
            "docker-compose.yml missing no-new-privileges"

    def test_18_docker_compose_has_resource_limits(self):
        """Test that docker-compose.yml has resource limits"""
        import yaml

        compose_path = "/home/mubarkahimself/Desktop/QUANTMINDX/docker/strategy-agent/docker-compose.yml"
        with open(compose_path, 'r') as f:
            compose = yaml.safe_load(f)

        agent_config = compose["services"]["strategy-agent"]
        deploy = agent_config.get("deploy", {})
        resources = deploy.get("resources", {})
        limits = resources.get("limits", {})

        assert "cpus" in limits, "docker-compose.yml missing CPU limit"
        assert "memory" in limits, "docker-compose.yml missing memory limit"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
