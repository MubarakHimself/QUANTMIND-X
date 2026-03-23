"""
Zero-Auth Configuration Endpoints

Configures Qwen CLI (OAuth / API key) and Gemini (ADC) without requiring
manual API key entry. Claude/Anthropic uses the normal ANTHROPIC_API_KEY env
var and does not need special zero-auth handling here.
"""
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
import os
import subprocess
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/zero-auth", tags=["zero-auth"])


class ZeroAuthStatus(BaseModel):
    qwen_configured: bool
    qwen_method: str  # "oauth" | "api_key" | "none"
    gemini_configured: bool
    gemini_method: str  # "adc" | "api_key" | "none"
    gemini_project: Optional[str] = None


class QwenCLIConfig(BaseModel):
    api_key: Optional[str] = None  # set QWEN_API_KEY directly if provided


class GeminiADCConfig(BaseModel):
    project_id: str
    credentials_path: Optional[str] = None  # path to service account JSON


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

@router.get("/status", response_model=ZeroAuthStatus)
def get_zero_auth_status():
    """Return auth configuration status for Qwen CLI and Gemini ADC."""
    qwen_api_key = os.getenv("QWEN_API_KEY")
    qwen_auth_file = Path.home() / ".qwen" / "auth.json"
    qwen_oauth = qwen_auth_file.exists()

    qwen_ok = bool(qwen_api_key or qwen_oauth)
    qwen_method = (
        "api_key" if qwen_api_key
        else ("oauth" if qwen_oauth else "none")
    )

    gemini_ok = bool(
        os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_CLOUD_PROJECT")
    )
    gemini_method = (
        "adc" if (os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or os.getenv("GOOGLE_CLOUD_PROJECT"))
        else ("api_key" if os.getenv("GEMINI_API_KEY") else "none")
    )

    return ZeroAuthStatus(
        qwen_configured=qwen_ok,
        qwen_method=qwen_method,
        gemini_configured=gemini_ok,
        gemini_method=gemini_method,
        gemini_project=os.getenv("GOOGLE_CLOUD_PROJECT"),
    )


# ---------------------------------------------------------------------------
# Qwen CLI endpoints
# ---------------------------------------------------------------------------

@router.post("/qwen/authenticate")
def qwen_authenticate(config: QwenCLIConfig):
    """
    Trigger Qwen CLI browser-based OAuth flow or store an API key.

    If config.api_key is provided, it is stored via the /qwen/apikey logic.
    Otherwise, the Qwen CLI OAuth browser flow is launched via subprocess.
    """
    if config.api_key:
        return _store_qwen_api_key(config.api_key)

    try:
        result = subprocess.run(
            ["qwen", "authenticate"],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode == 0:
            return {"status": "authenticated", "method": "oauth", "output": result.stdout.strip()}
        return {"status": "error", "error": result.stderr.strip() or "qwen authenticate exited non-zero"}
    except FileNotFoundError:
        return {"status": "error", "error": "Qwen CLI not installed — run: pip install qwen-cli"}
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "message": "Browser OAuth flow is still running in background"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@router.post("/qwen/apikey")
def qwen_set_api_key(config: QwenCLIConfig):
    """Store QWEN_API_KEY in .env and os.environ."""
    if not config.api_key:
        return {"status": "error", "error": "api_key is required"}
    return _store_qwen_api_key(config.api_key)


@router.get("/qwen/status")
def qwen_status():
    """Return Qwen CLI configuration status."""
    qwen_api_key = os.getenv("QWEN_API_KEY")
    qwen_auth_file = Path.home() / ".qwen" / "auth.json"
    qwen_oauth = qwen_auth_file.exists()

    if qwen_api_key:
        method = "api_key"
        configured = True
    elif qwen_oauth:
        method = "oauth"
        configured = True
    else:
        method = "none"
        configured = False

    return {"configured": configured, "method": method}


@router.get("/qwen/test")
def qwen_test():
    """Check that the Qwen CLI is installed and responsive."""
    try:
        result = subprocess.run(
            ["qwen", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return {"status": "ok", "version": result.stdout.strip()}
        return {"status": "error", "error": result.stderr.strip()}
    except FileNotFoundError:
        return {"status": "not_installed", "error": "Qwen CLI not found — run: pip install qwen-cli"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ---------------------------------------------------------------------------
# Gemini ADC endpoints (unchanged)
# ---------------------------------------------------------------------------

@router.post("/gemini/adc")
def configure_gemini_adc(config: GeminiADCConfig):
    """Store Gemini ADC config in .env"""
    env_path = Path(".env")
    _update_env_var(env_path, "GOOGLE_CLOUD_PROJECT", config.project_id)
    if config.credentials_path:
        _update_env_var(env_path, "GOOGLE_APPLICATION_CREDENTIALS", config.credentials_path)
    os.environ["GOOGLE_CLOUD_PROJECT"] = config.project_id
    return {"status": "configured", "method": "adc", "project": config.project_id}


@router.get("/gemini/test")
def test_gemini_connection():
    """Test if Gemini ADC is configured."""
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not project:
        return {"status": "not_configured"}
    return {"status": "configured", "project": project, "credentials": creds or "default ADC"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _store_qwen_api_key(api_key: str) -> dict:
    """Persist QWEN_API_KEY to .env and current process environment."""
    env_path = Path(".env")
    _update_env_var(env_path, "QWEN_API_KEY", api_key)
    os.environ["QWEN_API_KEY"] = api_key
    return {"status": "configured", "method": "api_key"}


def _update_env_var(env_path: Path, key: str, value: str):
    """Update or add a variable in .env file."""
    lines = []
    found = False
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith(f"{key}="):
                lines.append(f"{key}={value}")
                found = True
            else:
                lines.append(line)
    if not found:
        lines.append(f"{key}={value}")
    env_path.write_text("\n".join(lines) + "\n")
