"""
AuthBar Component - Authentication Status Bar

Displays authentication status for AI providers (Gemini, Qwen, etc.)
used in the YouTube-EA pipeline.
"""

from typing import Optional
from dataclasses import dataclass

try:
    from textual.widget import Widget
    from textual.reactive import reactive
    from textual import log
    TEXTUAL_AVAILABLE = True
except ImportError:
    TEXTUAL_AVAILABLE = False
    Widget = object
    def reactive(*args, **kwargs):
        """Dummy reactive decorator when textual not available."""
        def decorator(func):
            return func
        return decorator
    def log(*args, **kwargs):
        pass

from rich.text import Text
from rich.panel import Panel


@dataclass
class AuthStatus:
    """Authentication status for a provider."""
    provider: str
    has_key: bool = False
    key_valid: bool = False
    connected: bool = False
    user: Optional[str] = None

    @property
    def status_text(self) -> str:
        """Get status text for display."""
        if not self.has_key:
            return f"{self.provider}: No key"
        if not self.key_valid:
            return f"{self.provider}: Invalid key"
        if self.connected:
            return f"{self.provider}: Connected"
        return f"{self.provider}: Testing..."

    @property
    def is_valid(self) -> bool:
        """Check if auth is valid and connected."""
        return self.has_key and self.key_valid and self.connected


if TEXTUAL_AVAILABLE:
    class AuthBar(Widget):
        """Authentication status bar widget.

        Displays connection status for AI provider API keys used in
        the YouTube-EA pipeline (Gemini, Qwen, etc.).
        """

        DEFAULT_CSS = """
        AuthBar {
            height: 3;
            dock: top;
            padding: 0 1;
        }

        AuthBar .auth-container {
            height: 100%;
            display: flex;
            flex-direction: row;
            align-items: center;
            gap: 2;
        }

        AuthBar .auth-item {
            padding: 0 1;
        }

        AuthBar .auth-item.connected {
            text-style: bold;
            background: $success;
        }

        AuthBar .auth-item.invalid {
            text-style: bold;
            background: $error;
        }

        AuthBar .auth-item.testing {
            text-style: dim;
        }

        AuthBar .status-indicator {
            text-style: bold;
        }
        """

        # Reactive status for providers
        gemini_status: reactive[AuthStatus] = reactive(AuthStatus("Gemini"))
        qwen_status: reactive[AuthStatus] = reactive(AuthStatus("Qwen"))
        checking_connection: reactive[bool] = reactive(False)

        def __init__(self, id: str | None = None):
            """Initialize the AuthBar widget."""
            super().__init__(id=id)
            self._api_base_url = "http://localhost:8000"

        def on_mount(self) -> None:
            """Initialize on mount."""
            self.update_auth_status()
            # Refresh auth status every 30 seconds
            self.set_interval(30.0, self.update_auth_status)

        def update_auth_status(self) -> None:
            """Update authentication status from environment/API."""
            import os

            # Check Gemini (Google AI) status
            gemini_key = os.getenv("GOOGLE_API_KEY", "")
            self.gemini_status = AuthStatus(
                provider="Gemini",
                has_key=bool(gemini_key),
                key_valid=self._validate_google_key(gemini_key),
                connected=False  # Will be updated by test_connection
            )

            # Check Qwen status
            qwen_key = os.getenv("QWEN_API_KEY", "")
            self.qwen_status = AuthStatus(
                provider="Qwen",
                has_key=bool(qwen_key),
                key_valid=self._validate_qwen_key(qwen_key),
                connected=False
            )

            # Test connections
            self.test_connections()

        def _validate_google_key(self, key: str) -> bool:
            """Validate Google API key format."""
            return key.startswith("AIza") and len(key) > 30

        def _validate_qwen_key(self, key: str) -> bool:
            """Validate Qwen API key format."""
            return len(key) > 10

        async def test_connections(self) -> None:
            """Test API connections."""
            if self.checking_connection:
                return

            self.checking_connection = True

            try:
                import httpx

                async with httpx.AsyncClient(timeout=5.0) as client:
                    # Test Gemini
                    if self.gemini_status.has_key and self.gemini_status.key_valid:
                        try:
                            response = await client.post(
                                f"{self._api_base_url}/api/ai/test-connection",
                                json={"provider": "google"}
                            )
                            self.gemini_status = AuthStatus(
                                provider="Gemini",
                                has_key=True,
                                key_valid=True,
                                connected=response.status_code == 200
                            )
                        except Exception as e:
                            log(f"Gemini connection test failed: {e}")
                            self.gemini_status = AuthStatus(
                                provider="Gemini",
                                has_key=True,
                                key_valid=True,
                                connected=False
                            )

                    # Test Qwen
                    if self.qwen_status.has_key and self.qwen_status.key_valid:
                        try:
                            response = await client.post(
                                f"{self._api_base_url}/api/ai/test-connection",
                                json={"provider": "qwen"}
                            )
                            self.qwen_status = AuthStatus(
                                provider="Qwen",
                                has_key=True,
                                key_valid=True,
                                connected=response.status_code == 200
                            )
                        except Exception as e:
                            log(f"Qwen connection test failed: {e}")
                            self.qwen_status = AuthStatus(
                                provider="Qwen",
                                has_key=True,
                                key_valid=True,
                                connected=False
                            )
            except Exception as e:
                log(f"Connection test error: {e}")
            finally:
                self.checking_connection = False

        def _get_status_icon(self, status: AuthStatus) -> str:
            """Get status icon for display."""
            if not status.has_key:
                return "✗"
            if not status.key_valid:
                return "⚠"
            if status.connected:
                return "✓"
            return "⟳"

        def _get_status_class(self, status: AuthStatus) -> str:
            """Get CSS class for status."""
            if not status.has_key:
                return "invalid"
            if not status.key_valid:
                return "invalid"
            if status.connected:
                return "connected"
            return "testing"

        def render(self) -> Panel:
            """Render the auth bar."""
            from rich.align import Align

            # Build status items
            items = []
            for status in [self.gemini_status, self.qwen_status]:
                icon = self._get_status_icon(status)
                text_class = self._get_status_class(status)

                text = Text()
                text.append(icon, style="bold")
                text.append(" ")
                text.append(status.status_text, style=text_class)

                items.append(Text.assemble(text))

            # Join with separator
            separator = Text(" │ ", style="dim")
            content = Text()
            for i, item in enumerate(items):
                if i > 0:
                    content.append(separator)
                content.append(item)

            return Panel(
                Align.center(content),
                title="🔑 Authentication",
                border_style="dim",
                padding=(0, 1)
            )
else:
    # Stub class when textual not available
    class AuthBar:
        """Stub AuthBar when textual not available."""
        def __init__(self, *args, **kwargs):
            pass
