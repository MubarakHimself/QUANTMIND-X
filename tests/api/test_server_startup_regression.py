from pathlib import Path
import re


def test_startup_event_does_not_shadow_os_module():
    """Regression: startup_event must not re-import os as a local symbol.

    A local `import os` inside startup_event causes UnboundLocalError when
    earlier lines read `os.environ` for graph memory migration config.
    """
    server_path = Path("src/api/server.py")
    source = server_path.read_text(encoding="utf-8")

    match = re.search(
        r"async def startup_event\(\):(?P<body>[\s\S]*?)\n@app\.",
        source,
        re.MULTILINE,
    )
    assert match, "startup_event function block not found"

    body = match.group("body")
    assert "graph_db = os.environ.get(" in body
    assert "import os" not in body

