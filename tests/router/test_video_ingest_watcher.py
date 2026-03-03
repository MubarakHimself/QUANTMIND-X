import pytest
from pathlib import Path
from src.router.video_ingest_watcher import VideoIngestWatcher, VideoIngestWatchEvent

@pytest.fixture
def temp_video_dir(tmp_path):
    video_dir = tmp_path / "video_in" / "downloads"
    video_dir.mkdir(parents=True)
    return video_dir

def test_video_ingest_watcher_init(temp_video_dir):
    watcher = VideoIngestWatcher(watch_dir=temp_video_dir)
    assert watcher.watch_dir == temp_video_dir
    assert not watcher.is_running

def test_detects_analysis_json(temp_video_dir):
    """Test that watcher detects new analysis.json"""
    watcher = VideoIngestWatcher(watch_dir=temp_video_dir)

    # Create mock analysis.json
    video_id = "test_video_123"
    video_dir = temp_video_dir / video_id
    video_dir.mkdir()

    analysis_file = video_dir / "analysis.json"
    analysis_file.write_text('{"strategy": "RSI scalper", "symbols": ["EURUSD"]}')

    # Trigger handler manually for testing
    events = []
    watcher.handler.callback = lambda e: events.append(e)
    watcher.handler.on_created(MockEvent(src_path=str(analysis_file)))

    assert len(events) == 1
    assert events[0].video_id == video_id

class MockEvent:
    def __init__(self, src_path):
        self.src_path = src_path
        self.is_directory = False
