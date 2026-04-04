from unittest.mock import Mock
from src.api.ide_handlers_video_ingest import VideoIngestAPIHandler
from src.video_ingest.processor import VideoIngestProcessor
from src.video_ingest.providers import GeminiCLIProvider, OpenRouterProvider


def test_process_video_submits_real_queue_job(monkeypatch):
    handler = VideoIngestAPIHandler()
    mock_queue = Mock()
    mock_queue.submit_job.return_value = "job_123"

    monkeypatch.setattr(
        "src.api.ide_handlers_video_ingest._get_runtime",
        lambda: (Mock(), Mock(), mock_queue),
    )

    result = handler.process_video(
        url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        strategy_name="My Strategy: VWAP",
        is_playlist=False,
    )

    assert result["job_id"] == "job_123"
    assert result["status"] == "PENDING"
    assert result["strategy_folder"] == "my_strategy_vwap"
    mock_queue.submit_job.assert_called_once()


def test_process_video_returns_playlist_jobs(monkeypatch):
    handler = VideoIngestAPIHandler()
    mock_processor = Mock()
    mock_processor.create_playlist_jobs.return_value = ["job_a", "job_b"]
    mock_queue = Mock()

    monkeypatch.setattr(
        "src.api.ide_handlers_video_ingest._get_runtime",
        lambda: (Mock(), mock_processor, mock_queue),
    )

    result = handler.process_video(
        url="https://www.youtube.com/playlist?list=PL123",
        strategy_name="Playlist Batch",
        is_playlist=True,
    )

    assert result["job_id"] == "job_a"
    assert result["job_ids"] == ["job_a", "job_b"]
    assert result["strategy_folder"] == "playlist_batch"


def test_get_job_status_adds_current_stage(monkeypatch):
    handler = VideoIngestAPIHandler()
    mock_status = Mock()
    mock_status.status.value = "ANALYZING"
    mock_status.to_dict.return_value = {
        "job_id": "job_abc",
        "status": "ANALYZING",
        "progress": 70,
    }
    mock_queue = Mock()
    mock_queue.get_job_status.return_value = mock_status

    monkeypatch.setattr(
        "src.api.ide_handlers_video_ingest._get_runtime",
        lambda: (Mock(), Mock(), mock_queue),
    )

    result = handler.get_job_status("job_abc")

    assert result["job_id"] == "job_abc"
    assert result["current_stage"] == "ANALYZING"


def test_get_auth_status_reports_openrouter(monkeypatch):
    handler = VideoIngestAPIHandler()

    class MockConfig:
        openrouter_api_key = "test-key"
        gemini_yolo_mode = True
        gemini_api_key = None
        qwen_api_key = None
        qwen_headless = True
        qwen_model = "qwen-vl-plus"

    monkeypatch.setattr(
        "src.api.ide_handlers_video_ingest.VideoIngestConfig.from_env",
        lambda: MockConfig(),
    )
    monkeypatch.setattr(
        "src.api.ide_handlers_video_ingest.GeminiCLIProvider.is_authenticated",
        lambda self: False,
    )

    status = handler.get_auth_status()

    assert status == {
        "openrouter": True,
        "gemini": False,
        "qwen": False,
    }


def test_processor_uses_gemini_oauth_when_no_api_keys(monkeypatch, tmp_path):
    class MockConfig:
        openrouter_api_key = None
        openrouter_model = "anthropic/claude-sonnet-4"
        qwen_api_key = None
        qwen_headless = True
        qwen_model = "qwen-vl-plus"
        gemini_api_key = None
        gemini_yolo_mode = True
        cache_dir = tmp_path / "cache"
        cache_max_size_gb = 1
        cache_max_age_days = 1
        max_retry_attempts = 1
        base_retry_delay = 0.1

    monkeypatch.setattr(
        "src.video_ingest.providers.GeminiCLIProvider.is_authenticated",
        lambda self: True,
    )

    processor = VideoIngestProcessor(config=MockConfig())

    assert any(isinstance(provider, GeminiCLIProvider) for provider in processor.providers)
    assert not any(isinstance(provider, OpenRouterProvider) for provider in processor.providers)
