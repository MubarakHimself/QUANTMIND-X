import json
from unittest.mock import Mock

from src.api.ide_handlers_video_ingest import VideoIngestAPIHandler
from src.video_ingest.processor import VideoIngestProcessor
from src.video_ingest.models import VideoIngestConfig
from src.video_ingest.providers import GeminiCLIProvider, OpenRouterProvider


def _patch_wf1_root(monkeypatch, tmp_path):
    assets_dir = tmp_path / "shared_assets"
    monkeypatch.setattr("src.api.wf1_artifacts.ASSETS_DIR", assets_dir)
    monkeypatch.setattr("src.api.wf1_artifacts.WF1_STRATEGIES_ROOT", assets_dir / "strategies")
    return assets_dir


def test_process_video_submits_real_queue_job(monkeypatch, tmp_path):
    handler = VideoIngestAPIHandler()
    mock_queue = Mock()
    mock_queue.submit_job.return_value = "job_123"
    _patch_wf1_root(monkeypatch, tmp_path)

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
    options = mock_queue.submit_job.call_args.kwargs["options"]
    assert options.workflow_id.startswith("wf1_")
    assert options.strategy_id == "my_strategy_vwap"
    assert options.strategy_family == "scalping"
    assert options.source_bucket == "single-videos"
    assert options.output_dir == tmp_path / "shared_assets" / "strategies" / "scalping" / "single-videos" / "my_strategy_vwap" / "source" / "timelines"

    workflow_manifest = tmp_path / "shared_assets" / "strategies" / "scalping" / "single-videos" / "my_strategy_vwap" / "workflow" / "manifest.json"
    assert workflow_manifest.exists()
    payload = json.loads(workflow_manifest.read_text(encoding="utf-8"))
    assert payload["job_ids"] == ["job_123"]
    assert payload["strategy_id"] == "my_strategy_vwap"


def test_process_video_returns_playlist_jobs(monkeypatch, tmp_path):
    handler = VideoIngestAPIHandler()
    mock_processor = Mock()
    mock_processor.create_playlist_jobs.return_value = ["job_a", "job_b"]
    mock_queue = Mock()
    _patch_wf1_root(monkeypatch, tmp_path)

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
    options = mock_processor.create_playlist_jobs.call_args.kwargs["options"]
    assert options.strategy_id == "playlist_batch"
    assert options.source_bucket == "playlists"
    assert options.output_dir == tmp_path / "shared_assets" / "strategies" / "scalping" / "playlists" / "playlist_batch" / "source" / "timelines"

    workflow_manifest = tmp_path / "shared_assets" / "strategies" / "scalping" / "playlists" / "playlist_batch" / "workflow" / "manifest.json"
    payload = json.loads(workflow_manifest.read_text(encoding="utf-8"))
    assert payload["job_ids"] == ["job_a", "job_b"]


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


def test_video_ingest_config_reads_openrouter_from_provider_router(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)

    class MockProvider:
        api_key = "or-key"
        base_url = "https://openrouter.ai/api/v1"
        model_list = [{"id": "google/gemini-2.0-flash-lite-001"}]

    class MockRouter:
        def get_provider(self, provider_type):
            assert provider_type == "openrouter"
            return MockProvider()

    monkeypatch.setattr("src.agents.providers.router.get_router", lambda: MockRouter())

    config = VideoIngestConfig.from_env()

    assert config.openrouter_api_key == "or-key"
    assert config.openrouter_model == "google/gemini-2.0-flash-lite-001"
    assert config.openrouter_base_url == "https://openrouter.ai/api/v1"
