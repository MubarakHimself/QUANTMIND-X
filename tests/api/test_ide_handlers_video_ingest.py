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
    assert payload["status"] == "running"
    assert payload["current_stage"] == "video_ingest"
    assert payload["job_statuses"] == {"job_123": "PENDING"}

    meta_path = tmp_path / "shared_assets" / "strategies" / "scalping" / "single-videos" / "my_strategy_vwap" / ".meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["status"] == "processing"
    assert meta["has_video_ingest"] is False


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
    assert payload["job_statuses"] == {"job_a": "PENDING", "job_b": "PENDING"}
    assert payload["current_stage"] == "video_ingest"
    assert payload["status"] == "running"


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


def test_get_job_status_hydrates_manifest_and_meta_after_completion(monkeypatch, tmp_path):
    from src.api.wf1_artifacts import ensure_bundle

    _patch_wf1_root(monkeypatch, tmp_path)
    bundle = ensure_bundle(
        strategy_name="Completion Probe",
        is_playlist=False,
        workflow_id="wf1_complete_001",
        source_url="https://youtube.com/watch?v=done",
    )
    bundle.workflow_manifest_path.write_text(
        json.dumps(
            {
                "workflow_id": "wf1_complete_001",
                "workflow_type": "wf1_creation",
                "strategy_id": "completion_probe",
                "strategy_family": "scalping",
                "source_bucket": "single-videos",
                "job_ids": ["job_done"],
                "job_statuses": {"job_done": "PENDING"},
                "current_stage": "video_ingest",
                "status": "running",
            }
        ),
        encoding="utf-8",
    )

    mock_status = Mock()
    mock_status.status.value = "COMPLETED"
    mock_status.error = None
    mock_status.to_dict.return_value = {
        "job_id": "job_done",
        "status": "COMPLETED",
        "progress": 100,
        "result_path": str(bundle.timeline_dir / "job_done.json"),
    }
    mock_queue = Mock()
    mock_queue.get_job_status.return_value = mock_status

    monkeypatch.setattr(
        "src.api.ide_handlers_video_ingest._get_runtime",
        lambda: (Mock(), Mock(), mock_queue),
    )

    handler = VideoIngestAPIHandler()
    result = handler.get_job_status("job_done")

    assert result["status"] == "COMPLETED"
    assert result["current_stage"] == "research"

    workflow_manifest = json.loads(bundle.workflow_manifest_path.read_text(encoding="utf-8"))
    assert workflow_manifest["job_statuses"] == {"job_done": "COMPLETED"}
    assert workflow_manifest["status"] == "running"
    assert workflow_manifest["current_stage"] == "research"
    assert workflow_manifest["waiting_reason"] == "research"

    meta = json.loads(bundle.meta_path.read_text(encoding="utf-8"))
    assert meta["status"] == "processing"
    assert meta["has_video_ingest"] is True


def test_list_jobs_exposes_wf1_artifact_context(monkeypatch, tmp_path):
    from src.api.wf1_artifacts import ensure_bundle

    _patch_wf1_root(monkeypatch, tmp_path)
    bundle = ensure_bundle(
        strategy_name="Quota Probe",
        is_playlist=False,
        workflow_id="wf1_quota_probe",
        source_url="https://youtube.com/watch?v=quota",
    )
    raw_error = (
        "All providers failed. Last error: Gemini CLI execution failed:\n"
        "Attempt 1 failed: You have exhausted your capacity on this model.\n"
        "RetryableQuotaError: You have exhausted your capacity on this model."
    )
    bundle.workflow_manifest_path.write_text(
        json.dumps(
            {
                "workflow_id": "wf1_quota_probe",
                "workflow_type": "wf1_creation",
                "strategy_id": "quota_probe",
                "strategy_family": "scalping",
                "source_bucket": "single-videos",
                "job_ids": ["job_quota"],
                "job_statuses": {"job_quota": "FAILED"},
                "current_stage": "video_ingest",
                "status": "failed",
                "waiting_reason": None,
                "blocking_error": raw_error,
            }
        ),
        encoding="utf-8",
    )
    bundle.meta_path.write_text(
        json.dumps(
            {
                "name": "quota_probe",
                "workflow_id": "wf1_quota_probe",
                "strategy_id": "quota_probe",
                "strategy_family": "scalping",
                "source_bucket": "single-videos",
                "status": "quarantined",
                "has_video_ingest": False,
                "blocking_error": raw_error,
            }
        ),
        encoding="utf-8",
    )

    failed_job = Mock()
    failed_job.job_id = "job_quota"
    failed_job.status.value = "FAILED"
    failed_job.progress = 83
    failed_job.video_url = "https://youtube.com/watch?v=quota"
    failed_job.created_at.isoformat.return_value = "2026-04-04T10:15:00+00:00"
    failed_job.updated_at.isoformat.return_value = "2026-04-04T10:25:00+00:00"
    failed_job.result_path = None
    failed_job.error = raw_error
    failed_job.logs = []

    mock_queue = Mock()
    mock_queue.list_jobs.return_value = [failed_job]

    monkeypatch.setattr(
        "src.api.ide_handlers_video_ingest._get_runtime",
        lambda: (Mock(), Mock(), mock_queue),
    )

    handler = VideoIngestAPIHandler()

    result = handler.list_jobs(limit=20)

    assert len(result) == 1
    job = result[0]
    assert job["job_id"] == "job_quota"
    assert job["strategy_id"] == "quota_probe"
    assert job["strategy_asset_id"] == "strategies/scalping/single-videos/quota_probe"
    assert job["strategy_status"] == "quarantined"
    assert job["workflow_id"] == "wf1_quota_probe"
    assert job["workflow_status"] == "failed"
    assert job["current_stage"] == "video_ingest"
    assert job["blocking_error"] == "Provider quota exhausted during analysis. Switch provider/model or wait for quota reset."
    assert job["blocking_error_detail"] == raw_error


def test_list_jobs_summarizes_legacy_queue_errors_without_artifact_context(monkeypatch):
    raw_error = (
        "All providers failed. Last error: Gemini CLI execution failed:\n"
        "Attempt 1 failed: You have exhausted your capacity on this model.\n"
        "RetryableQuotaError: You have exhausted your capacity on this model."
    )

    failed_job = Mock()
    failed_job.job_id = "job_legacy"
    failed_job.status.value = "FAILED"
    failed_job.progress = 70
    failed_job.video_url = "https://youtube.com/watch?v=legacy"
    failed_job.created_at.isoformat.return_value = "2026-04-04T10:15:00+00:00"
    failed_job.updated_at.isoformat.return_value = "2026-04-04T10:25:00+00:00"
    failed_job.result_path = None
    failed_job.error = raw_error
    failed_job.logs = []

    mock_queue = Mock()
    mock_queue.list_jobs.return_value = [failed_job]

    monkeypatch.setattr(
        "src.api.ide_handlers_video_ingest._get_runtime",
        lambda: (Mock(), Mock(), mock_queue),
    )

    handler = VideoIngestAPIHandler()

    result = handler.list_jobs(limit=20)

    assert len(result) == 1
    job = result[0]
    assert job["job_id"] == "job_legacy"
    assert job["blocking_error"] == "Provider quota exhausted during analysis. Switch provider/model or wait for quota reset."
    assert job["blocking_error_detail"] == raw_error


def test_list_jobs_summarizes_provider_runtime_dependency_errors(monkeypatch):
    raw_error = (
        "All providers failed. Last error: OpenRouter API call failed: "
        "No module named 'urllib3.packages.six.moves'"
    )

    failed_job = Mock()
    failed_job.job_id = "job_provider_runtime"
    failed_job.status.value = "FAILED"
    failed_job.progress = 70
    failed_job.video_url = "https://youtube.com/watch?v=runtime"
    failed_job.created_at.isoformat.return_value = "2026-04-04T10:15:00+00:00"
    failed_job.updated_at.isoformat.return_value = "2026-04-04T10:25:00+00:00"
    failed_job.result_path = None
    failed_job.error = raw_error
    failed_job.logs = []

    mock_queue = Mock()
    mock_queue.list_jobs.return_value = [failed_job]

    monkeypatch.setattr(
        "src.api.ide_handlers_video_ingest._get_runtime",
        lambda: (Mock(), Mock(), mock_queue),
    )

    handler = VideoIngestAPIHandler()

    result = handler.list_jobs(limit=20)

    assert len(result) == 1
    job = result[0]
    assert job["job_id"] == "job_provider_runtime"
    assert job["blocking_error"] == "Provider runtime is missing a required dependency. Repair the provider environment before retrying."
    assert job["blocking_error_detail"] == raw_error


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


def test_get_runtime_hot_refreshes_processor_when_provider_config_changes(monkeypatch, tmp_path):
    import src.api.ide_handlers_video_ingest as handlers

    handlers._runtime.clear()

    class MockConfig:
        def __init__(self, key: str, model: str):
            self.openrouter_api_key = key
            self.openrouter_model = model
            self.openrouter_base_url = "https://openrouter.ai/api/v1"
            self.qwen_api_key = None
            self.qwen_headless = True
            self.qwen_model = "qwen-vl-plus"
            self.gemini_api_key = None
            self.gemini_yolo_mode = True
            self.qwen_requests_per_day = 2000
            self.cache_dir = tmp_path / "cache"
            self.cache_max_size_gb = 1
            self.cache_max_age_days = 1
            self.max_concurrent_jobs = 2
            self.job_db_path = tmp_path / "jobs.db"
            self.output_dir = tmp_path / "out"
            self.default_frame_interval = 30
            self.default_audio_bitrate = "128k"
            self.default_audio_channels = 1
            self.max_retry_attempts = 1
            self.base_retry_delay = 0.1
            self.log_level = "INFO"
            self.log_file = None

        def runtime_fingerprint(self):
            return f"{self.openrouter_api_key}:{self.openrouter_model}"

    configs = iter([
        MockConfig("key-one", "google/gemini-2.0-flash-lite-001"),
        MockConfig("key-two", "google/gemini-2.5-flash-lite"),
    ])

    processor_instances = []

    class FakeProcessor:
        def __init__(self, config):
            self.config = config
            self.callbacks = []
            processor_instances.append(self)

        def set_status_callback(self, callback):
            self.callbacks.append(callback)

    class FakeQueue:
        def __init__(self, config):
            self.config = config
            self.db_path = config.job_db_path
            self.max_concurrent = config.max_concurrent_jobs
            self.started = 0
            self.job_processors = []

        def set_job_processor(self, processor):
            self.job_processors.append(processor)

        def start(self):
            self.started += 1

        def get_active_job_count(self):
            return 0

        def stop(self, wait=True):
            self.stopped = wait

    monkeypatch.setattr(
        handlers.VideoIngestConfig,
        "from_env",
        classmethod(lambda cls: next(configs)),
    )
    monkeypatch.setattr(handlers, "VideoIngestProcessor", FakeProcessor)
    monkeypatch.setattr(handlers, "JobQueueManager", FakeQueue)

    config_one, processor_one, queue_one = handlers.refresh_runtime(force=True)
    config_two, processor_two, queue_two = handlers.refresh_runtime(force=False)

    assert config_one.openrouter_api_key == "key-one"
    assert config_two.openrouter_api_key == "key-two"
    assert processor_one is not processor_two
    assert queue_one is queue_two
    assert queue_two.config is config_two
    assert len(queue_two.job_processors) == 2
