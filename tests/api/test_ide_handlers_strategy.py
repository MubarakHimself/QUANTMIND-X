from src.api.ide_handlers_strategy import StrategyAPIHandler


def _patch_roots(monkeypatch, tmp_path):
    assets_dir = tmp_path / "shared_assets"
    strategies_root = assets_dir / "strategies"
    legacy_root = tmp_path / "strategies"
    monkeypatch.setattr("src.api.wf1_artifacts.ASSETS_DIR", assets_dir)
    monkeypatch.setattr("src.api.wf1_artifacts.WF1_STRATEGIES_ROOT", strategies_root)
    monkeypatch.setattr("src.api.ide_handlers_strategy.STRATEGIES_DIR", legacy_root)
    return assets_dir, strategies_root, legacy_root


def test_get_strategy_exposes_source_artifacts_for_wf1_bundle(monkeypatch, tmp_path):
    from src.api.wf1_artifacts import ensure_bundle

    _patch_roots(monkeypatch, tmp_path)
    bundle = ensure_bundle(
        strategy_name="London Scalper",
        is_playlist=False,
        workflow_id="wf1_test_002",
        source_url="https://youtube.com/watch?v=abc123",
    )
    (bundle.source_dir / "audio").mkdir(parents=True, exist_ok=True)
    (bundle.source_dir / "captions").mkdir(parents=True, exist_ok=True)
    (bundle.timeline_dir / "chunks" / "job_123").mkdir(parents=True, exist_ok=True)
    (bundle.source_dir / "audio" / "audio.mp3").write_bytes(b"audio")
    (bundle.source_dir / "captions" / "captions.en.vtt").write_text("WEBVTT", encoding="utf-8")
    (bundle.timeline_dir / "job_123.json").write_text("{}", encoding="utf-8")
    (bundle.timeline_dir / "chunks" / "job_123" / "part_1.json").write_text("{}", encoding="utf-8")
    (bundle.source_dir / "chunks").mkdir(parents=True, exist_ok=True)
    (bundle.source_dir / "chunks" / "job_123.json").write_text("{}", encoding="utf-8")

    handler = StrategyAPIHandler()
    payload = handler.get_strategy("london_scalper")

    assert payload is not None
    assert payload["has_video_ingest"] is True
    assert payload["has_source_audio"] is True
    assert payload["has_source_captions"] is True
    assert payload["has_chunk_manifest"] is True
    assert payload["source_artifacts"]["audio_files"] == ["source/audio/audio.mp3"]
    assert payload["source_artifacts"]["caption_files"] == ["source/captions/captions.en.vtt"]
    assert payload["source_artifacts"]["timeline_files"] == ["source/timelines/job_123.json"]
    assert payload["source_artifacts"]["chunk_timeline_files"] == ["source/timelines/chunks/job_123/part_1.json"]
