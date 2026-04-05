from src.api.ide_handlers_strategy import StrategyAPIHandler


def _patch_roots(monkeypatch, tmp_path):
    assets_dir = tmp_path / "shared_assets"
    strategies_root = assets_dir / "strategies"
    legacy_root = tmp_path / "strategies"
    monkeypatch.setattr("src.api.wf1_artifacts.ASSETS_DIR", assets_dir)
    monkeypatch.setattr("src.api.wf1_artifacts.WF1_STRATEGIES_ROOT", strategies_root)
    monkeypatch.setattr("src.api.wf1_artifacts.STRATEGIES_DIR", legacy_root)
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
    assert payload["asset_id"] == "strategies/scalping/single-videos/london_scalper"
    assert payload["relative_root"] == "scalping/single-videos/london_scalper"
    assert payload["source_artifacts"]["audio_files"] == ["source/audio/audio.mp3"]
    assert payload["source_artifacts"]["caption_files"] == ["source/captions/captions.en.vtt"]
    assert payload["source_artifacts"]["timeline_files"] == ["source/timelines/job_123.json"]
    assert payload["source_artifacts"]["chunk_timeline_files"] == ["source/timelines/chunks/job_123/part_1.json"]


def test_get_strategy_exposes_stage_artifacts_for_wf1_bundle(monkeypatch, tmp_path):
    from src.api.wf1_artifacts import ensure_bundle

    _patch_roots(monkeypatch, tmp_path)
    bundle = ensure_bundle(
        strategy_name="Me At The Zoo Smoke",
        is_playlist=False,
        workflow_id="wf1_test_stage_001",
        source_url="https://youtube.com/watch?v=jNQXAC9IVRw",
    )
    (bundle.research_dir / "trd").mkdir(parents=True, exist_ok=True)
    (bundle.development_dir / "ea").mkdir(parents=True, exist_ok=True)
    (bundle.variants_dir / "vanilla").mkdir(parents=True, exist_ok=True)
    (bundle.compilation_dir / "builds").mkdir(parents=True, exist_ok=True)
    (bundle.reports_dir / "backtests").mkdir(parents=True, exist_ok=True)
    (bundle.research_dir / "trd" / "strategy.trd.md").write_text("# TRD", encoding="utf-8")
    (bundle.development_dir / "ea" / "main.mq5").write_text("// mq5", encoding="utf-8")
    (bundle.variants_dir / "vanilla" / "london_scalper_v1.mq5").write_text("// variant", encoding="utf-8")
    (bundle.compilation_dir / "builds" / "main.ex5").write_bytes(b"ex5")
    (bundle.reports_dir / "backtests" / "summary.md").write_text("# Report", encoding="utf-8")

    handler = StrategyAPIHandler()
    payload = handler.get_strategy("strategies/scalping/single-videos/me_at_the_zoo_smoke")

    assert payload is not None
    assert payload["has_trd"] is True
    assert payload["has_ea"] is True
    assert payload["has_variants"] is True
    assert payload["has_compilation"] is True
    assert payload["has_reports"] is True
    assert payload["research_files"] == ["research/trd/strategy.trd.md"]
    assert payload["development_files"] == ["development/ea/main.mq5"]
    assert payload["variant_files"] == ["variants/vanilla/london_scalper_v1.mq5"]
    assert payload["compilation_files"] == ["compilation/builds/main.ex5"]
    assert payload["report_files"] == ["reports/backtests/summary.md"]


def test_get_strategy_resolves_canonical_asset_id_without_name_collisions(monkeypatch, tmp_path):
    from src.api.wf1_artifacts import ensure_bundle

    _patch_roots(monkeypatch, tmp_path)
    single_bundle = ensure_bundle(
        strategy_name="London Scalper",
        is_playlist=False,
        workflow_id="wf1_single",
        source_url="https://youtube.com/watch?v=single",
    )
    playlist_bundle = ensure_bundle(
        strategy_name="London Scalper",
        is_playlist=True,
        workflow_id="wf1_playlist",
        source_url="https://youtube.com/playlist?list=abc",
    )
    (single_bundle.source_dir / "captions").mkdir(parents=True, exist_ok=True)
    (single_bundle.source_dir / "captions" / "single.vtt").write_text("WEBVTT", encoding="utf-8")
    (playlist_bundle.source_dir / "audio").mkdir(parents=True, exist_ok=True)
    (playlist_bundle.source_dir / "audio" / "playlist.mp3").write_bytes(b"audio")

    handler = StrategyAPIHandler()

    single_payload = handler.get_strategy("strategies/scalping/single-videos/london_scalper")
    playlist_payload = handler.get_strategy("strategies/scalping/playlists/london_scalper")

    assert single_payload is not None
    assert playlist_payload is not None
    assert single_payload["asset_id"] == "strategies/scalping/single-videos/london_scalper"
    assert playlist_payload["asset_id"] == "strategies/scalping/playlists/london_scalper"
    assert single_payload["has_source_captions"] is True
    assert single_payload["has_source_audio"] is False
    assert playlist_payload["has_source_audio"] is True
    assert playlist_payload["has_source_captions"] is False


def test_get_strategy_exposes_quarantine_blocking_error_from_meta(monkeypatch, tmp_path):
    from src.api.wf1_artifacts import ensure_bundle
    import json

    _patch_roots(monkeypatch, tmp_path)
    bundle = ensure_bundle(
        strategy_name="Quota Probe",
        is_playlist=False,
        workflow_id="wf1_quota_probe",
        source_url="https://youtube.com/watch?v=quota",
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
                "has_video_ingest": True,
                "blocking_error": "Provider quota exhausted during analysis. Switch provider/model or wait for quota reset.",
                "blocking_error_detail": "raw provider trace",
            }
        ),
        encoding="utf-8",
    )

    handler = StrategyAPIHandler()
    payload = handler.get_strategy("strategies/scalping/single-videos/quota_probe")

    assert payload is not None
    assert payload["status"] == "quarantined"
    assert payload["blocking_error"] == "Provider quota exhausted during analysis. Switch provider/model or wait for quota reset."
    assert payload["blocking_error_detail"] == "raw provider trace"


def test_list_strategies_hides_placeholder_wf1_roots_without_operator_visible_signal(monkeypatch, tmp_path):
    from src.api.wf1_artifacts import ensure_bundle

    _patch_roots(monkeypatch, tmp_path)

    ensure_bundle(
        strategy_name="probe",
        is_playlist=False,
        workflow_id="wf1_probe",
    )
    visible = ensure_bundle(
        strategy_name="London Scalper",
        is_playlist=False,
        workflow_id="wf1_visible",
        source_url="https://youtube.com/watch?v=abc123",
    )

    handler = StrategyAPIHandler()
    payload = handler.list_strategies()

    assert [item["asset_id"] for item in payload] == [
        "strategies/scalping/single-videos/london_scalper"
    ]
    assert payload[0]["workflow_id"] == visible.workflow_id
