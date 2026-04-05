import json

from src.api.wf1_artifacts import ensure_bundle, iter_strategy_roots


def test_ensure_bundle_creates_canonical_wf1_tree(monkeypatch, tmp_path):
    assets_dir = tmp_path / "shared_assets"
    legacy_dir = tmp_path / "strategies"
    monkeypatch.setattr("src.api.wf1_artifacts.ASSETS_DIR", assets_dir)
    monkeypatch.setattr("src.api.wf1_artifacts.WF1_STRATEGIES_ROOT", assets_dir / "strategies")
    monkeypatch.setattr("src.api.wf1_artifacts.STRATEGIES_DIR", legacy_dir)

    bundle = ensure_bundle(
        strategy_name="London Scalper",
        is_playlist=False,
        workflow_id="wf1_test_001",
        source_url="https://youtube.com/watch?v=abc123",
    )

    assert bundle.root == assets_dir / "strategies" / "scalping" / "single-videos" / "london_scalper"
    assert bundle.timeline_dir.exists()
    assert bundle.research_dir.exists()
    assert bundle.development_dir.exists()
    assert bundle.variants_dir.exists()
    assert bundle.compilation_dir.exists()
    assert bundle.backtests_dir.exists()
    assert bundle.reports_dir.exists()
    assert bundle.workflow_dir.exists()

    meta = json.loads(bundle.meta_path.read_text(encoding="utf-8"))
    assert meta["workflow_id"] == "wf1_test_001"
    assert meta["strategy_id"] == "london_scalper"
    assert meta["strategy_family"] == "scalping"
    assert meta["source_bucket"] == "single-videos"
    assert meta["artifact_root"] == "strategies/scalping/single-videos/london_scalper"

    request_manifest = json.loads(bundle.request_manifest_path.read_text(encoding="utf-8"))
    assert request_manifest["artifact_type"] == "video_ingest_request"
    assert request_manifest["source_url"] == "https://youtube.com/watch?v=abc123"

    workflow_manifest = json.loads(bundle.workflow_manifest_path.read_text(encoding="utf-8"))
    assert workflow_manifest["workflow_type"] == "wf1_creation"
    assert workflow_manifest["current_stage"] == "video_ingest"


def test_iter_strategy_roots_migrates_legacy_strategy_tree(monkeypatch, tmp_path):
    assets_dir = tmp_path / "shared_assets"
    legacy_dir = tmp_path / "strategies"
    monkeypatch.setattr("src.api.wf1_artifacts.ASSETS_DIR", assets_dir)
    monkeypatch.setattr("src.api.wf1_artifacts.WF1_STRATEGIES_ROOT", assets_dir / "strategies")
    monkeypatch.setattr("src.api.wf1_artifacts.STRATEGIES_DIR", legacy_dir)

    legacy_root = legacy_dir / "legacy_scalper"
    (legacy_root / "ea").mkdir(parents=True, exist_ok=True)
    (legacy_root / "backtest").mkdir(parents=True, exist_ok=True)
    (legacy_root / "trd").mkdir(parents=True, exist_ok=True)
    (legacy_root / "status.json").write_text(
        json.dumps({"status": "ready", "created_at": "2026-04-04T00:00:00+00:00"}),
        encoding="utf-8",
    )
    (legacy_root / "ea" / "LegacyEA.mq5").write_text("// code", encoding="utf-8")
    (legacy_root / "backtest" / "report.json").write_text("{}", encoding="utf-8")
    (legacy_root / "trd" / "trd.md").write_text("# TRD", encoding="utf-8")

    roots = list(iter_strategy_roots() or [])

    assert len(roots) == 1
    root = roots[0]
    assert root == assets_dir / "strategies" / "scalping" / "single-videos" / "legacy_scalper"
    assert (root / "development" / "LegacyEA.mq5").exists()
    assert (root / "backtests" / "report.json").exists()
    assert (root / "research" / "trd" / "trd.md").exists()
    assert not legacy_root.exists()
