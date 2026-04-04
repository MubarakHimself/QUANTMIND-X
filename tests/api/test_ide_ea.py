import asyncio
from pathlib import Path

from fastapi import HTTPException

from src.api.ide_ea import get_ea_details, list_eas


def test_list_eas_reads_real_wf1_files(monkeypatch, tmp_path):
    strategy_root = tmp_path / "shared_assets" / "strategies" / "scalping" / "single-videos" / "london_scalper"
    dev_dir = strategy_root / "development"
    dev_dir.mkdir(parents=True, exist_ok=True)
    (dev_dir / "LondonScalper.mq5").write_text("// ea", encoding="utf-8")

    monkeypatch.setattr("src.api.ide_ea.iter_strategy_roots", lambda: iter([strategy_root]))

    response = asyncio.run(list_eas(limit=20, offset=0))

    assert response.total == 1
    assert response.items[0]["id"] == "LondonScalper"
    assert response.items[0]["strategy_id"] == "london_scalper"
    assert response.items[0]["status"] == "ready"


def test_get_ea_details_404_when_missing(monkeypatch):
    monkeypatch.setattr("src.api.ide_ea.iter_strategy_roots", lambda: iter([]))

    try:
        asyncio.run(get_ea_details("missing"))
        raise AssertionError("expected HTTPException")
    except HTTPException as exc:
        assert exc.status_code == 404
