from src.video_ingest.models import JobOptions, VideoIngestConfig
from src.video_ingest.processor import VideoIngestProcessor


def _make_processor(model: str) -> VideoIngestProcessor:
    processor = VideoIngestProcessor.__new__(VideoIngestProcessor)
    processor.config = VideoIngestConfig(openrouter_model=model)
    return processor


def test_chunk_plan_uses_smaller_budget_for_cheaper_flash_lite_model():
    lite = _make_processor("google/gemini-2.0-flash-lite-001")
    richer = _make_processor("google/gemini-2.5-flash-lite")

    lite_chunks = lite._build_chunk_plan(4 * 60 * 60, JobOptions())
    richer_chunks = richer._build_chunk_plan(4 * 60 * 60, JobOptions())

    assert len(lite_chunks) == 8
    assert len(richer_chunks) == 6
    assert lite_chunks[-1]["end"] == 4 * 60 * 60
    assert richer_chunks[-1]["end"] == 4 * 60 * 60


def test_chunk_plan_falls_back_to_default_budget_for_unknown_model():
    processor = _make_processor("unknown/model")

    chunks = processor._build_chunk_plan(2 * 60 * 60, JobOptions())

    assert len(chunks) == 4
    assert chunks[0]["start"] == 0
    assert chunks[-1]["end"] == 2 * 60 * 60
