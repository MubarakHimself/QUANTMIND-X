# tests/agents/tools/test_trd_variants.py
import pytest
from src.agents.tools.strategies_yt.trd_tools import TRDGenerator


def test_generate_both_variants():
    """TRDGenerator should generate both vanilla and spiced variants."""
    generator = TRDGenerator()
    video_context = {"url": "https://youtube.com/watch?v=123", "title": "Test Strategy"}

    result = generator.generate_trd(video_context, generate_variants="both")

    assert "vanilla" in result
    assert "spiced" in result
    assert result["vanilla"]["sources"] == ["video"]
    assert "article" in result["spiced"]["sources"]


def test_vanilla_trd_contains_video_only():
    """Vanilla TRD should contain only video-derived data."""
    generator = TRDGenerator()
    video_context = {"url": "https://youtube.com/watch?v=123", "title": "Test"}

    result = generator.generate_trd(video_context, generate_variants="vanilla")

    assert "sources" in result
    assert result["sources"] == ["video"]
    assert "articles" not in result or result.get("articles") == []
