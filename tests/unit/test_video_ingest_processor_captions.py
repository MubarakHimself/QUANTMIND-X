from pathlib import Path

from src.video_ingest.models import JobOptions, RateLimit, TimelineClip, TimelineOutput, VideoMetadata
from src.video_ingest.processor import VideoIngestProcessor


class _StubRateLimiter:
    def can_proceed(self, provider_name):
        return True

    def record_usage(self, provider_name):
        return None


class _CapturingProvider:
    def __init__(self):
        self.calls = []

    def analyze(self, frames, audio, prompt):
        self.calls.append({"frames": frames, "audio": audio, "prompt": prompt})
        return TimelineOutput(
            video_url="unknown",
            title="stub",
            duration_seconds=30,
            processed_at="2026-04-04T00:00:00",
            model_provider="stub",
            timeline=[
                TimelineClip(
                    clip_id=1,
                    timestamp_start="00:00:00",
                    timestamp_end="00:00:30",
                    transcript="stub transcript",
                    visual_description="stub visual",
                    frame_path=str(frames[0]) if frames else "",
                )
            ],
        )

    def get_rate_limit(self):
        return RateLimit(requests_per_day=None)


def _make_processor() -> VideoIngestProcessor:
    processor = VideoIngestProcessor.__new__(VideoIngestProcessor)
    processor.rate_limiter = _StubRateLimiter()
    processor.providers = []
    return processor


def test_parse_webvtt_and_slice_text_for_chunk(tmp_path):
    processor = _make_processor()
    caption_path = tmp_path / "captions.en.vtt"
    caption_path.write_text(
        "WEBVTT\n\n"
        "00:00:00.000 --> 00:00:08.000\n"
        "Welcome back traders\n\n"
        "00:00:08.000 --> 00:00:18.000\n"
        "Today we scalp London open\n\n"
        "00:15:00.000 --> 00:15:09.000\n"
        "Second chunk starts here\n",
        encoding="utf-8",
    )

    segments = processor._parse_caption_file(caption_path)

    assert len(segments) == 3
    assert processor._captions_for_window(segments, 0, 60) == "Welcome back traders\nToday we scalp London open"
    assert processor._captions_for_window(segments, 900, 960) == "Second chunk starts here"
    assert processor._captions_for_window(segments, 1200, 1260) is None


def test_analyze_content_prefers_captions_and_omits_audio(tmp_path):
    processor = _make_processor()
    provider = _CapturingProvider()
    processor.providers = [provider]

    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"frame")
    audio = tmp_path / "audio.mp3"
    audio.write_bytes(b"audio")

    timeline, provider_name = processor._analyze_content(
        [frame],
        audio,
        VideoMetadata(
            file_path=tmp_path / "video.mp4",
            duration=30,
            resolution="1920x1080",
            format="mp4",
            file_size=1,
            url="https://www.youtube.com/watch?v=abc123",
            title="Captioned Video",
        ),
        JobOptions(),
        captions_text="This is the caption transcript.",
    )

    assert provider_name == "_CapturingProvider"
    assert timeline.timeline[0].transcript == "stub transcript"
    assert provider.calls[0]["audio"] is None
    assert "Caption transcript" in provider.calls[0]["prompt"]
    assert "This is the caption transcript." in provider.calls[0]["prompt"]
