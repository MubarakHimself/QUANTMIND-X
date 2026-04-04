import json
from pathlib import Path

from src.video_ingest.models import JobOptions, TimelineClip, TimelineOutput, VideoMetadata
from src.video_ingest.processor import VideoIngestProcessor


def _make_processor() -> VideoIngestProcessor:
    processor = VideoIngestProcessor.__new__(VideoIngestProcessor)
    processor._update_status = lambda *args, **kwargs: None
    return processor


def test_persist_chunk_plan_writes_job_scoped_manifest(tmp_path):
    processor = _make_processor()
    artifact_root = tmp_path / "strategy"
    options = JobOptions(artifact_root=artifact_root)
    chunk_plan = [
        {"start": 0, "end": 900, "label": "part_1"},
        {"start": 900, "end": 1800, "label": "part_2"},
    ]

    processor._persist_chunk_plan("job_123", chunk_plan, options)

    manifest_path = artifact_root / "source" / "chunks" / "job_123.json"
    assert manifest_path.exists()

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["job_id"] == "job_123"
    assert payload["chunk_count"] == 2
    assert payload["total_duration_seconds"] == 1800
    assert payload["chunks"] == [
        {
            "chunk_id": 1,
            "label": "part_1",
            "start_seconds": 0,
            "end_seconds": 900,
            "duration_seconds": 900,
            "status": "pending",
            "timeline_path": None,
        },
        {
            "chunk_id": 2,
            "label": "part_2",
            "start_seconds": 900,
            "end_seconds": 1800,
            "duration_seconds": 900,
            "status": "pending",
            "timeline_path": None,
        },
    ]


def test_process_chunked_video_rebases_timestamps_and_persists_chunk_timelines(tmp_path):
    processor = _make_processor()
    artifact_root = tmp_path / "strategy"
    options = JobOptions(artifact_root=artifact_root)
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"video")

    def fake_extract_video_chunk(
        video_file: Path,
        start_seconds: int,
        end_seconds: int,
        output_path: Path,
    ) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(f"{start_seconds}-{end_seconds}".encode("utf-8"))
        return output_path

    processor._extract_video_chunk = fake_extract_video_chunk
    processor._extract_frames = lambda *args, **kwargs: [tmp_path / "frame.jpg"]
    processor._extract_audio = lambda *args, **kwargs: tmp_path / "audio.mp3"

    def fake_analyze(frames, audio_path, metadata, options, captions_text=None):
        return (
            TimelineOutput(
                video_url=metadata.url,
                title=metadata.title or "Untitled",
                duration_seconds=int(metadata.duration),
                processed_at="2026-04-04T00:00:00",
                model_provider="openrouterprovider",
                timeline=[
                    TimelineClip(
                        clip_id=1,
                        timestamp_start="00:00:00",
                        timestamp_end="00:00:30",
                        transcript=f"segment {int(metadata.duration)}",
                        visual_description=metadata.file_path.parent.name,
                        frame_path="frame.jpg",
                    )
                ],
            ),
            "OpenRouterProvider",
        )

    processor._analyze_content = fake_analyze

    chunk_plan = [
        {"start": 0, "end": 900, "label": "part_1"},
        {"start": 900, "end": 1800, "label": "part_2"},
    ]
    processor._persist_chunk_plan("job_123", chunk_plan, options)

    timeline, provider_used = processor._process_chunked_video(
        job_id="job_123",
        video_metadata=VideoMetadata(
            file_path=video_path,
            duration=1800,
            resolution="1920x1080",
            format="mp4",
            file_size=video_path.stat().st_size,
            url="https://www.youtube.com/watch?v=abc123",
            title="London Scalper",
        ),
        chunk_plan=chunk_plan,
        work_dir=tmp_path / "work",
        options=options,
    )

    assert provider_used == "OpenRouterProvider"
    assert [clip.clip_id for clip in timeline.timeline] == [1, 2]
    assert [clip.timestamp_start for clip in timeline.timeline] == ["00:00:00", "00:15:00"]
    assert [clip.timestamp_end for clip in timeline.timeline] == ["00:00:30", "00:15:30"]

    chunk_a = artifact_root / "source" / "timelines" / "chunks" / "job_123" / "part_1.json"
    chunk_b = artifact_root / "source" / "timelines" / "chunks" / "job_123" / "part_2.json"
    assert chunk_a.exists()
    assert chunk_b.exists()

    manifest_path = artifact_root / "source" / "chunks" / "job_123.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["chunks"][0]["status"] == "completed"
    assert payload["chunks"][0]["timeline_path"] == "source/timelines/chunks/job_123/part_1.json"
    assert payload["chunks"][1]["status"] == "completed"
    assert payload["chunks"][1]["timeline_path"] == "source/timelines/chunks/job_123/part_2.json"
