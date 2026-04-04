import types

from src.video_ingest.providers import OpenRouterProvider


def test_openrouter_direct_call_uses_input_audio_for_audio_models(monkeypatch, tmp_path):
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"frame-data")
    audio = tmp_path / "audio.mp3"
    audio.write_bytes(b"audio-data")

    captured = {}

    class MockResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": "Transcript: hello | Visual: chart on screen"
                        }
                    }
                ]
            }

    def fake_post(url, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["payload"] = json
        return MockResponse()

    monkeypatch.setitem(__import__("sys").modules, "requests", types.SimpleNamespace(post=fake_post))

    provider = OpenRouterProvider(
        api_key="test-key",
        model="google/gemini-2.0-flash-lite-001",
    )

    timeline = provider._call_openrouter_api_direct([frame], audio, "Analyze this clip")

    assert timeline.timeline[0].transcript == "hello"
    content = captured["payload"]["messages"][1]["content"]
    assert content[0] == {"type": "text", "text": "Analyze this clip"}
    assert content[1]["type"] == "image_url"
    assert content[2]["type"] == "input_audio"
    assert content[2]["inputAudio"]["format"] == "mp3"
    assert content[2]["inputAudio"]["data"]
