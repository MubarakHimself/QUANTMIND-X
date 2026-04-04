from src.monitoring.tracing import get_trace_exporter_endpoint


def test_trace_exporter_endpoint_defaults_to_otel_collector(monkeypatch):
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("GRAFANA_TEMPO_OTLP_ENDPOINT", raising=False)

    assert get_trace_exporter_endpoint() == "http://otel-collector:4317"


def test_trace_exporter_endpoint_prefers_otel_env(monkeypatch):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "https://otel.quantmindx.local")
    monkeypatch.setenv("GRAFANA_TEMPO_OTLP_ENDPOINT", "https://tempo.quantmindx.local")

    assert get_trace_exporter_endpoint() == "https://otel.quantmindx.local"
