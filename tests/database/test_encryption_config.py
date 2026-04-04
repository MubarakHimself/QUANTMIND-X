from pathlib import Path

from src.database import encryption


def test_machine_key_file_uses_env_override(monkeypatch, tmp_path):
    key_file = tmp_path / "secure-state" / "machine.key"
    monkeypatch.setenv("QUANTMIND_MACHINE_KEY_FILE", str(key_file))
    encryption.SecureStorage._instance = None

    key = encryption._get_or_create_machine_key()

    assert key
    assert key_file.exists()
    assert key_file.read_text().strip() == key
