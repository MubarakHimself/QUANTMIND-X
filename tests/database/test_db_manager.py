from unittest.mock import patch

from src.database.db_manager import DBManager, HOTDBManager, DEFAULT_SQLITE_PATH


def test_hot_db_manager_falls_back_to_sqlite_when_driver_missing():
    calls = []

    def fake_init(self, db_url=None, is_hot=False):
        calls.append((db_url, is_hot))
        if len(calls) == 1:
            raise ModuleNotFoundError("psycopg2")
        self._db_url = db_url
        self.engine = None
        self.SessionLocal = None

    with patch.object(DBManager, "__init__", fake_init):
        manager = HOTDBManager("postgresql://postgres:password@localhost:5432/quantmind_hot")

    assert calls == [
        ("postgresql://postgres:password@localhost:5432/quantmind_hot", False),
        (DEFAULT_SQLITE_PATH, False),
    ]
    assert manager._db_url == DEFAULT_SQLITE_PATH
