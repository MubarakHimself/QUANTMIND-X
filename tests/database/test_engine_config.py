import importlib
from unittest.mock import MagicMock

import sqlalchemy
import sqlalchemy.orm as orm

import src.config as config_module


def _reload_engine_module(monkeypatch, database_url: str):
    monkeypatch.setenv("DATABASE_URL", database_url)
    config_module._settings = None

    create_engine_mock = MagicMock(return_value=MagicMock(name="engine"))
    sessionmaker_mock = MagicMock(return_value=MagicMock(name="session_factory"))
    scoped_session_mock = MagicMock(return_value=MagicMock(name="scoped_session"))

    monkeypatch.setattr(sqlalchemy, "create_engine", create_engine_mock)
    monkeypatch.setattr(orm, "sessionmaker", sessionmaker_mock)
    monkeypatch.setattr(orm, "scoped_session", scoped_session_mock)
    monkeypatch.setattr(
        sqlalchemy.event,
        "listens_for",
        lambda *args, **kwargs: (lambda fn: fn),
    )

    import src.database.engine as engine_module

    return importlib.reload(engine_module), create_engine_mock


def test_postgres_engine_avoids_sqlite_only_kwargs(monkeypatch):
    _, create_engine_mock = _reload_engine_module(
        monkeypatch,
        "postgresql://postgres:password@db.internal:5432/quantmind_hot",
    )

    kwargs = create_engine_mock.call_args.kwargs

    assert create_engine_mock.call_args.args[0] == "postgresql://postgres:password@db.internal:5432/quantmind_hot"
    assert "connect_args" not in kwargs
    assert "poolclass" not in kwargs


def test_sqlite_engine_keeps_sqlite_specific_kwargs(monkeypatch):
    _, create_engine_mock = _reload_engine_module(
        monkeypatch,
        "sqlite:///data/quantmind.db",
    )

    kwargs = create_engine_mock.call_args.kwargs

    assert kwargs["connect_args"] == {"check_same_thread": False}
    assert kwargs["poolclass"].__name__ == "StaticPool"
