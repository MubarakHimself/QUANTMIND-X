import pytest
from unittest.mock import patch

from src.agents.departments.department_mail import (
    DepartmentMailService,
    RedisConnectionFailedError,
    RedisDepartmentMailService,
    get_mail_service,
    reset_mail_service,
)


@pytest.fixture(autouse=True)
def reset_mail_service_cache():
    """Ensure each test starts with a clean mail service cache."""
    reset_mail_service()
    yield
    reset_mail_service()


def test_get_mail_service_falls_back_to_sqlite(tmp_path):
    """When Redis is unreachable, the factory should return the SQLite service."""
    with patch.object(RedisDepartmentMailService, "health_check", return_value=False):
        mail_service = get_mail_service(
            db_path=str(tmp_path / "mail.db"),
            use_redis=True,
        )

    assert isinstance(mail_service, DepartmentMailService)


def test_force_redis_flag_prevents_fallback(tmp_path):
    """force_redis=True must raise instead of falling back to SQLite."""
    with patch.object(RedisDepartmentMailService, "health_check", return_value=False):
        with pytest.raises(RedisConnectionFailedError):
            get_mail_service(
                db_path=str(tmp_path / "mail.db"),
                use_redis=True,
                force_redis=True,
            )
