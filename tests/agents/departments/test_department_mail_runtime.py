"""
Runtime tests for department mail service selection and fallback behavior.
"""

from pathlib import Path
from unittest.mock import patch

from src.agents.departments.department_mail import (
    DepartmentMailService,
    MessageType,
    RedisDepartmentMailService,
    reset_mail_service,
    reset_redis_mail_service,
)


class TestDepartmentMailRuntimeFallback:
    def teardown_method(self):
        reset_mail_service()
        reset_redis_mail_service()

    def test_get_mail_service_falls_back_to_sqlite_when_redis_unavailable(self, tmp_path):
        from src.agents.departments.department_mail import get_mail_service

        db_path = tmp_path / "department_mail.db"

        with patch.object(RedisDepartmentMailService, "health_check", return_value=False):
            service = get_mail_service(db_path=str(db_path), use_redis=True)

        assert isinstance(service, DepartmentMailService)
        assert service.db_path == Path(db_path)

    def test_get_mail_service_uses_redis_when_connection_is_healthy(self):
        from src.agents.departments.department_mail import get_mail_service

        with patch.object(RedisDepartmentMailService, "health_check", return_value=True):
            service = get_mail_service(use_redis=True)

        assert isinstance(service, RedisDepartmentMailService)

    def test_sqlite_mark_read_accepts_department_kwarg(self, tmp_path):
        service = DepartmentMailService(db_path=str(tmp_path / "department_mail.db"))
        message = service.send(
            from_dept="research",
            to_dept="development",
            type=MessageType.DISPATCH,
            subject="Test task",
            body="Test body",
        )

        assert service.mark_read(message.id, dept="development") is True
