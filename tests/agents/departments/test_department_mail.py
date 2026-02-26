"""
Tests for Department Mail Service

Task Group: Core Infrastructure - Cross-department messaging
"""
import pytest
import tempfile
import os
from pathlib import Path


class TestDepartmentMailSchema:
    """Test mail service schema initialization."""

    def test_mail_service_creates_database(self):
        """Mail service should create database file on init."""
        from src.agents.departments.department_mail import DepartmentMailService

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_mail.db")
            service = DepartmentMailService(db_path=db_path)

            assert os.path.exists(db_path)
            service.close()

    def test_mail_service_creates_messages_table(self):
        """Mail service should create messages table with correct schema."""
        from src.agents.departments.department_mail import DepartmentMailService

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_mail.db")
            service = DepartmentMailService(db_path=db_path)

            # Check table exists
            cursor = service.db.cursor()
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='messages'
            """)
            result = cursor.fetchone()
            assert result is not None
            assert result[0] == "messages"

            # Check columns
            cursor.execute("PRAGMA table_info(messages)")
            columns = {row[1] for row in cursor.fetchall()}
            expected_columns = {
                'id', 'from_dept', 'to_dept', 'type', 'subject',
                'body', 'priority', 'timestamp', 'read'
            }
            assert columns == expected_columns

            service.close()

    def test_mail_service_enables_wal_mode(self):
        """Mail service should enable WAL mode for concurrent access."""
        from src.agents.departments.department_mail import DepartmentMailService

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_mail.db")
            service = DepartmentMailService(db_path=db_path)

            cursor = service.db.cursor()
            cursor.execute("PRAGMA journal_mode")
            result = cursor.fetchone()
            assert result[0].lower() == "wal"

            service.close()
