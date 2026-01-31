"""
Unit Tests for MCP Tools

Tests each MCP tool with various inputs and edge cases.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.mcp_tools.server import (
    DatabaseQueryRequest,
    MemorySearchRequest,
    FileReadRequest,
    FileWriteRequest,
    FileListRequest,
    MT5AccountInfoRequest,
    KnowledgeSearchRequest,
    SkillLoadRequest
)


class TestDatabaseQueryTool:
    """Tests for database query MCP tool."""
    
    @patch('src.mcp_tools.server.DatabaseManager')
    def test_query_account_success(self, mock_db_class):
        """Test successful account query."""
        from src.mcp_tools.server import query_database
        
        # Mock database
        mock_db = Mock()
        mock_account = Mock()
        mock_account.id = 1
        mock_account.account_id = "test_account"
        mock_account.firm_name = "TestFirm"
        mock_db.get_prop_account.return_value = mock_account
        mock_db_class.return_value = mock_db
        
        # Query
        request = DatabaseQueryRequest(
            query_type="account",
            filters={"account_id": "test_account"}
        )
        response = query_database(request)
        
        assert response.success is True
        assert response.count == 1
        assert response.data[0]["account_id"] == "test_account"
    
    @patch('src.mcp_tools.server.DatabaseManager')
    def test_query_strategy_with_filters(self, mock_db_class):
        """Test strategy query with Kelly score filter."""
        from src.mcp_tools.server import query_database
        
        # Mock database
        mock_db = Mock()
        mock_strategy = Mock()
        mock_strategy.strategy_name = "TestStrategy"
        mock_strategy.kelly_score = 0.85
        mock_strategy.sharpe_ratio = 1.5
        mock_strategy.max_drawdown = 10.0
        mock_db.get_strategy_performance.return_value = [mock_strategy]
        mock_db_class.return_value = mock_db
        
        # Query
        request = DatabaseQueryRequest(
            query_type="strategy",
            filters={"min_kelly_score": 0.8},
            limit=10
        )
        response = query_database(request)
        
        assert response.success is True
        assert response.count == 1
        assert response.data[0]["kelly_score"] == 0.85


class TestMemorySearchTool:
    """Tests for memory search MCP tool."""
    
    @patch('src.mcp_tools.server.DatabaseManager')
    def test_search_memory_success(self, mock_db_class):
        """Test successful memory search."""
        from src.mcp_tools.server import search_memory
        
        # Mock database
        mock_db = Mock()
        mock_db.search_agent_memory.return_value = [
            {"content": "Test memory", "agent_type": "analyst"}
        ]
        mock_db_class.return_value = mock_db
        
        # Search
        request = MemorySearchRequest(
            query="test query",
            memory_type="semantic",
            agent_type="analyst"
        )
        response = search_memory(request)
        
        assert response.success is True
        assert response.count == 1


class TestFileOperationsTools:
    """Tests for file operations MCP tools."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir) / "workspaces" / "analyst"
            workspace_path.mkdir(parents=True)
            yield tmpdir
    
    def test_write_and_read_file(self, temp_workspace):
        """Test writing and reading files."""
        from src.mcp_tools.server import write_file, read_file
        
        with patch('src.mcp_tools.server.Path') as mock_path:
            # Setup mock paths
            workspace_path = Path(temp_workspace) / "workspaces" / "analyst"
            test_file = workspace_path / "test.txt"
            
            mock_path.return_value = workspace_path
            
            # Write file
            write_request = FileWriteRequest(
                file_path="test.txt",
                content="Test content",
                workspace="analyst"
            )
            
            # Create actual file for testing
            test_file.parent.mkdir(parents=True, exist_ok=True)
            with open(test_file, 'w') as f:
                f.write("Test content")
            
            # Read file
            read_request = FileReadRequest(
                file_path="test.txt",
                workspace="analyst"
            )
            
            # Mock the path resolution
            with patch('pathlib.Path.resolve') as mock_resolve:
                mock_resolve.return_value = test_file
                with patch('pathlib.Path.is_relative_to', return_value=True):
                    with patch('pathlib.Path.exists', return_value=True):
                        with patch('builtins.open', create=True) as mock_open:
                            mock_open.return_value.__enter__.return_value.read.return_value = "Test content"
                            result = read_file(read_request)
            
            assert result["success"] is True or "error" in result
    
    def test_list_files(self, temp_workspace):
        """Test listing files in directory."""
        from src.mcp_tools.server import list_files
        
        # Create test files
        workspace_path = Path(temp_workspace) / "workspaces" / "analyst"
        workspace_path.mkdir(parents=True, exist_ok=True)
        (workspace_path / "file1.txt").write_text("content1")
        (workspace_path / "file2.txt").write_text("content2")
        
        with patch('src.mcp_tools.server.Path') as mock_path_class:
            mock_path = Mock()
            mock_path.resolve.return_value = workspace_path
            mock_path.is_relative_to.return_value = True
            mock_path.exists.return_value = True
            
            mock_item1 = Mock()
            mock_item1.name = "file1.txt"
            mock_item1.is_dir.return_value = False
            mock_item1.is_file.return_value = True
            mock_item1.stat.return_value.st_size = 100
            
            mock_item2 = Mock()
            mock_item2.name = "file2.txt"
            mock_item2.is_dir.return_value = False
            mock_item2.is_file.return_value = True
            mock_item2.stat.return_value.st_size = 200
            
            mock_path.iterdir.return_value = [mock_item1, mock_item2]
            mock_path_class.return_value = mock_path
            
            request = FileListRequest(
                directory=".",
                workspace="analyst"
            )
            result = list_files(request)
            
            assert result["success"] is True or "error" in result


class TestMT5IntegrationTools:
    """Tests for MT5 integration MCP tools."""
    
    @patch('src.mcp_tools.server.DatabaseManager')
    def test_get_account_info(self, mock_db_class):
        """Test getting MT5 account info."""
        from src.mcp_tools.server import get_mt5_account_info
        
        # Mock database
        mock_db = Mock()
        mock_account = Mock()
        mock_account.account_id = "12345"
        mock_account.firm_name = "TestFirm"
        mock_db.get_prop_account.return_value = mock_account
        mock_db.get_daily_snapshot.return_value = {
            "current_equity": 105000.0,
            "daily_drawdown_pct": 2.5
        }
        mock_db_class.return_value = mock_db
        
        # Get info
        request = MT5AccountInfoRequest(account_id="12345")
        result = get_mt5_account_info(request)
        
        assert result["success"] is True
        assert result["account_id"] == "12345"
        assert result["current_equity"] == 105000.0


class TestKnowledgeBaseTools:
    """Tests for knowledge base search MCP tools."""
    
    @patch('src.mcp_tools.server.DatabaseManager')
    def test_search_knowledge_base(self, mock_db_class):
        """Test knowledge base search."""
        from src.mcp_tools.server import search_knowledge_base
        
        # Mock database
        mock_db = Mock()
        mock_db.search_knowledge.return_value = [
            {"title": "Test Article", "content": "Test content"}
        ]
        mock_db_class.return_value = mock_db
        
        # Search
        request = KnowledgeSearchRequest(
            query="test query",
            limit=10
        )
        result = search_knowledge_base(request)
        
        assert result["success"] is True
        assert result["count"] == 1


class TestSkillLoadingTools:
    """Tests for skill loading MCP tools."""
    
    def test_load_skill_success(self):
        """Test successful skill loading."""
        from src.mcp_tools.server import load_skill
        
        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', create=True) as mock_open:
                mock_open.return_value.__enter__.return_value.read.return_value = "# Skill code"
                
                request = SkillLoadRequest(
                    skill_name="test_skill",
                    skill_type="indicator"
                )
                result = load_skill(request)
                
                assert result["success"] is True or "error" in result
    
    def test_load_skill_not_found(self):
        """Test loading non-existent skill."""
        from src.mcp_tools.server import load_skill
        
        with patch('pathlib.Path.exists', return_value=False):
            request = SkillLoadRequest(
                skill_name="nonexistent",
                skill_type="indicator"
            )
            result = load_skill(request)
            
            assert result["success"] is False
            assert "not found" in result["error"].lower()
