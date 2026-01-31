"""
FastMCP Server for QuantMindX Backend

Provides MCP tools for agent access to database, memory, files, and MT5 integration.
"""

import logging
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Initialize FastMCP server
try:
    from mcp import FastMCP
    mcp = FastMCP("QuantMindX Backend")
    logger.info("FastMCP server initialized")
except ImportError:
    logger.warning("FastMCP not available - MCP tools will not be registered")
    mcp = None


# ============================================================================
# Database Query Tools
# ============================================================================

class DatabaseQueryRequest(BaseModel):
    """Request schema for database queries."""
    query_type: str = Field(description="Type of query: account, snapshot, strategy, task")
    filters: Dict[str, Any] = Field(default={}, description="Query filters")
    limit: int = Field(default=100, description="Maximum results to return", ge=1, le=1000)


class DatabaseQueryResponse(BaseModel):
    """Response schema for database queries."""
    success: bool
    data: List[Dict[str, Any]]
    count: int
    message: Optional[str] = None


if mcp:
    @mcp.tool()
    def query_database(request: DatabaseQueryRequest) -> DatabaseQueryResponse:
        """
        Query database with type-safe validation.
        
        **Validates: Property 15: MCP Tool Schema Validation**
        
        Args:
            request: Database query request with filters
            
        Returns:
            Query results with metadata
        """
        try:
            from src.database.manager import DatabaseManager
            
            db = DatabaseManager()
            results = []
            
            if request.query_type == "account":
                # Query prop firm accounts
                account_id = request.filters.get("account_id")
                if account_id:
                    account = db.get_prop_account(account_id)
                    if account:
                        results = [{
                            "id": account.id,
                            "account_id": account.account_id,
                            "firm_name": account.firm_name
                        }]
            
            elif request.query_type == "snapshot":
                # Query daily snapshots
                account_id = request.filters.get("account_id")
                if account_id:
                    snapshot = db.get_daily_snapshot(account_id)
                    if snapshot:
                        results = [snapshot]
            
            elif request.query_type == "strategy":
                # Query strategy performance
                min_kelly = request.filters.get("min_kelly_score")
                strategies = db.get_strategy_performance(
                    min_kelly_score=min_kelly,
                    limit=request.limit
                )
                results = [
                    {
                        "strategy_name": s.strategy_name,
                        "kelly_score": s.kelly_score,
                        "sharpe_ratio": s.sharpe_ratio,
                        "max_drawdown": s.max_drawdown
                    }
                    for s in strategies
                ]
            
            elif request.query_type == "task":
                # Query agent tasks
                agent_type = request.filters.get("agent_type")
                status = request.filters.get("status")
                tasks = db.get_agent_tasks(
                    agent_type=agent_type,
                    status=status,
                    limit=request.limit
                )
                results = [
                    {
                        "id": t.id,
                        "agent_type": t.agent_type,
                        "task_type": t.task_type,
                        "status": t.status
                    }
                    for t in tasks
                ]
            
            return DatabaseQueryResponse(
                success=True,
                data=results,
                count=len(results)
            )
            
        except Exception as e:
            logger.error(f"Database query failed: {e}")
            return DatabaseQueryResponse(
                success=False,
                data=[],
                count=0,
                message=f"Error: {str(e)}"
            )


# ============================================================================
# Memory Search Tools
# ============================================================================

class MemorySearchRequest(BaseModel):
    """Request schema for memory search."""
    query: str = Field(description="Search query text")
    memory_type: str = Field(description="Type: semantic, episodic, procedural")
    agent_type: Optional[str] = Field(default=None, description="Filter by agent type")
    limit: int = Field(default=10, description="Maximum results", ge=1, le=100)


class MemorySearchResponse(BaseModel):
    """Response schema for memory search."""
    success: bool
    results: List[Dict[str, Any]]
    count: int
    message: Optional[str] = None


if mcp:
    @mcp.tool()
    def search_memory(request: MemorySearchRequest) -> MemorySearchResponse:
        """
        Search agent memory using semantic similarity.
        
        Args:
            request: Memory search request
            
        Returns:
            Matching memory entries
        """
        try:
            from src.database.manager import DatabaseManager
            
            db = DatabaseManager()
            
            # Search agent memory in ChromaDB
            results = db.search_agent_memory(
                query=request.query,
                agent_type=request.agent_type,
                memory_type=request.memory_type,
                limit=request.limit
            )
            
            return MemorySearchResponse(
                success=True,
                results=results,
                count=len(results)
            )
            
        except Exception as e:
            logger.error(f"Memory search failed: {e}")
            return MemorySearchResponse(
                success=False,
                results=[],
                count=0,
                message=f"Error: {str(e)}"
            )


# ============================================================================
# File Operations Tools
# ============================================================================

class FileReadRequest(BaseModel):
    """Request schema for file read."""
    file_path: str = Field(description="Path to file relative to workspace")
    workspace: str = Field(default="analyst", description="Workspace: analyst, quant, copilot")


class FileWriteRequest(BaseModel):
    """Request schema for file write."""
    file_path: str = Field(description="Path to file relative to workspace")
    content: str = Field(description="File content to write")
    workspace: str = Field(default="analyst", description="Workspace: analyst, quant, copilot")


class FileListRequest(BaseModel):
    """Request schema for file listing."""
    directory: str = Field(default=".", description="Directory path relative to workspace")
    workspace: str = Field(default="analyst", description="Workspace: analyst, quant, copilot")


if mcp:
    @mcp.tool()
    def read_file(request: FileReadRequest) -> Dict[str, Any]:
        """
        Read file from workspace.
        
        Args:
            request: File read request
            
        Returns:
            File content and metadata
        """
        try:
            from pathlib import Path
            
            workspace_path = Path(f"workspaces/{request.workspace}")
            file_path = workspace_path / request.file_path
            
            # Security: Ensure path is within workspace
            if not file_path.resolve().is_relative_to(workspace_path.resolve()):
                raise ValueError("Path traversal detected")
            
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {request.file_path}")
            
            with open(file_path, 'r') as f:
                content = f.read()
            
            return {
                "success": True,
                "content": content,
                "path": str(file_path),
                "size": len(content)
            }
            
        except Exception as e:
            logger.error(f"File read failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    @mcp.tool()
    def write_file(request: FileWriteRequest) -> Dict[str, Any]:
        """
        Write file to workspace.
        
        Args:
            request: File write request
            
        Returns:
            Write status and metadata
        """
        try:
            from pathlib import Path
            
            workspace_path = Path(f"workspaces/{request.workspace}")
            file_path = workspace_path / request.file_path
            
            # Security: Ensure path is within workspace
            if not file_path.resolve().is_relative_to(workspace_path.resolve()):
                raise ValueError("Path traversal detected")
            
            # Create parent directories
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write file
            with open(file_path, 'w') as f:
                f.write(request.content)
            
            return {
                "success": True,
                "path": str(file_path),
                "size": len(request.content)
            }
            
        except Exception as e:
            logger.error(f"File write failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    @mcp.tool()
    def list_files(request: FileListRequest) -> Dict[str, Any]:
        """
        List files in workspace directory.
        
        Args:
            request: File list request
            
        Returns:
            List of files and directories
        """
        try:
            from pathlib import Path
            
            workspace_path = Path(f"workspaces/{request.workspace}")
            dir_path = workspace_path / request.directory
            
            # Security: Ensure path is within workspace
            if not dir_path.resolve().is_relative_to(workspace_path.resolve()):
                raise ValueError("Path traversal detected")
            
            if not dir_path.exists():
                raise FileNotFoundError(f"Directory not found: {request.directory}")
            
            files = []
            for item in dir_path.iterdir():
                files.append({
                    "name": item.name,
                    "type": "directory" if item.is_dir() else "file",
                    "size": item.stat().st_size if item.is_file() else None
                })
            
            return {
                "success": True,
                "files": files,
                "count": len(files),
                "path": str(dir_path)
            }
            
        except Exception as e:
            logger.error(f"File list failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }


# ============================================================================
# MT5 Integration Tools
# ============================================================================

class MT5AccountInfoRequest(BaseModel):
    """Request schema for MT5 account info."""
    account_id: str = Field(description="MT5 account number")


if mcp:
    @mcp.tool()
    def get_mt5_account_info(request: MT5AccountInfoRequest) -> Dict[str, Any]:
        """
        Get MT5 account information.
        
        Args:
            request: Account info request
            
        Returns:
            Account information
        """
        try:
            from src.database.manager import DatabaseManager
            
            db = DatabaseManager()
            account = db.get_prop_account(request.account_id)
            
            if not account:
                return {
                    "success": False,
                    "error": f"Account {request.account_id} not found"
                }
            
            snapshot = db.get_daily_snapshot(request.account_id)
            
            return {
                "success": True,
                "account_id": account.account_id,
                "firm_name": account.firm_name,
                "current_equity": snapshot.get('current_equity') if snapshot else None,
                "daily_drawdown": snapshot.get('daily_drawdown_pct') if snapshot else None
            }
            
        except Exception as e:
            logger.error(f"MT5 account info failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }


# ============================================================================
# Knowledge Base Search Tools
# ============================================================================

class KnowledgeSearchRequest(BaseModel):
    """Request schema for knowledge base search."""
    query: str = Field(description="Search query text")
    limit: int = Field(default=10, description="Maximum results", ge=1, le=50)


if mcp:
    @mcp.tool()
    def search_knowledge_base(request: KnowledgeSearchRequest) -> Dict[str, Any]:
        """
        Search knowledge base using ChromaDB.
        
        Args:
            request: Knowledge search request
            
        Returns:
            Matching knowledge articles
        """
        try:
            from src.database.manager import DatabaseManager
            
            db = DatabaseManager()
            results = db.search_knowledge(request.query, limit=request.limit)
            
            return {
                "success": True,
                "results": results,
                "count": len(results)
            }
            
        except Exception as e:
            logger.error(f"Knowledge search failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }


# ============================================================================
# Skill Loading Tools
# ============================================================================

class SkillLoadRequest(BaseModel):
    """Request schema for skill loading."""
    skill_name: str = Field(description="Name of skill to load")
    skill_type: str = Field(description="Type: indicator, strategy, utility")


if mcp:
    @mcp.tool()
    def load_skill(request: SkillLoadRequest) -> Dict[str, Any]:
        """
        Load skill with dynamic registration.
        
        Args:
            request: Skill load request
            
        Returns:
            Skill metadata and status
        """
        try:
            from pathlib import Path
            
            # Search for skill in data/assets/skills
            skills_path = Path("data/assets/skills")
            skill_file = skills_path / request.skill_type / f"{request.skill_name}.py"
            
            if not skill_file.exists():
                return {
                    "success": False,
                    "error": f"Skill not found: {request.skill_name}"
                }
            
            # Read skill content
            with open(skill_file, 'r') as f:
                skill_code = f.read()
            
            return {
                "success": True,
                "skill_name": request.skill_name,
                "skill_type": request.skill_type,
                "code": skill_code,
                "path": str(skill_file)
            }
            
        except Exception as e:
            logger.error(f"Skill loading failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }


# Export server instance
__all__ = ['mcp']
