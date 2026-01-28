#!/usr/bin/env python3
"""
ChromaDB MCP Server for QuantMindX Knowledge Base
Provides semantic search over MQL5 articles via MCP protocol.
Only works when run from the QuantMindX directory.

Enhanced with Task Group 3 performance optimizations:
- Query result caching with 60-second TTL
- HNSW index optimization (M=16, ef_construction=100)
- Connection pooling with health checks
- Comprehensive error handling and logging
"""

import json
import asyncio
import functools
import logging
import os
import sys
import re
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("quantmindx-kb-chroma")

# Pydantic models for structured retrieval tools
try:
    from pydantic import BaseModel, Field
except ImportError:
    print("Pydantic not found. Install with: pip install pydantic")
    exit(1)

# Security check: Only run from QuantMindX directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if PROJECT_ROOT.name != "QUANTMINDX":
    # Also check if we're already in QUANTMINDX directory
    cwd = Path.cwd()
    if cwd.name == "QUANTMINDX":
        PROJECT_ROOT = cwd
    else:
        print(f"Error: Must be run from QuantMindX directory", file=sys.stderr)
        print(f"Script location: {PROJECT_ROOT}", file=sys.stderr)
        print(f"Current directory: {cwd}", file=sys.stderr)
        sys.exit(1)

os.chdir(PROJECT_ROOT)

try:
    from mcp.server import Server  # type: ignore
    from mcp.types import Tool, TextContent  # type: ignore
except ImportError:
    print("MCP SDK not found. Install with: pip install mcp")
    exit(1)

# Type aliases for clarity
MCPResult = List[TextContent]
ToolList = List[Tool]

try:
    import chromadb
except ImportError:
    print("ChromaDB not found. Install with: pip install chromadb")
    exit(1)

# Configuration
CHROMA_PATH = PROJECT_ROOT / "data" / "chromadb"
COLLECTION_NAME = "mql5_knowledge"
SCRAPED_DIR = PROJECT_ROOT / "data" / "scraped_articles"

# Git repository paths for raw file storage
GIT_REPO_PATH = PROJECT_ROOT / "data" / "git" / "assets-hub"
TEMPLATES_PATH = GIT_REPO_PATH / "templates"
SKILLS_PATH = GIT_REPO_PATH / "skills"

# =============================================================================
# Task Group 3: Connection Pooling and Health Check
# =============================================================================

class ChromaDBConnectionManager:
    """Manages ChromaDB client connection with health checks."""

    def __init__(self, path: Path):
        self._path = path
        self._client: Optional[chromadb.PersistentClient] = None
        self._last_health_check: float = 0
        self._health_check_interval: float = 30  # seconds
        self._lock = asyncio.Lock()

    @property
    def client(self) -> chromadb.PersistentClient:
        """Get or create ChromaDB client with health check."""
        if self._client is None:
            logger.info(f"Initializing ChromaDB client at {self._path}")
            self._client = chromadb.PersistentClient(path=str(self._path))

        return self._client

    async def health_check(self) -> bool:
        """Check if ChromaDB connection is healthy."""
        try:
            # Only check periodically to avoid overhead
            current_time = time.time()
            if current_time - self._last_health_check < self._health_check_interval:
                return True

            # Verify connection by listing collections
            collections = self.client.list_collections()
            self._last_health_check = current_time
            logger.debug(f"ChromaDB health check passed: {len(collections)} collections")
            return True
        except Exception as e:
            logger.error(f"ChromaDB health check failed: {e}")
            # Try to recover by recreating client
            self._client = None
            return False

    async def ensure_connection(self) -> chromadb.PersistentClient:
        """Ensure connection is healthy before returning client."""
        if not await self.health_check():
            logger.warning("ChromaDB connection unhealthy, attempting recovery...")
            self._client = None
            # Force new client creation
            _ = self.client
            if await self.health_check():
                logger.info("ChromaDB connection recovered")
            else:
                raise ConnectionError("Failed to recover ChromaDB connection")
        return self.client


# Initialize connection manager
connection_manager = ChromaDBConnectionManager(CHROMA_PATH)


# =============================================================================
# Task Group 3: Query Result Caching with TTL
# =============================================================================

class QueryCache:
    """In-memory cache for query results with TTL."""

    def __init__(self, ttl_seconds: int = 60):
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._ttl = ttl_seconds
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0

    def _make_key(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Create cache key from tool name and serialized arguments."""
        # Sort arguments for consistent keys
        sorted_args = json.dumps(arguments, sort_keys=True)
        return f"{tool_name}:{sorted_args}"

    async def get(self, tool_name: str, arguments: Dict[str, Any]) -> Optional[Any]:
        """Get cached result if available and not expired."""
        cache_key = self._make_key(tool_name, arguments)

        async with self._lock:
            if cache_key in self._cache:
                result, timestamp = self._cache[cache_key]
                if time.time() - timestamp < self._ttl:
                    self._hits += 1
                    logger.debug(f"Cache hit for {tool_name}: {cache_key[:50]}...")
                    return result
                else:
                    # Expired, remove from cache
                    del self._cache[cache_key]

            self._misses += 1
            return None

    async def set(self, tool_name: str, arguments: Dict[str, Any], result: Any) -> None:
        """Cache result with current timestamp."""
        cache_key = self._make_key(tool_name, arguments)

        async with self._lock:
            self._cache[cache_key] = (result, time.time())

            # Clean up old entries periodically
            if len(self._cache) > 1000:
                await self._cleanup_expired()

    async def _cleanup_expired(self) -> None:
        """Remove expired entries from cache."""
        current_time = time.time()
        expired_keys = [
            key for key, (_, timestamp) in self._cache.items()
            if current_time - timestamp >= self._ttl
        ]
        for key in expired_keys:
            del self._cache[key]

        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "size": len(self._cache),
            "hit_rate": hit_rate
        }


# Initialize query cache with 60-second TTL
query_cache = QueryCache(ttl_seconds=60)


# =============================================================================
# Task Group 3: Decorator for Cached Tool Calls with Logging
# =============================================================================

def cached_tool_call(tool_name: str):
    """Decorator to cache tool results and log execution time."""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.perf_counter()

            # Try to get from cache
            cached_result = await query_cache.get(tool_name, kwargs)
            if cached_result is not None:
                logger.info(f"Cached result for {tool_name}")
                return cached_result

            # Execute function
            try:
                result = await func(*args, **kwargs)
                execution_time = time.perf_counter() - start_time

                # Log execution
                logger.info(
                    f"Tool call: {tool_name} | "
                    f"Args: {list(kwargs.keys())[:3]} | "
                    f"Time: {execution_time*1000:.2f}ms"
                )

                # Cache result
                await query_cache.set(tool_name, kwargs, result)

                return result

            except Exception as e:
                execution_time = time.perf_counter() - start_time
                logger.error(
                    f"Tool call failed: {tool_name} | "
                    f"Error: {str(e)} | "
                    f"Time: {execution_time*1000:.2f}ms"
                )
                raise

        return wrapper
    return decorator


# =============================================================================
# HNSW Index Configuration (Task Group 3 optimized settings)
# =============================================================================

# Get ChromaDB client through connection manager
def get_client() -> chromadb.PersistentClient:
    """Synchronous wrapper to get client for collection initialization."""
    return connection_manager.client

# Initialize ChromaDB client
client = get_client()

# HNSW index configuration for all collections (optimized for performance)
HNSW_CONFIG = {
    "hnsw:space": "cosine",
    "hnsw:M": 16,  # Connections per node (optimized for recall/speed balance)
    "hnsw:construction_ef": 100,  # Build-time accuracy
    "hnsw:search_ef": 50  # Search-time accuracy (tunable for latency)
}

# Initialize all required collections
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata=HNSW_CONFIG
)

# Extended schema: Additional collections for Assets Hub
algorithm_templates_collection = client.get_or_create_collection(
    name="algorithm_templates",
    metadata=HNSW_CONFIG
)

agentic_skills_collection = client.get_or_create_collection(
    name="agentic_skills",
    metadata=HNSW_CONFIG
)

coding_standards_collection = client.get_or_create_collection(
    name="coding_standards",
    metadata=HNSW_CONFIG
)

bad_patterns_collection = client.get_or_create_collection(
    name="bad_patterns_graveyard",
    metadata=HNSW_CONFIG
)

def extract_title_from_frontmatter(doc: str) -> str:
    """Extract title from YAML frontmatter or markdown heading."""
    # Try YAML frontmatter first
    if doc.startswith('---'):
        lines = doc.split('\n')
        for line in lines[1:10]:  # Check first 10 lines
            if line.startswith('title:'):
                return line.split(':', 1)[1].strip()
            if line == '---':
                break

    # Try first # heading
    match = re.search(r'^#\s+(.+)$', doc, re.MULTILINE)
    if match:
        return match.group(1).strip()

    return "Untitled"


# =============================================================================
# Pydantic Models for Structured Asset Retrieval Tools
# =============================================================================

class AlgorithmTemplate(BaseModel):
    """Schema for algorithm template retrieval."""
    category: Literal["trend_following", "mean_reversion", "breakout", "momentum", "arbitrage"]
    language: Literal["mq5", "python"] = "mq5"
    complexity: Literal["basic", "intermediate", "advanced"] = "intermediate"
    include_indicators: bool = True


class TemplateResult(BaseModel):
    """Schema for template result."""
    name: str
    category: str
    description: str
    code: str
    dependencies: List[str]
    example_usage: str


class CodingStandardsResult(BaseModel):
    """Schema for coding standards result."""
    project_name: str = "QuantMindX"
    version: str = "1.0"
    conventions: Dict[str, str]
    file_structure: List[str]
    documentation_required: bool = True
    testing_required: bool = True


class BadPatternResult(BaseModel):
    """Schema for bad patterns result."""
    patterns: List[Dict[str, str]]
    total_count: int
    last_updated: datetime


class SkillDefinition(BaseModel):
    """Schema for skill definition."""
    name: str
    category: Literal["trading_skills", "system_skills", "data_skills"]
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    code: str
    dependencies: List[str]
    example_usage: str
    version: str


# MCP Server
server = Server("quantmindx-kb-chroma")

@server.list_tools()
async def list_tools() -> ToolList:
    """List available tools."""
    return [
        Tool(
            name="search_knowledge_base",
            description="Search the MQL5 knowledge base for articles about trading strategies, indicators, Expert Advisors, and MQL5 programming.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results (default 5)",
                        "default": 5
                    },
                    "category_filter": {
                        "type": "string",
                        "description": "Optional category filter (e.g., 'Trading Systems', 'Integration')"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_article_content",
            description="Get the full content of a specific article by file path.",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Relative file path from search results"
                    }
                },
                "required": ["file_path"]
            }
        ),
        Tool(
            name="list_skills",
            description="List available agentic skills.",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="get_skill",
            description="Get the definition of a specific skill.",
            inputSchema={
                "type": "object",
                "properties": {
                    "skill_name": {"type": "string"}
                },
                "required": ["skill_name"]
            }
        ),
        Tool(
            name="list_templates",
            description="List available code templates.",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="get_template",
            description="Get the content of a specific code template.",
            inputSchema={
                "type": "object",
                "properties": {
                    "template_name": {"type": "string"}
                },
                "required": ["template_name"]
            }
        ),
        # -------------------------------------------------------------------------
        # NEW: Structured Asset Retrieval Tools (Task Group 2)
        # -------------------------------------------------------------------------
        Tool(
            name="get_algorithm_template",
            description="Retrieve algorithm templates by category, language, and complexity. Returns structured template with code, dependencies, and usage examples.",
            inputSchema={
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["trend_following", "mean_reversion", "breakout", "momentum", "arbitrage"],
                        "description": "Algorithm category"
                    },
                    "language": {
                        "type": "string",
                        "enum": ["mq5", "python"],
                        "default": "mq5",
                        "description": "Programming language"
                    },
                    "complexity": {
                        "type": "string",
                        "enum": ["basic", "intermediate", "advanced"],
                        "default": "intermediate",
                        "description": "Complexity level"
                    },
                    "include_indicators": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include indicator dependencies"
                    }
                },
                "required": ["category"]
            }
        ),
        Tool(
            name="get_coding_standards",
            description="Retrieve project coding standards including naming conventions, file structure, and requirements. Defaults to QuantMindX v1.0 standards.",
            inputSchema={
                "type": "object",
                "properties": {
                    "project_name": {
                        "type": "string",
                        "default": "QuantMindX",
                        "description": "Project name for standards lookup"
                    }
                }
            }
        ),
        Tool(
            name="get_bad_patterns",
            description="Retrieve known anti-patterns from the bad patterns graveyard. Returns patterns sorted by severity with solutions.",
            inputSchema={
                "type": "object",
                "properties": {
                    "severity": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                        "description": "Filter by severity level (optional)"
                    },
                    "limit": {
                        "type": "integer",
                        "default": 20,
                        "description": "Maximum number of patterns to return"
                    }
                }
            }
        ),
        Tool(
            name="load_skill",
            description="Load an agentic skill definition by name. Returns skill schema, code, dependencies, and usage examples. Validates conformance to AgentSkill interface.",
            inputSchema={
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "Name of the skill to load"
                    },
                    "resolve_dependencies": {
                        "type": "boolean",
                        "default": True,
                        "description": "Recursively resolve skill dependencies"
                    }
                },
                "required": ["skill_name"]
            }
        ),
        # -------------------------------------------------------------------------
        Tool(
            name="kb_stats",
            description="Get statistics about the knowledge base.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="list_categories",
            description="List all available categories in the knowledge base.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> MCPResult:
    """Handle tool calls with caching, logging, and error handling."""

    # Try cache first for read operations
    if name in ("search_knowledge_base", "get_article_content", "kb_stats",
                "list_categories", "list_templates", "get_template",
                "list_skills", "get_skill"):
        cached_result = await query_cache.get(name, arguments)
        if cached_result is not None:
            return cached_result

    start_time = time.perf_counter()

    try:
        result = await _execute_tool(name, arguments)
        execution_time = time.perf_counter() - start_time

        # Log successful execution
        logger.info(
            f"Tool: {name} | "
            f"Args: {json.dumps(arguments, sort_keys=True)[:100]} | "
            f"Time: {execution_time*1000:.2f}ms"
        )

        # Cache read operations
        if name in ("search_knowledge_base", "get_article_content", "kb_stats",
                    "list_categories", "list_templates", "get_template",
                    "list_skills", "get_skill"):
            await query_cache.set(name, arguments, result)

        return result

    except Exception as e:
        execution_time = time.perf_counter() - start_time
        logger.error(
            f"Tool failed: {name} | "
            f"Error: {str(e)} | "
            f"Traceback: {traceback.format_exc()} | "
            f"Time: {execution_time*1000:.2f}ms"
        )

        # Return user-friendly error response
        error_response = {
            "error": True,
            "tool": name,
            "message": str(e),
            "suggestion": _get_error_suggestion(name, e)
        }
        return [TextContent(type="text", text=json.dumps(error_response, indent=2))]


def _get_error_suggestion(tool_name: str, error: Exception) -> str:
    """Provide helpful suggestions for common errors."""
    error_str = str(error).lower()

    if "connection" in error_str or "chroma" in error_str:
        return "Check if ChromaDB is running and accessible. Try restarting the MCP server."
    elif "collection" in error_str and "not found" in error_str:
        return "The requested collection may not exist. Try running the migration script."
    elif "git" in error_str:
        return "Git operation failed. Check file permissions and repository status."
    elif "validation" in error_str or "invalid" in error_str:
        return f"Check input parameters for {tool_name}. Refer to tool schema."
    else:
        return "Contact support with the error details if issue persists."


async def _execute_tool(name: str, arguments: Dict[str, Any]) -> MCPResult:
    """Execute the actual tool logic."""

    if name == "search_knowledge_base":
        query = arguments.get("query", "")
        limit = arguments.get("limit", 5)
        category_filter = arguments.get("category_filter")

        # Use connection manager for client access
        db_client = await connection_manager.ensure_connection()
        search_collection = db_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata=HNSW_CONFIG
        )

        # Search ChromaDB (built-in embeddings)
        results = search_collection.query(
            query_texts=[query],
            n_results=limit * 2  # Get more, filter later
        )

        # Process results
        filtered = []
        seen_titles = set()
        distances = results.get('distances', [[]])
        metadatas = results.get('metadatas', [[]])

        if results['documents'] and len(results['documents'][0]) > 0:
            for i, doc in enumerate(results['documents'][0]):
                metadata = metadatas[0][i] if metadatas and len(metadatas) > 0 and i < len(metadatas[0]) else {}
                # Extract title from frontmatter since metadata title is "---"
                title = extract_title_from_frontmatter(doc)
                distance = distances[0][i] if distances and len(distances[0]) > i else 0

                # Dedupe by title
                if title in seen_titles:
                    continue

                # Category filter
                if category_filter:
                    categories = str(metadata.get('categories', '') or '')
                    if category_filter.lower() not in categories.lower():
                        continue

                seen_titles.add(title)

                # Convert distance to relevance score (cosine distance -> similarity)
                relevance_score = 1 - distance if distance else 0

                # Get preview (skip frontmatter)
                preview_lines = doc.split('\n')
                preview_start = 0
                for j, line in enumerate(preview_lines):
                    if j > 0 and line == '---':
                        preview_start = j + 1
                        break
                preview = '\n'.join(preview_lines[preview_start:preview_start + 10]).strip()
                if len(preview) > 300:
                    preview = preview[:300] + "..."

                filtered.append({
                    "title": title,
                    "file_path": metadata.get('file_path', ''),
                    "categories": metadata.get('categories', ''),
                    "relevance_score": round(relevance_score, 3),
                    "preview": preview
                })

                if len(filtered) >= limit:
                    break

        return [TextContent(
            type="text",
            text=json.dumps({"results": filtered, "query": query}, indent=2)
        )]

    elif name == "get_article_content":
        file_path = arguments.get("file_path", "")
        full_path = SCRAPED_DIR / file_path

        if full_path.exists():
            content = full_path.read_text(encoding='utf-8')
            return [TextContent(type="text", text=content)]
        else:
            return [TextContent(type="text", text=f"Article not found: {file_path}")]

    elif name == "kb_stats":
        db_client = await connection_manager.ensure_connection()
        stats_collection = db_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata=HNSW_CONFIG
        )
        count = stats_collection.count()

        # Get cache stats
        cache_stats = query_cache.get_stats()

        stats = {
            "collection": COLLECTION_NAME,
            "total_articles": count,
            "storage_path": str(CHROMA_PATH),
            "embedding_function": "ChromaDB built-in (all-MiniLM-L6-v2)",
            "cache_stats": cache_stats
        }

        return [TextContent(type="text", text=json.dumps(stats, indent=2))]

    elif name == "list_categories":
        db_client = await connection_manager.ensure_connection()
        categories_collection = db_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata=HNSW_CONFIG
        )

        # Get unique categories from all metadata
        results = categories_collection.get(include=["metadatas"])
        categories = set()

        if results['metadatas']:
            for meta in results['metadatas']:
                cat = meta.get('categories', '')
                if cat:
                    categories.add(cat)

        return [TextContent(type="text", text=json.dumps({
            "categories": sorted(list(categories))
        }, indent=2))]

    elif name == "list_skills":
        skills_dir = PROJECT_ROOT / "data" / "assets" / "skills"
        skills = []
        if skills_dir.exists():
            for f in skills_dir.glob("*.yaml"):
                skills.append(f.name)
        return [TextContent(type="text", text=json.dumps({"skills": skills}, indent=2))]

    elif name == "get_skill":
        skill_name = arguments.get("skill_name")
        skill_path = PROJECT_ROOT / "data" / "assets" / "skills" / skill_name
        # Security check to prevent directory traversal
        if ".." in skill_name or not skill_path.is_relative_to(PROJECT_ROOT):
             return [TextContent(type="text", text="Error: Invalid skill path.")]

        if skill_path.exists():
            return [TextContent(type="text", text=skill_path.read_text())]
        else:
             return [TextContent(type="text", text=f"Skill not found: {skill_name}")]

    elif name == "list_templates":
        templates_dir = PROJECT_ROOT / "data" / "assets" / "templates"
        templates = []
        if templates_dir.exists():
            for f in templates_dir.glob("*"):
                templates.append(f.name)
        return [TextContent(type="text", text=json.dumps({"templates": templates}, indent=2))]

    elif name == "get_template":
        template_name = arguments.get("template_name")
        template_path = PROJECT_ROOT / "data" / "assets" / "templates" / template_name
        # Security check
        if ".." in template_name or not template_path.is_relative_to(PROJECT_ROOT):
             return [TextContent(type="text", text="Error: Invalid template path.")]

        if template_path.exists():
            return [TextContent(type="text", text=template_path.read_text())]
        else:
             return [TextContent(type="text", text=f"Template not found: {template_name}")]

    # -------------------------------------------------------------------------
    # NEW: Structured Asset Retrieval Tools Handlers (Task Group 2)
    # -------------------------------------------------------------------------

    elif name == "get_algorithm_template":
        category = arguments.get("category")
        language = arguments.get("language", "mq5")
        complexity = arguments.get("complexity", "intermediate")
        include_indicators = arguments.get("include_indicators", True)

        # Build metadata filters
        where_filter = {"category": category}
        if language:
            where_filter["language"] = language
        if complexity:
            where_filter["complexity"] = complexity

        # Query algorithm_templates collection with connection manager
        db_client = await connection_manager.ensure_connection()
        algo_collection = db_client.get_or_create_collection(
            name="algorithm_templates",
            metadata=HNSW_CONFIG
        )

        results = algo_collection.query(
            query_texts=[f"{category} {complexity} algorithm"],
            n_results=5,
            where=where_filter
        )

        if results['documents'] and len(results['documents'][0]) > 0:
            templates = []
            for i, doc in enumerate(results['documents'][0]):
                metadata = results.get('metadatas', [[]])[0][i] if results.get('metadatas') else {}

                # Parse dependencies from JSON string if present
                deps_str = metadata.get('dependencies', '[]')
                try:
                    dependencies = json.loads(deps_str) if isinstance(deps_str, str) else deps_str
                except json.JSONDecodeError:
                    dependencies = []

                template_result = TemplateResult(
                    name=metadata.get('name', 'Unnamed'),
                    category=metadata.get('category', category),
                    description=doc.split('\n')[0] if doc else '',
                    code=doc,
                    dependencies=dependencies,
                    example_usage=metadata.get('example_usage', '')
                )
                templates.append(template_result.model_dump())

            return [TextContent(
                type="text",
                text=json.dumps({
                    "templates": templates,
                    "category": category,
                    "language": language,
                    "complexity": complexity
                }, indent=2)
            )]
        else:
            return [TextContent(
                type="text",
                text=json.dumps({
                    "error": f"No algorithm templates found for category '{category}', language '{language}', complexity '{complexity}'",
                    "suggestion": "Try different parameters or check if templates are indexed."
                }, indent=2)
            )]

    elif name == "get_coding_standards":
        project_name = arguments.get("project_name", "QuantMindX")

        # Query coding_standards collection with connection manager
        db_client = await connection_manager.ensure_connection()
        standards_collection = db_client.get_or_create_collection(
            name="coding_standards",
            metadata=HNSW_CONFIG
        )

        results = standards_collection.get(
            where={"project_name": project_name},
            include=["documents", "metadatas"]
        )

        if results['documents'] and len(results['documents'][0]) > 0:
            metadata = results['metadatas'][0][0]
            doc = results['documents'][0][0]

            # Parse file_structure from JSON string
            file_structure_str = metadata.get('file_structure', '[]')
            try:
                file_structure = json.loads(file_structure_str) if isinstance(file_structure_str, str) else file_structure_str
            except json.JSONDecodeError:
                file_structure = []

            standards_result = CodingStandardsResult(
                project_name=metadata.get('project_name', project_name),
                version=metadata.get('version', '1.0'),
                conventions={
                    "naming": metadata.get('naming_convention', 'snake_case'),
                    "indentation": metadata.get('indentation', '4 spaces'),
                    "documentation": metadata.get('documentation', 'Required'),
                    "testing": metadata.get('testing', 'Required')
                },
                file_structure=file_structure,
                documentation_required=metadata.get('documentation_required', True),
                testing_required=metadata.get('testing_required', True)
            )

            return [TextContent(
                type="text",
                text=json.dumps({
                    **standards_result.model_dump(),
                    "full_document": doc
                }, indent=2)
            )]
        else:
            # Return QuantMindX v1.0 defaults
            default_standards = CodingStandardsResult(
                project_name="QuantMindX",
                version="1.0",
                conventions={
                    "naming": "snake_case",
                    "indentation": "4 spaces",
                    "documentation": "Required",
                    "testing": "Required",
                    "file_headers": "Required",
                    "type_hints": "Required for Python"
                },
                file_structure=["includes/", "src/", "tests/", "docs/"],
                documentation_required=True,
                testing_required=True
            )

            return [TextContent(
                type="text",
                text=json.dumps({
                    **default_standards.model_dump(),
                    "note": f"No coding standards found for '{project_name}'. Using QuantMindX v1.0 defaults."
                }, indent=2)
            )]

    elif name == "get_bad_patterns":
        severity = arguments.get("severity")
        limit = arguments.get("limit", 20)

        # Query bad_patterns_graveyard collection with connection manager
        db_client = await connection_manager.ensure_connection()
        patterns_collection = db_client.get_or_create_collection(
            name="bad_patterns_graveyard",
            metadata=HNSW_CONFIG
        )

        where_filter = {"severity": severity} if severity else None
        results = patterns_collection.get(
            where=where_filter,
            limit=limit,
            include=["documents", "metadatas"]
        )

        patterns = []
        if results['documents'] and len(results['documents'][0]) > 0:
            for i, doc in enumerate(results['documents'][0]):
                metadata = results.get('metadatas', [[]])[0][i] if results.get('metadatas') else {}

                pattern = {
                    "name": metadata.get('name', 'unnamed_pattern'),
                    "severity": metadata.get('severity', 'medium'),
                    "description": doc,
                    "solution": metadata.get('solution', 'Review and refactor'),
                    "category": metadata.get('category', 'general')
                }
                patterns.append(pattern)

        # Sort by severity (high > medium > low)
        severity_order = {"high": 0, "medium": 1, "low": 2}
        patterns.sort(key=lambda p: severity_order.get(p['severity'], 1))

        result = BadPatternResult(
            patterns=patterns,
            total_count=len(patterns),
            last_updated=datetime.now()
        )

        return [TextContent(
            type="text",
            text=json.dumps(result.model_dump(), indent=2, default=str)
        )]

    elif name == "load_skill":
        skill_name = arguments.get("skill_name")
        resolve_dependencies = arguments.get("resolve_dependencies", True)

        if not skill_name:
            return [TextContent(
                type="text",
                text=json.dumps({"error": "skill_name is required"}, indent=2)
            )]

        # Query agentic_skills collection with connection manager
        db_client = await connection_manager.ensure_connection()
        skills_collection = db_client.get_or_create_collection(
            name="agentic_skills",
            metadata=HNSW_CONFIG
        )

        results = skills_collection.get(
            ids=[skill_name],
            include=["documents", "metadatas"]
        )

        if results['documents'] and len(results['documents'][0]) > 0:
            metadata = results['metadatas'][0][0]
            doc = results['documents'][0][0]

            # Parse schemas from JSON strings
            input_schema = metadata.get('input_schema', {})
            output_schema = metadata.get('output_schema', {})

            if isinstance(input_schema, str):
                try:
                    input_schema = json.loads(input_schema)
                except json.JSONDecodeError:
                    input_schema = {}
            if isinstance(output_schema, str):
                try:
                    output_schema = json.loads(output_schema)
                except json.JSONDecodeError:
                    output_schema = {}

            # Parse dependencies
            deps_str = metadata.get('dependencies', '[]')
            try:
                dependencies = json.loads(deps_str) if isinstance(deps_str, str) else deps_str
            except json.JSONDecodeError:
                dependencies = []

            # Resolve dependencies recursively if requested
            resolved_dependencies = []
            if resolve_dependencies and dependencies:
                for dep_name in dependencies:
                    dep_results = skills_collection.get(
                        ids=[dep_name],
                        include=["metadatas"]
                    )
                    if dep_results['metadatas'] and len(dep_results['metadatas'][0]) > 0:
                        resolved_dependencies.append(dep_results['metadatas'][0][0])

            # Validate AgentSkill interface (requires name, description)
            skill_def = SkillDefinition(
                name=metadata.get('name', skill_name),
                category=metadata.get('category', 'system_skills'),
                description=metadata.get('description', ''),
                input_schema=input_schema,
                output_schema=output_schema,
                code=doc,
                dependencies=dependencies,
                example_usage=metadata.get('example_usage', ''),
                version=metadata.get('version', '1.0')
            )

            return [TextContent(
                type="text",
                text=json.dumps({
                    **skill_def.model_dump(),
                    "resolved_dependencies": resolved_dependencies if resolve_dependencies else [],
                    "valid_agent_skill": bool(skill_def.name and skill_def.description)
                }, indent=2)
            )]
        else:
            return [TextContent(
                type="text",
                text=json.dumps({
                    "error": f"Skill '{skill_name}' not found.",
                    "suggestion": "Check if the skill is indexed in the agentic_skills collection.",
                    "available_categories": ["trading_skills", "system_skills", "data_skills"]
                }, indent=2)
            )]

    return [TextContent(type="text", text=f"Unknown tool: {name}")]

async def main():
    """Run the MCP server."""
    from mcp.server.stdio import stdio_server  # type: ignore

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())
