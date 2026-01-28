#!/usr/bin/env python3
"""
Focused tests for ChromaDB schema extension and data layer operations.
Tests cover: collection creation, document insertion, query operations, and Git file operations.

Run only these tests: pytest tests/mcp_servers/quantmindx_kb/test_chromadb_data_layer.py -v
"""

import pytest
import sys
from pathlib import Path
import tempfile
import shutil

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestChromaDBCollections:
    """Test ChromaDB collection creation with HNSW configuration."""

    @pytest.fixture
    def temp_chroma_path(self):
        """Create temporary directory for test ChromaDB data."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def chroma_client(self, temp_chroma_path):
        """Create ChromaDB client with temporary storage."""
        try:
            import chromadb
        except ImportError:
            pytest.skip("ChromaDB not installed")

        client = chromadb.PersistentClient(path=temp_chroma_path)
        return client

    def test_collection_creation_with_hnsw_config(self, chroma_client):
        """Test collection creation with HNSW index configuration (M=16, ef_construction=100, cosine)."""
        collection = chroma_client.get_or_create_collection(
            name="test_algorithm_templates",
            metadata={"hnsw:space": "cosine", "hnsw:M": 16, "hnsw:construction_ef": 100}
        )

        assert collection is not None
        assert collection.name == "test_algorithm_templates"
        assert collection.count() == 0

    def test_document_insertion_with_metadata(self, chroma_client):
        """Test document insertion with metadata for algorithm templates."""
        collection = chroma_client.get_or_create_collection(
            name="test_templates_insert",
            metadata={"hnsw:space": "cosine"}
        )

        # Insert test document
        collection.add(
            ids=["template_001"],
            documents=["Basic RSI strategy template with entry and exit logic"],
            metadatas=[{
                "name": "RSI Basic",
                "category": "mean_reversion",
                "language": "mq5",
                "complexity": "basic",
                "version": "1.0"
            }]
        )

        assert collection.count() == 1

        # Verify retrieval
        results = collection.get(ids=["template_001"], include=["metadatas"])
        assert results["metadatas"][0]["category"] == "mean_reversion"
        assert results["metadatas"][0]["language"] == "mq5"

    def test_query_operations_with_filters(self, chroma_client):
        """Test query operations with metadata filters."""
        collection = chroma_client.get_or_create_collection(
            name="test_templates_query",
            metadata={"hnsw:space": "cosine"}
        )

        # Insert multiple documents
        collection.add(
            ids=["tmpl_001", "tmpl_002", "tmpl_003"],
            documents=[
                "RSI mean reversion strategy",
                "Moving average crossover trend following",
                "Bollinger bands breakout strategy"
            ],
            metadatas=[
                {"category": "mean_reversion", "language": "mq5", "complexity": "basic"},
                {"category": "trend_following", "language": "mq5", "complexity": "intermediate"},
                {"category": "breakout", "language": "python", "complexity": "advanced"}
            ]
        )

        # Query without filter
        results = collection.query(query_texts=["trading strategy"], n_results=3)
        assert len(results["ids"][0]) == 3

        # Query with category metadata filter (using where)
        # Note: ChromaDB where clause syntax
        results_filtered = collection.query(
            query_texts=["strategy"],
            n_results=10,
            where={"category": "mean_reversion"}
        )
        assert len(results_filtered["ids"][0]) == 1
        assert results_filtered["metadatas"][0][0]["category"] == "mean_reversion"


class TestGitFileOperations:
    """Test Git client operations for template storage."""

    @pytest.fixture
    def temp_git_repo(self):
        """Create temporary Git repository for testing."""
        import subprocess

        temp_dir = tempfile.mkdtemp()
        original_dir = Path.cwd()

        # Initialize git repo
        subprocess.run(["git", "init"], cwd=temp_dir, capture_output=True, check=True)
        subprocess.run(
            ["git", "config", "user.email", "test@quantmindx.com"],
            cwd=temp_dir,
            capture_output=True,
            check=True
        )
        subprocess.run(
            ["git", "config", "user.name", "Test Bot"],
            cwd=temp_dir,
            capture_output=True,
            check=True
        )

        yield temp_dir

        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_read_template_from_git(self, temp_git_repo):
        """Test reading template file from Git repository."""
        templates_dir = Path(temp_git_repo) / "templates"
        templates_dir.mkdir(parents=True)

        # Create test template file
        template_file = templates_dir / "rsi_basic.mq5"
        template_content = "// RSI Strategy Template\ninput int RSI_Period = 14;"
        template_file.write_text(template_content)

        # Commit the file
        import subprocess
        subprocess.run(["git", "add", "."], cwd=temp_git_repo, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "Add template"],
            cwd=temp_git_repo,
            capture_output=True,
            check=True
        )

        # Read it back
        read_content = template_file.read_text()
        assert "RSI_Period" in read_content
        assert read_content == template_content

    def test_write_template_to_git(self, temp_git_repo):
        """Test writing template file to Git repository."""
        templates_dir = Path(temp_git_repo) / "templates" / "mean_reversion"
        templates_dir.mkdir(parents=True)

        # Write new template
        template_file = templates_dir / "new_strategy.mq5"
        template_content = "// New strategy\ntemplate<typename T>"
        template_file.write_text(template_content)

        assert template_file.exists()
        assert template_file.read_text() == template_content

    def test_commit_template_changes(self, temp_git_repo):
        """Test committing template changes to Git repository."""
        import subprocess

        templates_dir = Path(temp_git_repo) / "templates"
        templates_dir.mkdir(parents=True)

        # Create and commit initial file
        template_file = templates_dir / "strategy.mq5"
        template_file.write_text("// Version 1")
        subprocess.run(["git", "add", "."], cwd=temp_git_repo, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "Initial commit"],
            cwd=temp_git_repo,
            capture_output=True,
            check=True
        )

        # Modify and commit changes
        template_file.write_text("// Version 2\n// Updated logic")
        subprocess.run(["git", "add", "."], cwd=temp_git_repo, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "Update strategy logic"],
            cwd=temp_git_repo,
            capture_output=True,
            check=True
        )

        # Verify commit history
        result = subprocess.run(
            ["git", "log", "--oneline"],
            cwd=temp_git_repo,
            capture_output=True,
            text=True,
            check=True
        )
        assert "Update strategy logic" in result.stdout
        assert "Initial commit" in result.stdout


class TestChromaDBSchemaExtension:
    """Test ChromaDB schema extension for new collections."""

    @pytest.fixture
    def temp_chroma_path(self):
        """Create temporary directory for test ChromaDB data."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def extended_chroma_client(self, temp_chroma_path):
        """Create ChromaDB client with extended schema."""
        try:
            import chromadb
        except ImportError:
            pytest.skip("ChromaDB not installed")

        client = chromadb.PersistentClient(path=temp_chroma_path)

        # Create all required collections
        collections_config = [
            ("mql5_knowledge", "Existing MQL5 articles collection"),
            ("algorithm_templates", "Algorithm template library"),
            ("agentic_skills", "Agentic skill definitions"),
            ("coding_standards", "Project coding standards"),
            ("bad_patterns_graveyard", "Anti-patterns to avoid")
        ]

        for collection_name, description in collections_config:
            client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine", "hnsw:M": 16, "hnsw:construction_ef": 100}
            )

        return client

    def test_all_collections_created(self, extended_chroma_client):
        """Test that all 5 collections are created and accessible."""
        expected_collections = {
            "mql5_knowledge",
            "algorithm_templates",
            "agentic_skills",
            "coding_standards",
            "bad_patterns_graveyard"
        }

        # List all collections
        existing_collections = {coll.name for coll in extended_chroma_client.list_collections()}

        assert existing_collections == expected_collections

    def test_skills_collection_schema(self, extended_chroma_client):
        """Test skills collection accepts skill definition metadata."""
        skills_collection = extended_chroma_client.get_collection("agentic_skills")

        # Insert skill with full metadata
        skills_collection.add(
            ids=["skill_001"],
            documents=["Calculate RSI indicator with configurable period"],
            metadatas=[{
                "name": "calculate_rsi",
                "category": "trading_skills",
                "dependencies": "fetch_market_data",
                "version": "1.2",
                "input_schema": '{"period": "int", "symbol": "str"}',
                "output_schema": '{"rsi_value": "float"}'
            }]
        )

        assert skills_collection.count() == 1

        # Verify schema metadata
        result = skills_collection.get(ids=["skill_001"], include=["metadatas"])
        metadata = result["metadatas"][0]
        assert metadata["category"] == "trading_skills"
        assert "input_schema" in metadata
        assert "output_schema" in metadata

    def test_bad_patterns_collection_severity(self, extended_chroma_client):
        """Test bad patterns collection supports severity levels."""
        patterns_collection = extended_chroma_client.get_collection("bad_patterns_graveyard")

        # Insert anti-patterns with different severities
        patterns_collection.add(
            ids=["pattern_001", "pattern_002"],
            documents=[
                "Hardcoded stop loss values fail to adapt to market volatility",
                "Missing position sizing leads to oversized trades"
            ],
            metadatas=[
                {"name": "Hardcoded Stop Loss", "severity": "high", "category": "risk_management"},
                {"name": "No Position Sizing", "severity": "medium", "category": "risk_management"}
            ]
        )

        assert patterns_collection.count() == 2

        # Query by severity
        results = patterns_collection.query(
            query_texts=["stop loss"],
            n_results=10,
            where={"severity": "high"}
        )

        assert len(results["ids"][0]) == 1
        assert results["metadatas"][0][0]["severity"] == "high"
