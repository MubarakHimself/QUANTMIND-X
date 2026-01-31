"""
Unit Tests for LangMem Memory Management

Tests semantic, episodic, and procedural memory storage and retrieval.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from hypothesis import given, strategies as st, settings

from src.memory.langmem_integration import (
    SemanticMemory,
    EpisodicMemory,
    ProceduralMemory,
    ReflectionExecutor,
    create_manage_memory_tool,
    create_search_memory_tool,
    Triple,
    Episode,
    Instruction
)


class TestSemanticMemory:
    """Tests for semantic memory."""
    
    @pytest.fixture
    def semantic_memory(self):
        """Create semantic memory with mocked database."""
        with patch('src.database.manager.DatabaseManager'):
            namespace = ("memories", "user_123", "project_456")
            memory = SemanticMemory(namespace)
            memory.db_manager = Mock()
            return memory
    
    def test_store_triple(self, semantic_memory):
        """Test storing semantic triple."""
        memory_id = semantic_memory.store_triple(
            subject="Python",
            predicate="is_a",
            obj="programming_language",
            context="knowledge_base"
        )
        
        assert memory_id.startswith("triple_")
        semantic_memory.db_manager.add_agent_memory.assert_called_once()
    
    def test_search_triples(self, semantic_memory):
        """Test searching semantic triples."""
        semantic_memory.db_manager.search_agent_memory.return_value = [
            {"content": "Python is_a programming_language"}
        ]
        
        results = semantic_memory.search_triples("Python", limit=10)
        
        assert len(results) == 1
        semantic_memory.db_manager.search_agent_memory.assert_called_once()
    
    def test_namespace_validation(self):
        """Test namespace validation."""
        # Invalid namespace (not tuple)
        with pytest.raises(ValueError):
            SemanticMemory("invalid")
        
        # Invalid namespace (too short)
        with pytest.raises(ValueError):
            SemanticMemory(("memories",))
        
        # Invalid namespace (wrong prefix)
        with pytest.raises(ValueError):
            SemanticMemory(("wrong", "user_123"))


class TestEpisodicMemory:
    """Tests for episodic memory."""
    
    @pytest.fixture
    def episodic_memory(self):
        """Create episodic memory with mocked database."""
        with patch('src.database.manager.DatabaseManager'):
            namespace = ("memories", "user_123", "project_456")
            memory = EpisodicMemory(namespace)
            memory.db_manager = Mock()
            return memory
    
    def test_store_episode(self, episodic_memory):
        """Test storing episode."""
        memory_id = episodic_memory.store_episode(
            observation="Market volatility increased",
            thoughts="Should reduce position size",
            action="Reduced risk multiplier to 0.5",
            result="Avoided large drawdown",
            agent_type="analyst"
        )
        
        assert memory_id.startswith("episode_")
        episodic_memory.db_manager.add_agent_memory.assert_called_once()
    
    def test_search_episodes(self, episodic_memory):
        """Test searching episodes."""
        episodic_memory.db_manager.search_agent_memory.return_value = [
            {"content": "Test episode", "agent_type": "analyst"}
        ]
        
        results = episodic_memory.search_episodes(
            query="volatility",
            agent_type="analyst",
            limit=10
        )
        
        assert len(results) == 1


class TestProceduralMemory:
    """Tests for procedural memory."""
    
    @pytest.fixture
    def procedural_memory(self):
        """Create procedural memory with mocked database."""
        with patch('src.database.manager.DatabaseManager'):
            namespace = ("memories", "user_123", "project_456")
            memory = ProceduralMemory(namespace)
            memory.db_manager = Mock()
            return memory
    
    def test_store_instruction(self, procedural_memory):
        """Test storing instruction."""
        memory_id = procedural_memory.store_instruction(
            task="Calculate Kelly Criterion",
            steps=["Get win rate", "Get avg win/loss", "Apply formula"],
            conditions={"min_trades": 30},
            expected_outcome="Optimal position size"
        )
        
        assert memory_id.startswith("instruction_")
        procedural_memory.db_manager.add_agent_memory.assert_called_once()
    
    def test_search_instructions(self, procedural_memory):
        """Test searching instructions."""
        procedural_memory.db_manager.search_agent_memory.return_value = [
            {"content": "Kelly Criterion calculation"}
        ]
        
        results = procedural_memory.search_instructions("Kelly", limit=10)
        
        assert len(results) == 1


class TestReflectionExecutor:
    """Tests for reflection executor."""
    
    def test_queue_memory(self):
        """Test queuing memory for consolidation."""
        executor = ReflectionExecutor(consolidation_delay=30)
        
        memory_data = {
            "type": "semantic",
            "content": "Test memory"
        }
        
        executor.queue_memory(memory_data)
        
        assert len(executor.pending_memories) == 1
        assert "queued_at" in executor.pending_memories[0]
    
    def test_consolidation_delay_enforcement(self):
        """
        Test that consolidation waits for delay period.
        
        **Validates: Property 18: Memory Consolidation Timing**
        """
        executor = ReflectionExecutor(consolidation_delay=30)
        
        # Queue memory
        memory_data = {"type": "semantic", "content": "Test"}
        executor.queue_memory(memory_data)
        
        # Immediately process - should not consolidate yet
        processed = executor.process_pending_memories()
        assert processed == 0
        assert len(executor.pending_memories) == 1
        
        # Simulate time passing
        executor.pending_memories[0]['queued_at'] = datetime.utcnow() - timedelta(minutes=31)
        
        # Now should process
        processed = executor.process_pending_memories()
        assert processed == 1
        assert len(executor.pending_memories) == 0


class TestMemoryTools:
    """Tests for memory management tools."""
    
    @patch('src.database.manager.DatabaseManager')
    def test_manage_memory_tool(self, mock_db):
        """Test memory management tool."""
        namespace = ("memories", "user_123", "project_456")
        manage_memory = create_manage_memory_tool(namespace)
        
        # Store semantic memory
        result = manage_memory(
            memory_type="semantic",
            action="store",
            subject="Test",
            predicate="is",
            obj="example",
            context="test"
        )
        
        assert result["success"] is True
        assert "memory_id" in result
    
    @patch('src.database.manager.DatabaseManager')
    def test_search_memory_tool(self, mock_db):
        """Test memory search tool."""
        namespace = ("memories", "user_123", "project_456")
        search_memory = create_search_memory_tool(namespace)
        
        # Mock search results
        mock_db.return_value.search_agent_memory.return_value = []
        
        # Search across all memory types
        result = search_memory(
            query="test query",
            memory_types=["semantic", "episodic", "procedural"],
            limit=10
        )
        
        assert result["success"] is True
        assert "results" in result


class TestMemoryNamespaceProperties:
    """
    Property tests for memory namespace hierarchy.
    
    **Validates: Property 17: Memory Namespace Hierarchy**
    **Feature: quantmindx-unified-backend, Property 17: Memory Namespace Hierarchy**
    """
    
    @given(
        user_id=st.text(min_size=1, max_size=50, alphabet=st.characters(blacklist_characters=['\x00', '\n', '\r'])),
        project_id=st.text(min_size=1, max_size=50, alphabet=st.characters(blacklist_characters=['\x00', '\n', '\r']))
    )
    @settings(max_examples=100)
    def test_namespace_hierarchy_enforced(self, user_id, project_id):
        """
        Property: Namespace MUST follow hierarchical pattern.
        
        **Validates: Requirements 10.4**
        """
        with patch('src.database.manager.DatabaseManager'):
            # Valid namespace
            namespace = ("memories", user_id, project_id)
            memory = SemanticMemory(namespace)
            
            assert memory.namespace == namespace
            assert memory.namespace[0] == "memories"
            assert len(memory.namespace) == 3
    
    @given(
        levels=st.integers(min_value=0, max_value=5)
    )
    @settings(max_examples=100)
    def test_namespace_minimum_levels(self, levels):
        """
        Property: Namespace MUST have at least 2 levels.
        
        **Validates: Requirements 10.4**
        """
        with patch('src.database.manager.DatabaseManager'):
            if levels < 2:
                # Should raise error
                namespace = tuple(["level"] * levels)
                with pytest.raises(ValueError):
                    SemanticMemory(namespace)
            else:
                # Should succeed
                namespace = ("memories",) + tuple([f"level{i}" for i in range(levels - 1)])
                memory = SemanticMemory(namespace)
                assert len(memory.namespace) >= 2
    
    @given(
        namespace_parts=st.lists(
            st.text(min_size=1, max_size=30, alphabet=st.characters(blacklist_characters=['\x00', '\n', '\r'])),
            min_size=2,
            max_size=6
        )
    )
    @settings(max_examples=100)
    def test_namespace_consistency_across_memory_types(self, namespace_parts):
        """
        Property: All memory types MUST accept same valid namespace.
        
        **Validates: Requirements 10.4**
        """
        with patch('src.database.manager.DatabaseManager'):
            namespace = ("memories",) + tuple(namespace_parts[1:])
            
            # All memory types should accept the same namespace
            semantic = SemanticMemory(namespace)
            episodic = EpisodicMemory(namespace)
            procedural = ProceduralMemory(namespace)
            
            assert semantic.namespace == namespace
            assert episodic.namespace == namespace
            assert procedural.namespace == namespace
    
    @given(
        invalid_prefix=st.text(min_size=1, max_size=20).filter(lambda x: x != "memories"),
        user_id=st.text(min_size=1, max_size=50, alphabet=st.characters(blacklist_characters=['\x00', '\n', '\r']))
    )
    @settings(max_examples=100)
    def test_namespace_prefix_validation(self, invalid_prefix, user_id):
        """
        Property: Namespace MUST start with 'memories' prefix.
        
        **Validates: Requirements 10.4**
        """
        with patch('src.database.manager.DatabaseManager'):
            namespace = (invalid_prefix, user_id)
            
            # Should raise ValueError for invalid prefix
            with pytest.raises(ValueError, match="must start with 'memories'"):
                SemanticMemory(namespace)
    
    @given(
        namespace_type=st.one_of(
            st.lists(st.text(min_size=1, max_size=20), min_size=2, max_size=5),
            st.text(min_size=1, max_size=50),
            st.integers(),
            st.dictionaries(st.text(), st.text())
        )
    )
    @settings(max_examples=100)
    def test_namespace_type_validation(self, namespace_type):
        """
        Property: Namespace MUST be a tuple type.
        
        **Validates: Requirements 10.4**
        """
        with patch('src.database.manager.DatabaseManager'):
            if not isinstance(namespace_type, tuple):
                # Should raise ValueError for non-tuple
                with pytest.raises(ValueError, match="must be a tuple"):
                    SemanticMemory(namespace_type)
            else:
                # Tuple types should pass type validation (may fail other validations)
                try:
                    SemanticMemory(namespace_type)
                except ValueError as e:
                    # Other validation errors are acceptable
                    assert "must be a tuple" not in str(e)


class TestMemoryDataClasses:
    """Tests for memory data classes."""
    
    def test_triple_creation(self):
        """Test Triple dataclass."""
        triple = Triple(
            subject="Python",
            predicate="is_a",
            object="language",
            context="test"
        )
        
        assert triple.subject == "Python"
        assert triple.object == "language"
        assert triple.timestamp is not None
    
    def test_episode_creation(self):
        """Test Episode dataclass."""
        episode = Episode(
            observation="Test observation",
            thoughts="Test thoughts",
            action="Test action",
            result="Test result"
        )
        
        assert episode.observation == "Test observation"
        assert episode.timestamp is not None
    
    def test_instruction_creation(self):
        """Test Instruction dataclass."""
        instruction = Instruction(
            task="Test task",
            steps=["Step 1", "Step 2"],
            conditions={"key": "value"},
            expected_outcome="Test outcome"
        )
        
        assert instruction.task == "Test task"
        assert len(instruction.steps) == 2
        assert instruction.timestamp is not None
