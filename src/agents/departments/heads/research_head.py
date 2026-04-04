"""
Research Department Head

Responsible for:
- Strategy research and development
- Backtesting and validation
- Alpha research and data science
- Hypothesis generation with knowledge base + web research
"""
import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.agents.departments.heads.base import DepartmentHead
from src.agents.departments.types import Department, get_department_config, SubAgentType

logger = logging.getLogger(__name__)


# Hypothesis output schema
@dataclass
class Hypothesis:
    """Structured hypothesis output format."""
    symbol: str
    timeframe: str
    hypothesis: str
    supporting_evidence: List[str]
    confidence_score: float
    recommended_next_steps: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "hypothesis": self.hypothesis,
            "supporting_evidence": self.supporting_evidence,
            "confidence_score": self.confidence_score,
            "recommended_next_steps": self.recommended_next_steps,
        }


# Research task input schema
@dataclass
class ResearchTask:
    """Research task input."""
    query: str
    symbols: Optional[List[str]] = None
    timeframes: Optional[List[str]] = None
    session_id: Optional[str] = None


class ResearchHead(DepartmentHead):
    """Research Department Head for strategy development and hypothesis generation."""

    # Confidence threshold for TRD escalation
    TRD_ESCALATION_THRESHOLD = 0.75

    def __init__(self, mail_db_path: str = ".quantmind/department_mail.db"):
        config = get_department_config(Department.RESEARCH)
        super().__init__(config=config, mail_db_path=mail_db_path)

        # Initialize knowledge clients
        self._init_knowledge_clients()

        # Track session for memory graph
        self._current_session_id: Optional[str] = None

    def _init_knowledge_clients(self):
        """Initialize knowledge retrieval clients."""
        # PageIndex client for full-text search
        try:
            from src.agents.knowledge.router import kb_router
            self.pageindex_client = kb_router
            logger.info("PageIndex client initialized for ResearchHead")
        except Exception as e:
            logger.warning(f"PageIndex client not available: {e}")
            self.pageindex_client = None

        # ChromaDB embedding service for semantic search
        try:
            from src.memory.graph.embedding_service import EmbeddingService

            chroma_path = os.environ.get("CHROMA_DB_PATH", ".quantmind/chroma_db")
            self.embedding_service = EmbeddingService(
                chroma_path=chroma_path,
                use_chroma=True
            )
            logger.info("ChromaDB embedding service initialized for ResearchHead")
        except Exception as e:
            logger.warning(f"ChromaDB embedding service not available: {e}")
            self.embedding_service = None

        # Web research client (MCP tools)
        try:
            from src.agents.mcp.integration import get_mcp_integration
            self.mcp_integration = get_mcp_integration()
            logger.info("MCP integration initialized for web research")
        except Exception as e:
            logger.warning(f"MCP integration not available: {e}")
            self.mcp_integration = None

    def get_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "develop_strategy",
                "description": "Develop a new trading strategy",
                "parameters": {
                    "type": "Strategy type (trend, mean_reversion, breakout)",
                    "symbols": "Target symbols",
                },
            },
            {
                "name": "backtest_strategy",
                "description": "Backtest a strategy on historical data",
                "parameters": {
                    "strategy_id": "Strategy identifier",
                    "start_date": "Backtest start date",
                    "end_date": "Backtest end date",
                },
            },
            {
                "name": "research_hypothesis",
                "description": "Generate a research hypothesis using knowledge base and web research",
                "parameters": {
                    "query": "Research query or topic",
                    "symbols": "Target symbols (optional)",
                    "timeframes": "Timeframes to consider (optional)",
                },
            },
            {
                "name": "search_knowledge_base",
                "description": "Search the knowledge base for relevant information",
                "parameters": {
                    "query": "Search query",
                    "collection": "Collection to search (articles, books, logs)",
                },
            },
        ]

    def _format_tools_for_anthropic(self) -> list:
        """Convert the full active tool surface to Anthropic tool definitions."""
        return super()._format_tools_for_anthropic()

    async def process_task(self, task: str, context: dict = None) -> dict:
        """
        Process a research task via Claude SDK and generate a hypothesis.

        Args:
            task: Research query or instruction string
            context: Optional canvas/session context dict

        Returns:
            Dict with status, department, content, tool_calls
        """
        import os
        import anthropic

        memory_nodes = None
        try:
            if hasattr(self, "_read_relevant_memory"):
                memory_nodes = await self._read_relevant_memory(task)
        except Exception:
            pass

        full_system = self._build_system_prompt(
            canvas_context=context,
            memory_nodes=memory_nodes,
        )

        # Get tools formatted for Anthropic
        tools = self._format_tools_for_anthropic()

        # Call Claude
        try:
            if hasattr(self, "_invoke_claude"):
                result = await self._invoke_claude(
                    task=task,
                    canvas_context=context,
                    tools=tools if tools else None,
                )
            else:
                from src.agents.providers.router import get_router

                runtime_config = get_router().resolve_runtime_config()
                if not runtime_config or not runtime_config.api_key:
                    raise RuntimeError(
                        "No LLM runtime configured. Configure a provider in Settings or set QMX_LLM_* environment variables."
                    )
                client = anthropic.AsyncAnthropic(
                    api_key=runtime_config.api_key,
                    base_url=runtime_config.base_url,
                )
                kwargs = {
                    "model": runtime_config.model,
                    "max_tokens": 4096,
                    "system": full_system,
                    "messages": [{"role": "user", "content": task}],
                }
                if tools:
                    kwargs["tools"] = tools
                resp = await client.messages.create(**kwargs)
                content = "".join(b.text for b in resp.content if b.type == "text")
                result = {"content": content, "tool_calls": []}
        except Exception as e:
            logger.error(f"{self.department.value} Claude call failed: {e}")
            return {"status": "error", "error": str(e), "department": self.department.value}

        # Dispatch hypothesis to Development if content references strategy/hypothesis
        if "hypothesis" in result.get("content", "").lower() or "strategy" in result.get("content", "").lower():
            try:
                from src.agents.departments.department_mail import MessageType, Priority
                self.mail_service.send(
                    from_dept=self.department.value,
                    to_dept=Department.DEVELOPMENT.value,
                    type=MessageType.STRATEGY_DISPATCH,
                    subject=f"Research hypothesis: {task[:80]}",
                    body=result["content"],
                    priority=Priority.NORMAL,
                )
                logger.info("Research hypothesis dispatched to Development")
            except Exception as e:
                logger.warning(f"Failed to dispatch hypothesis: {e}")

        # Write opinion to graph memory
        try:
            if hasattr(self, "_write_opinion_node") and result.get("content"):
                await self._write_opinion_node(
                    content=f"Task: {task[:200]}\nResult: {result['content'][:500]}",
                    confidence=0.7,
                    tags=[self.department.value],
                )
        except Exception:
            pass

        return {
            "status": "success",
            "department": self.department.value,
            "content": result.get("content", ""),
            "tool_calls": result.get("tool_calls", []),
        }

    def process_research_task(self, task: ResearchTask) -> Hypothesis:
        """
        Process a research task using the structured knowledge pipeline.

        Args:
            task: Research task with query and optional parameters

        Returns:
            Hypothesis object with structured output
        """
        self._current_session_id = task.session_id
        logger.info(f"Processing research task: {task.query}")

        # Step 1: Query knowledge sources in parallel
        knowledge_results = self._query_knowledge_sources(task.query)

        # Step 2: Query semantic memory
        semantic_results = self._query_semantic_memory(task.query)

        # Step 3: Perform web research
        web_results = self._perform_web_research(task.query)

        # Step 4: Combine evidence
        all_evidence = self._combine_evidence(
            knowledge_results, semantic_results, web_results
        )

        # Step 5: Generate hypothesis with LLM
        hypothesis = self._generate_hypothesis(
            query=task.query,
            evidence=all_evidence,
            symbols=task.symbols or ["EURUSD"],
            timeframes=task.timeframes or ["H4", "D1"]
        )

        # Step 6: Write OPINION node to memory graph
        self._write_research_opinion(hypothesis, task.query)

        logger.info(f"Research complete. Hypothesis generated with confidence: {hypothesis.confidence_score}")

        return hypothesis

    def _query_knowledge_sources(self, query: str) -> List[Dict[str, Any]]:
        """Query PageIndex for full-text knowledge search."""
        results = []

        if not self.pageindex_client:
            logger.warning("PageIndex client not available, skipping knowledge search")
            return results

        try:
            # Search all collections
            search_results = self.pageindex_client.search_all(query, limit_per_collection=3)

            for collection, items in search_results.items():
                for item in items:
                    results.append({
                        "source": "pageindex",
                        "collection": collection,
                        "content": item.get("content", ""),
                        "score": item.get("score", 0.0),
                    })

            logger.info(f"PageIndex search returned {len(results)} results")
        except Exception as e:
            logger.error(f"PageIndex search failed: {e}")

        return results

    def _query_semantic_memory(self, query: str) -> List[Dict[str, Any]]:
        """Query ChromaDB for semantic search."""
        results = []

        if not self.embedding_service:
            logger.warning("Embedding service not available, skipping semantic search")
            return results

        try:
            # Generate embedding for query
            query_embedding = self.embedding_service.generate_embedding(query)

            # Search the graph store for similar nodes
            # This is a simplified implementation - would need graph store integration
            # For now, we'll use the embedding service's similarity search
            if hasattr(self.embedding_service, 'search_similar'):
                similar_nodes = self.embedding_service.search_similar(
                    query_embedding,
                    limit=5
                )
                for node in similar_nodes:
                    results.append({
                        "source": "chroma",
                        "content": node.get("content", ""),
                        "score": node.get("score", 0.0),
                    })

            logger.info(f"Semantic search returned {len(results)} results")
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")

        return results

    def _perform_web_research(self, query: str) -> List[Dict[str, Any]]:
        """Perform web research using MCP tools or fallback to PageIndex web content."""
        results = []

        # Try MCP integration first (primary web research path)
        if self.mcp_integration:
            try:
                logger.info(f"Web research via MCP for query: {query}")
                # Results would come from MCP web search tool
                # Implementation depends on MCP tools server capabilities
            except Exception as e:
                logger.error(f"MCP web research failed: {e}")

        # Fallback: Search PageIndex for web-crawled content
        if not results and self.pageindex_client:
            try:
                logger.info("Falling back to PageIndex for web content")
                search_results = self.pageindex_client.search_all(query, limit_per_collection=2)
                for collection, items in search_results.items():
                    for item in items:
                        results.append({
                            "source": "pageindex_web_fallback",
                            "collection": collection,
                            "content": item.get("content", ""),
                            "score": item.get("score", 0.0),
                        })
                logger.info(f"PageIndex fallback returned {len(results)} web results")
            except Exception as e:
                logger.error(f"PageIndex web fallback failed: {e}")

        if not results:
            logger.warning("No web research results available (MCP unavailable, PageIndex fallback empty)")

        return results

    def _combine_evidence(
        self,
        knowledge_results: List[Dict[str, Any]],
        semantic_results: List[Dict[str, Any]],
        web_results: List[Dict[str, Any]]
    ) -> List[str]:
        """Combine evidence from all sources."""
        evidence = []

        # Add knowledge base evidence
        for result in knowledge_results:
            content = result.get("content", "")
            if content:
                evidence.append(f"[KB-{result.get('collection', 'unknown')}]: {content[:200]}")

        # Add semantic memory evidence
        for result in semantic_results:
            content = result.get("content", "")
            if content:
                evidence.append(f"[Memory]: {content[:200]}")

        # Add web research evidence
        for result in web_results:
            content = result.get("content", "")
            if content:
                evidence.append(f"[Web]: {content[:200]}")

        return evidence

    def _generate_hypothesis(
        self,
        query: str,
        evidence: List[str],
        symbols: List[str],
        timeframes: List[str]
    ) -> Hypothesis:
        """
        Generate hypothesis using LLM with evidence from all sources.

        This is a simplified implementation. In production, this would
        call the LLM to synthesize evidence into a hypothesis.
        """
        # Calculate confidence based on evidence quality
        confidence = self._calculate_confidence(evidence)

        # Build hypothesis (simplified - would use LLM in production)
        hypothesis_text = self._synthesize_hypothesis_text(query, evidence)

        # Generate recommended next steps
        next_steps = self._generate_next_steps(confidence, evidence)

        return Hypothesis(
            symbol=symbols[0] if symbols else "EURUSD",
            timeframe=timeframes[0] if timeframes else "H4",
            hypothesis=hypothesis_text,
            supporting_evidence=evidence[:5],  # Top 5 evidence items
            confidence_score=confidence,
            recommended_next_steps=next_steps
        )

    def _calculate_confidence(self, evidence: List[str]) -> float:
        """
        Calculate confidence score (0-1) based on evidence quality.

        In production, this would analyze:
        - Number of evidence sources
        - Quality/relevance scores
        - Consistency across sources
        """
        if not evidence:
            return 0.1

        # Base confidence from evidence count
        base_confidence = min(len(evidence) / 10.0, 0.5)

        # Additional factors would be considered in production
        # For now, return a reasonable confidence based on evidence
        return min(base_confidence + 0.3, 0.95)

    def _synthesize_hypothesis_text(self, query: str, evidence: List[str]) -> str:
        """Synthesize hypothesis text from evidence."""
        if not evidence:
            return f"Insufficient evidence to form hypothesis for: {query}"

        # In production, this would use LLM to synthesize
        # For now, create a basic hypothesis
        evidence_summary = " | ".join([e[:50] for e in evidence[:3]])
        return f"Based on research: {query}. Evidence: {evidence_summary}"

    def _generate_next_steps(self, confidence: float, evidence: List[str]) -> List[str]:
        """Generate recommended next steps based on confidence and evidence."""
        next_steps = []

        if confidence >= self.TRD_ESCALATION_THRESHOLD:
            next_steps.append("Proceed to TRD Generation - escalate to Development department")
            next_steps.append("Schedule backtest validation")

        if confidence < 0.5:
            next_steps.append("Gather more evidence through extended research")
            next_steps.append("Consider alternative hypotheses")

        next_steps.append("Document findings in research repository")
        next_steps.append("Review with Risk department")

        return next_steps

    def _write_research_opinion(
        self,
        hypothesis: Hypothesis,
        original_query: str
    ) -> None:
        """
        Write OPINION node to memory graph after research action.

        Args:
            hypothesis: Generated hypothesis
            original_query: Original research query
        """
        try:
            from src.memory.graph.facade import get_graph_memory
            from src.memory.graph.types import (
                MemoryNode,
                MemoryNodeType,
                MemoryCategory,
                MemoryTier,
                SessionStatus,
            )
            facade = get_graph_memory()

            # Create OPINION node
            opinion_node = MemoryNode(
                node_type=MemoryNodeType.OPINION,
                category=MemoryCategory.SUBJECTIVE,
                title=f"Research Hypothesis: {hypothesis.symbol}",
                content=hypothesis.hypothesis,
                department=self.department.value,
                agent_id=self.agent_type,
                session_id=self._current_session_id,
                role="research_head",
                session_status=SessionStatus.COMMITTED,
                tier=MemoryTier.HOT,
                # OPINION-specific fields
                action=f"Generated hypothesis for {hypothesis.symbol}",
                reasoning=f"Based on {len(hypothesis.supporting_evidence)} evidence sources",
                confidence=hypothesis.confidence_score,
                alternatives_considered="Multiple timeframes and symbols evaluated",
                constraints_applied="Evidence quality and confidence threshold",
                agent_role="research_head"
            )

            created = facade.store.create_node(opinion_node)
            logger.info(f"Wrote OPINION node to memory graph: {created.id}")

        except Exception as e:
            logger.error(f"Failed to write OPINION node: {e}")

    async def process_research_with_subagents(
        self,
        task: ResearchTask,
        subagent_types: Optional[List[str]] = None
    ) -> Hypothesis:
        """
        Process research task with parallel sub-agents.

        Args:
            task: Research task
            subagent_types: List of sub-agent types to spawn (defaults to research sub-agents)

        Returns:
            Combined hypothesis from sub-agent results

        Note: Current implementation spawns sub-agents but does not wait for results.
        The spawn_worker() call is synchronous and returns immediately with agent_id.
        Full async result collection requires callback/streaming infrastructure.
        For now, falls back to local processing after spawning.
        """
        from src.agents.departments.types import SubAgentType

        if subagent_types is None:
            # Use SubAgentType enum values for research department sub-agents
            subagent_types = [
                SubAgentType.STRATEGY_RESEARCHER.value,
                SubAgentType.MARKET_ANALYST.value,
                SubAgentType.BACKTESTER.value,
            ]

        logger.info(f"Spawning {len(subagent_types)} sub-agents for parallel research")

        # Spawn sub-agents in parallel
        subagent_tasks = []
        for subagent_type in subagent_types:
            task_input = {
                "task": task.query,
                "symbols": task.symbols,
                "timeframes": task.timeframes,
                "session_id": task.session_id,
            }
            spawn_result = self.spawn_worker(
                worker_type=subagent_type,
                task=task.query,
                input_data=task_input
            )
            logger.info(f"Spawned {subagent_type}: {spawn_result}")
            subagent_tasks.append(spawn_result)

        # Note: sub-agent result collection not fully implemented
        # spawn_worker is synchronous - returns agent_id but doesn't wait for completion
        # TODO: Implement async result collection via callback or streaming
        # For now, process locally as fallback
        logger.info("Sub-agent results collection not implemented - falling back to local processing")
        return self.process_research_task(task)

    def should_escalate_to_trd(self, hypothesis: Hypothesis) -> bool:
        """
        Check if hypothesis meets TRD escalation criteria.

        Args:
            hypothesis: Generated hypothesis

        Returns:
            True if confidence >= threshold
        """
        return hypothesis.confidence_score >= self.TRD_ESCALATION_THRESHOLD

    def get_escalation_prompt(self, hypothesis: Hypothesis) -> str:
        """
        Get TRD escalation prompt for conversation thread.

        Args:
            hypothesis: Generated hypothesis

        Returns:
            Formatted escalation prompt
        """
        if not self.should_escalate_to_trd(hypothesis):
            return ""

        return f"""Research complete — hypothesis generated with {hypothesis.confidence_score:.0%} confidence.

**Symbol:** {hypothesis.symbol}
**Timeframe:** {hypothesis.timeframe}
**Hypothesis:** {hypothesis.hypothesis}

**Supporting Evidence:** {len(hypothesis.supporting_evidence)} sources
**Recommended Next Steps:**
{chr(10).join(f"- {step}" for step in hypothesis.recommended_next_steps)}

---

**Proceed to TRD?** (Confidence: {hypothesis.confidence_score:.0%} >= {self.TRD_ESCALATION_THRESHOLD:.0%})
"""
