"""
Storage for Paper Trading Results in ChromaDB.

Stores and retrieves paper trading performance metrics, trade events,
and agent history in ChromaDB for analysis and comparison.
"""

import json
import logging
from datetime import datetime, UTC, timedelta
from typing import Optional, List, Dict, Any

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from .models import AgentPerformance

logger = logging.getLogger(__name__)


class PaperTradingStorage:
    """
    Storage for paper trading results using ChromaDB.

    Features:
    - Store trade events with embeddings for semantic search
    - Store agent performance metrics
    - Query by agent, symbol, date range
    - Semantic search for similar strategies

    Collections:
    - paper_trading_results: Trade events and performance
    - agent_configurations: Strategy configurations

    Example:
        ```python
        storage = PaperTradingStorage()

        # Store trade event
        storage.store_trade_event(
            agent_id="strategy-rsi-001",
            event_type="entry",
            symbol="EURUSD",
            price=1.0850,
            lots=0.1,
            metadata={"rsi": 25.5}
        )

        # Store performance
        storage.store_performance(
            agent_id="strategy-rsi-001",
            performance=AgentPerformance(...)
        )

        # Query trades
        trades = storage.get_agent_trades("strategy-rsi-001")

        # Search similar strategies
        results = storage.search_similar_strategies(
            query="RSI reversal with oversold entry",
            n_results=5
        )
        ```
    """

    def __init__(
        self,
        persist_directory: str = "./data/chroma",
        collection_name: str = "paper_trading_results",
        use_embeddings: bool = True,
    ):
        """
        Initialize ChromaDB storage.

        Args:
            persist_directory: Directory for persistent storage
            collection_name: Name of the main collection
            use_embeddings: Whether to use embeddings for semantic search
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.use_embeddings = use_embeddings

        # Initialize ChromaDB client
        self._client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

        # Setup embedding function
        if use_embeddings:
            self._embedding_function = embedding_functions.DefaultEmbeddingFunction()
        else:
            self._embedding_function = None

        # Get or create collection
        self._collection = self._get_or_create_collection(collection_name)

        logger.info(f"ChromaDB storage initialized: {persist_directory}")

    def _get_or_create_collection(self, name: str):
        """Get or create a ChromaDB collection."""
        try:
            return self._client.get_collection(name)
        except Exception:
            return self._client.create_collection(
                name=name,
                embedding_function=self._embedding_function,
            )

    # ========================================================================
    # Store Trade Events
    # ========================================================================

    def store_trade_event(
        self,
        agent_id: str,
        event_type: str,  # entry, exit, stop_hit, take_profit
        symbol: str,
        price: float,
        lots: float,
        pnl: Optional[float] = None,
        order_id: Optional[int] = None,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Store a trade event in ChromaDB.

        Args:
            agent_id: Agent identifier
            event_type: Type of event (entry, exit, stop_hit, take_profit)
            symbol: Trading symbol
            price: Execution price
            lots: Trade size
            pnl: Profit/loss (for exits)
            order_id: MT5 order ticket
            timestamp: Event timestamp (default: now)
            metadata: Additional metadata (indicators, etc.)

        Returns:
            Document ID
        """
        if timestamp is None:
            timestamp = datetime.now(UTC)

        # Create document ID
        doc_id = f"{agent_id}_{event_type}_{int(timestamp.timestamp())}_{symbol}"

        # Create document text for embedding
        doc_text = self._create_trade_document(
            agent_id=agent_id,
            event_type=event_type,
            symbol=symbol,
            price=price,
            lots=lots,
            metadata=metadata,
        )

        # Prepare metadata
        doc_metadata = {
            "agent_id": agent_id,
            "event_type": event_type,
            "symbol": symbol,
            "price": str(price),
            "lots": str(lots),
            "timestamp": timestamp.isoformat(),
            "timestamp_unix": int(timestamp.timestamp()),
        }

        if pnl is not None:
            doc_metadata["pnl"] = str(pnl)

        if order_id is not None:
            doc_metadata["order_id"] = str(order_id)

        if metadata:
            # Flatten metadata for ChromaDB (must be string, int, float, bool)
            for k, v in metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    doc_metadata[k] = v
                else:
                    doc_metadata[k] = json.dumps(v)

        # Add to collection
        self._collection.add(
            documents=[doc_text],
            metadatas=[doc_metadata],
            ids=[doc_id],
        )

        logger.debug(f"Stored trade event: {doc_id}")
        return doc_id

    def _create_trade_document(
        self,
        agent_id: str,
        event_type: str,
        symbol: str,
        price: float,
        lots: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create document text for embedding."""
        parts = [
            f"Agent: {agent_id}",
            f"Event: {event_type}",
            f"Symbol: {symbol}",
            f"Price: {price}",
            f"Lots: {lots}",
        ]

        if metadata:
            parts.append("Indicators:")
            for k, v in metadata.items():
                if isinstance(v, (int, float)):
                    parts.append(f"  {k}: {v:.2f}")
                else:
                    parts.append(f"  {k}: {v}")

        return "\n".join(parts)

    # ========================================================================
    # Store Performance
    # ========================================================================

    def store_performance(
        self,
        agent_id: str,
        performance: AgentPerformance,
        configuration: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Store agent performance metrics.

        Args:
            agent_id: Agent identifier
            performance: AgentPerformance object
            configuration: Strategy configuration

        Returns:
            Document ID
        """
        # Create document ID
        doc_id = f"{agent_id}_performance_{int(performance.calculated_at.timestamp())}"

        # Create document text for embedding
        doc_text = self._create_performance_document(performance, configuration)

        # Prepare metadata
        doc_metadata = {
            "agent_id": agent_id,
            "document_type": "performance",
            "total_trades": performance.total_trades,
            "win_rate": performance.win_rate,
            "total_pnl": str(performance.total_pnl),
            "average_pnl": str(performance.average_pnl),
            "max_drawdown": str(performance.max_drawdown),
            "profit_factor": str(performance.profit_factor),
            "calculated_at": performance.calculated_at.isoformat(),
            "calculated_at_unix": int(performance.calculated_at.timestamp()),
        }

        if performance.sharpe_ratio is not None:
            doc_metadata["sharpe_ratio"] = str(performance.sharpe_ratio)

        if configuration:
            config_str = json.dumps(configuration)
            doc_metadata["configuration"] = config_str

        # Add to collection
        self._collection.add(
            documents=[doc_text],
            metadatas=[doc_metadata],
            ids=[doc_id],
        )

        logger.info(f"Stored performance for {agent_id}: {performance.total_trades} trades")
        return doc_id

    def _create_performance_document(
        self,
        performance: AgentPerformance,
        configuration: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create document text for embedding."""
        parts = [
            f"Agent Performance Summary",
            f"Total Trades: {performance.total_trades}",
            f"Win Rate: {performance.win_rate:.2f}%",
            f"Total P&L: ${performance.total_pnl:.2f}",
            f"Average P&L: ${performance.average_pnl:.2f}",
            f"Max Drawdown: ${performance.max_drawdown:.2f}",
            f"Profit Factor: {performance.profit_factor:.2f}",
        ]

        if performance.sharpe_ratio:
            parts.append(f"Sharpe Ratio: {performance.sharpe_ratio:.2f}")

        if performance.symbols_traded:
            parts.append(f"Symbols: {', '.join(performance.symbols_traded)}")

        if configuration:
            parts.append("\nConfiguration:")
            for k, v in configuration.items():
                if isinstance(v, (int, float)):
                    parts.append(f"  {k}: {v}")
                elif isinstance(v, list):
                    parts.append(f"  {k}: {', '.join(str(x) for x in v)}")
                else:
                    parts.append(f"  {k}: {v}")

        return "\n".join(parts)

    # ========================================================================
    # Query Trade Events
    # ========================================================================

    def get_agent_trades(
        self,
        agent_id: str,
        limit: int = 1000,
        event_type: Optional[str] = None,
        symbol: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get all trade events for an agent.

        Args:
            agent_id: Agent identifier
            limit: Maximum number of results
            event_type: Filter by event type
            symbol: Filter by symbol

        Returns:
            List of trade events with metadata
        """
        # Build filter
        where = {"agent_id": agent_id}
        if event_type:
            where["event_type"] = event_type
        if symbol:
            where["symbol"] = symbol

        # Query
        results = self._collection.query(
            query_texts=[""] if self.use_embeddings else None,
            n_results=limit,
            where=where,
        )

        # Format results
        trades = []
        if results and results["metadatas"]:
            for i, metadata in enumerate(results["metadatas"][0]):
                trades.append({
                    "id": results["ids"][0][i],
                    "metadata": metadata,
                })

        return trades

    def get_agent_performance(
        self,
        agent_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get latest performance metrics for an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Performance metadata or None
        """
        # Query for performance documents
        results = self._collection.query(
            query_texts=[""] if self.use_embeddings else None,
            n_results=1,
            where={
                "agent_id": agent_id,
                "document_type": "performance",
            },
        )

        if results and results["metadatas"] and results["metadatas"][0]:
            return results["metadatas"][0][0]

        return None

    # ========================================================================
    # Semantic Search
    # ========================================================================

    def search_similar_strategies(
        self,
        query: str,
        n_results: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar strategies using semantic search.

        Requires embeddings to be enabled.

        Args:
            query: Search query
            n_results: Number of results
            filters: Metadata filters

        Returns:
            List of matching documents
        """
        if not self.use_embeddings:
            logger.warning("Embeddings disabled, semantic search not available")
            return []

        # Query
        results = self._collection.query(
            query_texts=[query],
            n_results=n_results,
            where=filters,
        )

        # Format results
        matches = []
        if results and results["metadatas"]:
            for i, metadata in enumerate(results["metadatas"][0]):
                matches.append({
                    "id": results["ids"][0][i],
                    "metadata": metadata,
                    "distance": results["distances"][0][i] if "distances" in results else None,
                })

        return matches

    # ========================================================================
    # Analytics
    # ========================================================================

    def get_top_performers(
        self,
        min_trades: int = 10,
        n_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Get top performing agents.

        Args:
            min_trades: Minimum number of trades
            n_results: Number of results

        Returns:
            List of top performers
        """
        # Get all performance documents
        results = self._collection.query(
            query_texts=[""] if self.use_embeddings else None,
            n_results=n_results * 10,  # Get more to filter
            where={"document_type": "performance"},
        )

        if not results or not results["metadatas"]:
            return []

        # Filter and sort
        performers = []
        for metadata in results["metadatas"][0]:
            total_trades = metadata.get("total_trades", 0)
            if total_trades >= min_trades:
                performers.append(metadata)

        # Sort by total P&L
        performers.sort(
            key=lambda x: float(x.get("total_pnl", 0)),
            reverse=True,
        )

        return performers[:n_results]

    def get_symbol_performance(
        self,
        symbol: str,
    ) -> Dict[str, Any]:
        """
        Get aggregated performance for a symbol across all agents.

        Args:
            symbol: Trading symbol

        Returns:
            Aggregated metrics
        """
        # Get all trades for symbol
        results = self._collection.query(
            query_texts=[""] if self.use_embeddings else None,
            n_results=10000,
            where={"symbol": symbol, "event_type": "exit"},
        )

        if not results or not results["metadatas"]:
            return {"symbol": symbol, "total_trades": 0}

        # Aggregate
        total_trades = len(results["metadatas"][0])
        total_pnl = 0.0
        wins = 0

        for metadata in results["metadatas"][0]:
            pnl = float(metadata.get("pnl", 0))
            total_pnl += pnl
            if pnl > 0:
                wins += 1

        return {
            "symbol": symbol,
            "total_trades": total_trades,
            "total_pnl": total_pnl,
            "win_rate": (wins / total_trades * 100) if total_trades > 0 else 0,
            "average_pnl": total_pnl / total_trades if total_trades > 0 else 0,
        }

    # ========================================================================
    # Cleanup
    # ========================================================================

    def delete_agent_data(self, agent_id: str) -> int:
        """
        Delete all data for an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Number of documents deleted
        """
        # Get all documents for agent
        results = self._collection.query(
            query_texts=[""] if self.use_embeddings else None,
            n_results=10000,
            where={"agent_id": agent_id},
        )

        if not results or not results["ids"]:
            return 0

        # Delete
        ids_to_delete = results["ids"][0]
        self._collection.delete(ids=ids_to_delete)

        logger.info(f"Deleted {len(ids_to_delete)} documents for agent {agent_id}")
        return len(ids_to_delete)

    def close(self):
        """Close storage connection."""
        # ChromaDB PersistentClient doesn't need explicit closing
        logger.debug("ChromaDB storage closed")
