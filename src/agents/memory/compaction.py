"""
Session Compaction - Context token reduction for agent sessions.

Provides:
- SessionCompactor class for token reduction
- Critical state preservation (config, recent decisions)
- Non-critical history summarization
- Compaction thresholds with background processing support
- MemoryCompactor class for persistent memory optimization

Reference: https://platform.claude.com/cookbook/misc-session-memory-compaction
"""

import hashlib
import json
import logging
import os
import re
import sqlite3
import threading
import time
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)


class LLMClientProtocol(Protocol):
    """Protocol for LLM client used in summarization."""
    def create_completion(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]: ...


def estimate_tokens(text: str) -> int:
    """Estimate token count: ~4 chars per token."""
    return max(1, len(text) // 4) if text else 0


def count_tokens(text: str) -> int:
    """Alias for estimate_tokens for API clarity."""
    return estimate_tokens(text)


def remove_thinking_blocks(text: str) -> tuple:
    """Remove <think> blocks from text."""
    pattern = r"<think>.*?</think>"
    matches = re.findall(pattern, text, flags=re.DOTALL)
    cleaned = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()
    return cleaned, "".join(matches)


@dataclass
class CompactionResult:
    """Result of a compaction operation."""
    original_tokens: int
    compacted_tokens: int
    reduction_ratio: float
    summary: str
    preserved_messages: List[Dict[str, Any]]
    compacted_messages: List[Dict[str, Any]]
    duration_ms: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class CompactionConfig:
    """Configuration for session compaction."""
    context_limit: int = 10000
    min_tokens_to_init: int = 7500
    min_tokens_between_updates: int = 2000
    preserve_recent_count: int = 4
    preserve_message_types: List[str] = field(default_factory=lambda: ["system", "user"])
    summary_max_tokens: int = 2000
    enable_background: bool = True
    summarization_prompt: str = field(default_factory=lambda: (
        "Analyze the conversation and create a concise summary covering:\n"
        "1. Key topics and goals\n2. Important decisions\n3. Pending tasks\n"
        "4. User preferences\n\nConversation:\n{conversation}"
    ))


class SessionCompactor:
    """
    Session compaction for reducing token usage while preserving critical state.

    Usage:
        compactor = SessionCompactor(config=CompactionConfig(context_limit=10000))
        if compactor.should_compact(messages):
            result = compactor.compact(messages)
            messages = result.preserved_messages
    """

    def __init__(self, config: Optional[CompactionConfig] = None, llm_client: Optional[LLMClientProtocol] = None):
        self.config = config or CompactionConfig()
        self.llm_client = llm_client
        self._session_summary: Optional[str] = None
        self._last_summarized_index = 0
        self._tokens_at_last_update = 0
        self._update_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def estimate_message_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """Estimate total tokens in message list."""
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        total += estimate_tokens(item.get("text", ""))
            else:
                total += estimate_tokens(str(content))
            total += 4  # Role overhead
        return total

    def should_compact(self, messages: List[Dict[str, Any]]) -> bool:
        """Check if compaction should be triggered."""
        return self.estimate_message_tokens(messages) >= self.config.context_limit

    def should_init_background(self, messages: List[Dict[str, Any]]) -> bool:
        """Check if background summarization should start."""
        if not self.config.enable_background or self._session_summary:
            return False
        return self.estimate_message_tokens(messages) >= self.config.min_tokens_to_init

    def should_update_background(self, messages: List[Dict[str, Any]]) -> bool:
        """Check if background summary should be updated."""
        if not self._session_summary:
            return False
        tokens = self.estimate_message_tokens(messages)
        return (tokens - self._tokens_at_last_update) >= self.config.min_tokens_between_updates

    def compact(self, messages: List[Dict[str, Any]], summary: Optional[str] = None) -> CompactionResult:
        """Perform blocking compaction on messages."""
        start_time = time.perf_counter()
        original_tokens = self.estimate_message_tokens(messages)
        critical, non_critical = self._separate_critical_messages(messages)

        if summary is None and non_critical:
            summary = self._generate_summary(non_critical)

        preserved = self._build_compacted_messages(critical, summary)
        compacted_tokens = self.estimate_message_tokens(preserved)
        reduction = (original_tokens - compacted_tokens) / original_tokens if original_tokens else 0.0

        return CompactionResult(
            original_tokens=original_tokens, compacted_tokens=compacted_tokens,
            reduction_ratio=reduction, summary=summary or "", preserved_messages=preserved,
            compacted_messages=non_critical, duration_ms=(time.perf_counter() - start_time) * 1000
        )

    def instant_compact(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Perform instant compaction using pre-built session summary."""
        if self._update_thread and self._update_thread.is_alive():
            logger.info("Waiting for background summarization...")
            self._update_thread.join(timeout=30.0)

        with self._lock:
            summary = self._session_summary

        if summary is None:
            logger.warning("No summary, performing synchronous compaction")
            return self.compact(messages).preserved_messages

        with self._lock:
            unsummarized = messages[self._last_summarized_index:]

        return self._build_compacted_messages(unsummarized, summary)

    def _separate_critical_messages(self, messages: List[Dict[str, Any]]):
        """Separate critical from non-critical messages."""
        critical = [m for m in messages if m.get("role") == "system"]
        other = [m for m in messages if m.get("role") != "system"]
        recent = self.config.preserve_recent_count

        if len(other) <= recent:
            critical.extend(other)
            non_critical = []
        else:
            critical.extend(other[-recent:])
            non_critical = other[:-recent]
        return critical, non_critical

    def _generate_summary(self, messages: List[Dict[str, Any]]) -> str:
        """Generate summary of non-critical messages."""
        if not messages:
            return ""
        conversation = "\n\n".join(f"{m.get('role', 'unknown')}: {self._get_text_content(m)}" for m in messages)
        prompt = self.config.summarization_prompt.format(conversation=conversation)

        if self.llm_client:
            try:
                response = self.llm_client.create_completion([{"role": "user", "content": prompt}], max_tokens=self.config.summary_max_tokens)
                content = response.get("content") or response.get("choices", [{}])[0].get("message", {}).get("content", "")
                return remove_thinking_blocks(str(content))[0]
            except Exception as e:
                logger.error(f"LLM summarization failed: {e}")
        return self._fallback_summary(messages)

    def _get_text_content(self, msg: Dict) -> str:
        """Extract text content from message."""
        content = msg.get("content", "")
        if isinstance(content, list):
            return " ".join(item.get("text", "") for item in content if isinstance(item, dict))
        return str(content)

    def _fallback_summary(self, messages: List[Dict[str, Any]]) -> str:
        """Generate simple fallback summary without LLM."""
        if not messages:
            return "No previous conversation."
        topics = set()
        for msg in messages:
            for pattern in [r"about\s+(\w+)", r"regarding\s+(\w+)", r"topic:\s*(\w+)"]:
                topics.update(re.findall(pattern, str(msg.get("content", "")), re.IGNORECASE))
        topic_str = ", ".join(sorted(topics)) if topics else "various topics"
        roles = {}
        for msg in messages:
            r = msg.get("role", "unknown")
            roles[r] = roles.get(r, 0) + 1
        role_str = ", ".join(f"{v} {k}" for k, v in roles.items())
        return f"Previous discussion covered {topic_str}. Conversation included {role_str}."

    def _build_compacted_messages(self, critical: List[Dict], summary: Optional[str]) -> List[Dict]:
        """Build compacted message list."""
        result = [m for m in critical if m.get("role") == "system"]
        if summary:
            result.append({"role": "system", "content": f"Previous session summary: {summary}"})
        result.extend(m for m in critical if m.get("role") != "system")
        return result

    def start_background_monitoring(self, messages: List[Dict], llm_client: Optional[LLMClientProtocol] = None) -> None:
        """Start background summarization."""
        if not self.config.enable_background:
            return
        client = llm_client or self.llm_client
        if not client:
            logger.warning("No LLM client for background summarization")
            return
        if self._update_thread and self._update_thread.is_alive():
            return
        msgs = messages.copy()
        idx = len(messages)
        tokens = self.estimate_message_tokens(messages)
        self._update_thread = threading.Thread(target=self._background_summarization, args=(msgs, idx, tokens, client), daemon=True)
        self._update_thread.start()

    def _background_summarization(self, messages: List, idx: int, tokens: int, client) -> None:
        """Background thread for generating/updating session summary."""
        try:
            with self._lock:
                existing = self._session_summary
                last_idx = self._last_summarized_index

            if existing is None:
                summary = self._generate_summary(messages)
                logger.info("Initial session summary created")
            else:
                new_msgs = messages[last_idx:]
                prompt = f"Update this summary: {existing}\n\nNew messages:\n{self._format_for_summary(new_msgs)}"
                try:
                    response = client.create_completion([{"role": "user", "content": prompt}], max_tokens=self.config.summary_max_tokens)
                    summary = remove_thinking_blocks(response.get("content") or "")[0] or existing
                except:
                    summary = existing
                logger.info("Session summary updated")

            with self._lock:
                self._session_summary = summary
                self._last_summarized_index = idx
                self._tokens_at_last_update = tokens
        except Exception as e:
            logger.error(f"Background summarization error: {e}")

    def _format_for_summary(self, messages: List) -> str:
        return "\n\n".join(f"{m.get('role', 'unknown')}: {self._get_text_content(m)}" for m in messages)

    def get_summary(self) -> Optional[str]:
        with self._lock:
            return self._session_summary

    def reset(self) -> None:
        with self._lock:
            self._session_summary = None
            self._last_summarized_index = 0
            self._tokens_at_last_update = 0


# =============================================================================
# Memory Compaction for Persistent Storage
# =============================================================================


@dataclass
class MemoryCompactionResult:
    """Result of a memory compaction operation."""
    operation: str
    entries_processed: int
    entries_removed: int
    entries_modified: int
    bytes_saved: int
    duration_ms: float
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class MemoryCompactionConfig:
    """Configuration for memory compaction."""
    # Compression settings
    compress_old_entries: bool = True
    compression_age_days: int = 30
    compression_min_size_bytes: int = 1024

    # Deduplication settings
    remove_duplicates: bool = True
    duplicate_similarity_threshold: float = 0.95

    # Aggregation settings
    aggregate_similar: bool = True
    aggregation_threshold: float = 0.85
    max_cluster_size: int = 10

    # Cleanup settings
    cleanup_expired: bool = True
    expiration_days: int = 90
    min_importance_threshold: float = 0.1

    # Archive settings
    archive_old_data: bool = True
    archive_age_days: int = 60
    archive_path: Optional[Path] = None

    # Optimization settings
    optimize_storage: bool = True
    vacuum_after_compaction: bool = True


class MemoryCompactor:
    """
    Memory compactor for persistent memory storage optimization.

    Provides:
    - Compression of old memories using zlib
    - Duplicate detection and removal
    - Similar memory aggregation
    - Expired memory cleanup
    - Old data archival
    - Storage optimization (VACUUM)

    Usage:
        compactor = MemoryCompactor(db_path=Path("data/agent_memory/memory.db"))
        result = compactor.compress_old_entries()
        result = compactor.remove_duplicates()
        result = compactor.cleanup_expired()
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        config: Optional[MemoryCompactionConfig] = None,
    ):
        """
        Initialize the memory compactor.

        Args:
            db_path: Path to the SQLite memory database
            config: Compaction configuration
        """
        if db_path is None:
            db_path = Path(os.environ.get("AGENT_MEMORY_PATH", "data/agent_memory"))
            db_path = db_path / "memory.db"

        self.db_path = Path(db_path)
        self.config = config or MemoryCompactionConfig()

        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize archive path
        if self.config.archive_path is None:
            self.config.archive_path = self.db_path.parent / "archive"

        self.config.archive_path.mkdir(parents=True, exist_ok=True)

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _compute_content_hash(self, content: str) -> str:
        """Compute hash of content for duplicate detection."""
        return hashlib.sha256(content.encode()).hexdigest()

    def _compute_similarity(self, content1: str, content2: str) -> float:
        """Compute simple similarity between two content strings."""
        # Simple word-based similarity
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union)

    def compress_old_entries(self) -> MemoryCompactionResult:
        """
        Compress old memory entries using zlib.

        Returns:
            MemoryCompactionResult with compression statistics
        """
        start_time = time.perf_counter()
        entries_processed = 0
        entries_modified = 0
        bytes_saved = 0

        cutoff_date = datetime.utcnow() - timedelta(days=self.config.compression_age_days)
        cutoff_str = cutoff_date.isoformat()

        conn = self._get_connection()

        try:
            # Find entries to compress
            cursor = conn.execute(
                """
                SELECT id, key, value, namespace, LENGTH(value) as size
                FROM memories
                WHERE created_at < ?
                AND LENGTH(value) >= ?
                AND (metadata LIKE '%"compressed":false%' OR metadata LIKE '%"compressed"%' = 0)
                """,
                (cutoff_str, self.config.compression_min_size_bytes)
            )
            entries = cursor.fetchall()

            for row in entries:
                entries_processed += 1
                entry_id = row["id"]
                original_value = row["value"]
                original_size = len(original_value)

                try:
                    # Compress the value
                    compressed = zlib.compress(original_value.encode(), level=6)
                    compressed_str = compressed.decode('latin-1')  # Store as latin-1 string

                    # Update with compressed value
                    metadata = json.loads(row["metadata"]) if row["metadata"] else {}
                    metadata["compressed"] = True
                    metadata["original_size"] = original_size
                    metadata["compressed_size"] = len(compressed_str)

                    conn.execute(
                        """
                        UPDATE memories
                        SET value = ?, metadata = ?, updated_at = ?
                        WHERE id = ?
                        """,
                        (
                            compressed_str,
                            json.dumps(metadata),
                            datetime.utcnow().isoformat(),
                            entry_id
                        )
                    )

                    entries_modified += 1
                    bytes_saved += original_size - len(compressed_str)

                except Exception as e:
                    logger.warning(f"Failed to compress entry {entry_id}: {e}")

            conn.commit()

        finally:
            conn.close()

        duration_ms = (time.perf_counter() - start_time) * 1000

        return MemoryCompactionResult(
            operation="compress_old_entries",
            entries_processed=entries_processed,
            entries_removed=0,
            entries_modified=entries_modified,
            bytes_saved=bytes_saved,
            duration_ms=duration_ms,
            details={
                "cutoff_date": cutoff_str,
                "min_size": self.config.compression_min_size_bytes,
            }
        )

    def decompress_entry(self, entry_id: str) -> Optional[str]:
        """
        Decompress a single entry.

        Args:
            entry_id: ID of the entry to decompress

        Returns:
            Decompressed value or None if not found/not compressed
        """
        conn = self._get_connection()

        try:
            cursor = conn.execute(
                "SELECT value, metadata FROM memories WHERE id = ?",
                (entry_id,)
            )
            row = cursor.fetchone()

            if not row:
                return None

            metadata = json.loads(row["metadata"]) if row["metadata"] else {}

            if not metadata.get("compressed", False):
                return row["value"]

            # Decompress
            compressed_str = row["value"]
            decompressed = zlib.decompress(compressed_str.encode('latin-1')).decode()

            # Update metadata
            metadata["compressed"] = False
            del metadata["original_size"]
            del metadata["compressed_size"]

            conn.execute(
                "UPDATE memories SET value = ?, metadata = ?, updated_at = ? WHERE id = ?",
                (decompressed, json.dumps(metadata), datetime.utcnow().isoformat(), entry_id)
            )
            conn.commit()

            return decompressed

        finally:
            conn.close()

    def remove_duplicates(self) -> MemoryCompactionResult:
        """
        Remove duplicate memory entries based on content hash.

        Returns:
            MemoryCompactionResult with deduplication statistics
        """
        start_time = time.perf_counter()
        entries_processed = 0
        entries_removed = 0
        bytes_saved = 0

        conn = self._get_connection()

        try:
            # Get all entries grouped by content hash
            cursor = conn.execute(
                """
                SELECT id, value, namespace, key, created_at,
                       COUNT(*) as count,
                       MIN(id) as first_id
                FROM memories
                GROUP BY value
                HAVING count > 1
                """
            )
            duplicate_groups = cursor.fetchall()

            for group in duplicate_groups:
                entries_processed += group["count"]
                first_id = group["first_id"]

                # Keep the first entry, remove others
                cursor = conn.execute(
                    "SELECT id, LENGTH(value) as size FROM memories WHERE value = ? AND id != ?",
                    (group["value"], first_id)
                )
                duplicates = cursor.fetchall()

                for dup in duplicates:
                    conn.execute("DELETE FROM memories WHERE id = ?", (dup["id"],))
                    entries_removed += 1
                    bytes_saved += dup["size"]

            conn.commit()

        finally:
            conn.close()

        duration_ms = (time.perf_counter() - start_time) * 1000

        return MemoryCompactionResult(
            operation="remove_duplicates",
            entries_processed=entries_processed,
            entries_removed=entries_removed,
            entries_modified=0,
            bytes_saved=bytes_saved,
            duration_ms=duration_ms,
            details={"duplicate_groups_found": entries_processed - entries_removed}
        )

    def aggregate_similar(self) -> MemoryCompactionResult:
        """
        Aggregate similar memory entries into clusters.

        Returns:
            MemoryCompactionResult with aggregation statistics
        """
        start_time = time.perf_counter()
        entries_processed = 0
        entries_removed = 0
        bytes_saved = 0

        conn = self._get_connection()

        try:
            # Get all entries
            cursor = conn.execute(
                "SELECT id, value, namespace, key FROM memories ORDER BY created_at DESC"
            )
            all_entries = cursor.fetchall()

            processed_ids = set()
            clusters = []

            for i, entry in enumerate(all_entries):
                if entry["id"] in processed_ids:
                    continue

                cluster = [entry]
                processed_ids.add(entry["id"])

                # Find similar entries
                for j in range(i + 1, len(all_entries)):
                    if all_entries[j]["id"] in processed_ids:
                        continue

                    similarity = self._compute_similarity(
                        entry["value"],
                        all_entries[j]["value"]
                    )

                    if similarity >= self.config.aggregation_threshold:
                        cluster.append(all_entries[j])
                        processed_ids.add(all_entries[j]["id"])

                        if len(cluster) >= self.config.max_cluster_size:
                            break

                if len(cluster) > 1:
                    clusters.append(cluster)

            # Merge clusters
            for cluster in clusters:
                # Keep the most recent entry
                cluster.sort(key=lambda x: x["created_at"], reverse=True)
                primary = cluster[0]

                # Aggregate content
                aggregated_content = primary["value"]
                aggregated_metadata = {
                    "aggregated_from": len(cluster),
                    "cluster_ids": [e["id"] for e in cluster],
                    "original_keys": [e["key"] for e in cluster]
                }

                for dup in cluster[1:]:
                    conn.execute(
                        "DELETE FROM memories WHERE id = ?",
                        (dup["id"],)
                    )
                    entries_removed += 1
                    bytes_saved += len(dup["value"])

                # Update primary with aggregated metadata
                conn.execute(
                    "UPDATE memories SET metadata = ?, updated_at = ? WHERE id = ?",
                    (
                        json.dumps(aggregated_metadata),
                        datetime.utcnow().isoformat(),
                        primary["id"]
                    )
                )

                entries_processed += len(cluster)

            conn.commit()

        finally:
            conn.close()

        duration_ms = (time.perf_counter() - start_time) * 1000

        return MemoryCompactionResult(
            operation="aggregate_similar",
            entries_processed=entries_processed,
            entries_removed=entries_removed,
            entries_modified=len(clusters),
            bytes_saved=bytes_saved,
            duration_ms=duration_ms,
            details={"clusters_merged": len(clusters)}
        )

    def cleanup_expired(self) -> MemoryCompactionResult:
        """
        Remove expired memory entries.

        Returns:
            MemoryCompactionResult with cleanup statistics
        """
        start_time = time.perf_counter()
        entries_removed = 0
        bytes_saved = 0

        cutoff_date = datetime.utcnow() - timedelta(days=self.config.expiration_days)
        cutoff_str = cutoff_date.isoformat()

        conn = self._get_connection()

        try:
            # Get entries to delete
            cursor = conn.execute(
                """
                SELECT id, LENGTH(value) as size
                FROM memories
                WHERE created_at < ?
                AND importance < ?
                """,
                (cutoff_str, self.config.min_importance_threshold)
            )
            entries = cursor.fetchall()

            for entry in entries:
                conn.execute("DELETE FROM memories WHERE id = ?", (entry["id"],))
                entries_removed += 1
                bytes_saved += entry["size"]

            conn.commit()

        finally:
            conn.close()

        duration_ms = (time.perf_counter() - start_time) * 1000

        return MemoryCompactionResult(
            operation="cleanup_expired",
            entries_processed=entries_removed,
            entries_removed=entries_removed,
            entries_modified=0,
            bytes_saved=bytes_saved,
            duration_ms=duration_ms,
            details={
                "cutoff_date": cutoff_str,
                "min_importance": self.config.min_importance_threshold,
            }
        )

    def archive_old_data(self) -> MemoryCompactionResult:
        """
        Archive old memory entries to a separate file.

        Returns:
            MemoryCompactionResult with archival statistics
        """
        start_time = time.perf_counter()
        entries_processed = 0
        entries_removed = 0
        bytes_saved = 0

        cutoff_date = datetime.utcnow() - timedelta(days=self.config.archive_age_days)
        cutoff_str = cutoff_date.isoformat()

        archive_db = self.config.archive_path / f"archive_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.db"

        conn = self._get_connection()
        archive_conn = sqlite3.connect(str(archive_db))

        try:
            # Create archive schema
            archive_conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    namespace TEXT NOT NULL,
                    tags TEXT,
                    metadata TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    agent_id TEXT,
                    session_id TEXT,
                    archived_at TEXT NOT NULL
                )
            """)

            # Get entries to archive
            cursor = conn.execute(
                """
                SELECT * FROM memories
                WHERE created_at < ?
                """,
                (cutoff_str,)
            )
            entries = cursor.fetchall()

            for entry in entries:
                # Insert into archive
                archive_conn.execute(
                    """
                    INSERT INTO memories
                    (id, key, value, namespace, tags, metadata, created_at, updated_at,
                     agent_id, session_id, archived_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        entry["id"],
                        entry["key"],
                        entry["value"],
                        entry["namespace"],
                        entry["tags"],
                        entry["metadata"],
                        entry["created_at"],
                        entry["updated_at"],
                        entry["agent_id"],
                        entry["session_id"],
                        datetime.utcnow().isoformat()
                    )
                )

                # Delete from main database
                conn.execute("DELETE FROM memories WHERE id = ?", (entry["id"],))

                entries_processed += 1
                entries_removed += 1
                bytes_saved += len(entry["value"])

            archive_conn.commit()
            conn.commit()

        finally:
            conn.close()
            archive_conn.close()

        duration_ms = (time.perf_counter() - start_time) * 1000

        return MemoryCompactionResult(
            operation="archive_old_data",
            entries_processed=entries_processed,
            entries_removed=entries_removed,
            entries_modified=0,
            bytes_saved=bytes_saved,
            duration_ms=duration_ms,
            details={
                "cutoff_date": cutoff_str,
                "archive_file": str(archive_db),
            }
        )

    def optimize_storage(self) -> MemoryCompactionResult:
        """
        Optimize database storage using VACUUM.

        Returns:
            MemoryCompactionResult with optimization statistics
        """
        start_time = time.perf_counter()

        # Get size before
        db_size_before = self.db_path.stat().st_size if self.db_path.exists() else 0

        conn = self._get_connection()

        try:
            # Run VACUUM
            conn.execute("VACUUM")
            conn.commit()

        finally:
            conn.close()

        # Get size after
        db_size_after = self.db_path.stat().st_size if self.db_path.exists() else 0

        duration_ms = (time.perf_counter() - start_time) * 1000

        return MemoryCompactionResult(
            operation="optimize_storage",
            entries_processed=0,
            entries_removed=0,
            entries_modified=0,
            bytes_saved=db_size_before - db_size_after,
            duration_ms=duration_ms,
            details={
                "size_before": db_size_before,
                "size_after": db_size_after,
            }
        )

    def run_full_compaction(
        self,
        include_compression: bool = True,
        include_deduplication: bool = True,
        include_aggregation: bool = True,
        include_cleanup: bool = True,
        include_archive: bool = True,
        include_optimization: bool = True,
    ) -> Dict[str, MemoryCompactionResult]:
        """
        Run full memory compaction pipeline.

        Args:
            include_compression: Compress old entries
            include_deduplication: Remove duplicates
            include_aggregation: Aggregate similar entries
            include_cleanup: Clean up expired entries
            include_archive: Archive old entries
            include_optimization: Optimize storage

        Returns:
            Dictionary of results by operation
        """
        results = {}

        if include_compression and self.config.compress_old_entries:
            results["compression"] = self.compress_old_entries()

        if include_deduplication and self.config.remove_duplicates:
            results["deduplication"] = self.remove_duplicates()

        if include_aggregation and self.config.aggregate_similar:
            results["aggregation"] = self.aggregate_similar()

        if include_cleanup and self.config.cleanup_expired:
            results["cleanup"] = self.cleanup_expired()

        if include_archive and self.config.archive_old_data:
            results["archive"] = self.archive_old_data()

        if include_optimization and self.config.optimize_storage:
            results["optimization"] = self.optimize_storage()

        return results

    def get_compaction_stats(self) -> Dict[str, Any]:
        """
        Get current compaction-related statistics.

        Returns:
            Dictionary with compaction statistics
        """
        conn = self._get_connection()

        try:
            stats = {}

            # Total entries
            cursor = conn.execute("SELECT COUNT(*) FROM memories")
            stats["total_entries"] = cursor.fetchone()[0]

            # Compressed entries
            cursor = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE metadata LIKE '%\"compressed\":true%'"
            )
            stats["compressed_entries"] = cursor.fetchone()[0]

            # Aggregated entries
            cursor = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE metadata LIKE '%\"aggregated_from\"%'"
            )
            stats["aggregated_entries"] = cursor.fetchone()[0]

            # Database size
            stats["db_size_bytes"] = self.db_path.stat().st_size if self.db_path.exists() else 0

            # Old entries count (older than archive age)
            cutoff = datetime.utcnow() - timedelta(days=self.config.archive_age_days)
            cursor = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE created_at < ?",
                (cutoff.isoformat(),)
            )
            stats["old_entries"] = cursor.fetchone()[0]

            # Duplicate groups
            cursor = conn.execute(
                """
                SELECT COUNT(*) FROM (
                    SELECT value FROM memories GROUP BY value HAVING COUNT(*) > 1
                )
                """
            )
            stats["duplicate_groups"] = cursor.fetchone()[0]

            return stats

        finally:
            conn.close()


# Convenience function
def create_memory_compactor(
    db_path: Optional[Path] = None,
    **config_kwargs
) -> MemoryCompactor:
    """
    Create a MemoryCompactor with custom configuration.

    Args:
        db_path: Path to the SQLite memory database
        **config_kwargs: Configuration options

    Returns:
        Configured MemoryCompactor instance
    """
    config = MemoryCompactionConfig(**config_kwargs)
    return MemoryCompactor(db_path=db_path, config=config)
