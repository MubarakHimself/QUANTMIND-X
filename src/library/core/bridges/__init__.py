"""
QuantMindLib V1 — Bridge Definitions
"""
from src.library.core.bridges.registry_journal_bridges import RegistryBridge, JournalEntry, JournalBridge
from src.library.core.bridges.sentinel_dpr_bridges import SentinelBridge, DPRScore, DPRBridge

__all__ = [
    "SentinelBridge", "DPRScore", "DPRBridge",
    "RegistryBridge", "JournalEntry", "JournalBridge",
]
