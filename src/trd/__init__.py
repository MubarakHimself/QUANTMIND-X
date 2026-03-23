"""
TRD (Trading Strategy Document) Module

Handles parsing, validation, and processing of trading strategy documents.
"""
from src.trd.schema import TRDDocument, TRDParameter, PositionSizing
from src.trd.parser import TRDParser
from src.trd.validator import TRDValidator, Ambiguity, ValidationError
from src.trd.generator import TRDGenerator, create_trd_from_hypothesis
from src.trd.storage import TRDStorage, get_trd_storage, save_trd, load_trd, list_trds

__all__ = [
    "TRDDocument",
    "TRDParameter",
    "PositionSizing",
    "TRDParser",
    "TRDValidator",
    "Ambiguity",
    "ValidationError",
    "TRDGenerator",
    "create_trd_from_hypothesis",
    "TRDStorage",
    "get_trd_storage",
    "save_trd",
    "load_trd",
    "list_trds",
]
