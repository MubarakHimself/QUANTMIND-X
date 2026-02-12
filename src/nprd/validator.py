"""
Timeline JSON Validator.

This module provides validation for Timeline JSON output from NPRD processing.

Validates:
- JSON structure validity
- Required fields presence
- Timestamp ordering
- Clip format compliance
"""

import json
import re
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from .exceptions import ValidationError


@dataclass
class ValidationResult:
    """Result of timeline validation."""
    valid: bool
    errors: List[str]
    warnings: List[str]
    
    def __bool__(self) -> bool:
        return self.valid
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
        }


class TimelineValidator:
    """
    Validates Timeline JSON structure and content.
    
    Validation Rules:
    1. JSON must be valid and parseable
    2. Required fields: meta.video_url, meta.title, meta.duration_seconds, 
       meta.processed_at, meta.model_provider, timeline
    3. Timestamps must be in chronological order
    4. Each clip must have required fields: clip_id, timestamp_start, 
       timestamp_end, transcript, visual_description, frame_path
    5. Timestamp format must be HH:MM:SS
    """
    
    # Required fields in meta
    REQUIRED_META_FIELDS = {
        "video_url": str,
        "title": str,
        "duration_seconds": int,
        "processed_at": str,
        "model_provider": str,
    }
    
    # Optional fields in meta
    OPTIONAL_META_FIELDS = {
        "version": str,
    }
    
    # Required fields in each clip
    REQUIRED_CLIP_FIELDS = {
        "clip_id": int,
        "timestamp_start": str,
        "timestamp_end": str,
        "transcript": str,
        "visual_description": str,
        "frame_path": str,
    }
    
    # Timestamp format regex (HH:MM:SS)
    TIMESTAMP_PATTERN = re.compile(r"^(\d{2}):(\d{2}):(\d{2})$")
    
    def __init__(self, strict: bool = False):
        """
        Initialize validator.
        
        Args:
            strict: If True, treat warnings as errors
        """
        self.strict = strict
    
    def validate(self, data: Any) -> ValidationResult:
        """
        Validate timeline data.
        
        Args:
            data: Timeline data (can be string, dict, or already parsed)
            
        Returns:
            ValidationResult with errors and warnings
        """
        errors = []
        warnings = []
        
        # Parse JSON if string
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError as e:
                return ValidationResult(
                    valid=False,
                    errors=[f"Invalid JSON: {e}"],
                    warnings=[],
                )
        
        # Check top-level structure
        if not isinstance(data, dict):
            return ValidationResult(
                valid=False,
                errors=["Timeline must be a JSON object"],
                warnings=[],
            )
        
        # Validate meta section
        meta_errors, meta_warnings = self._validate_meta(data.get("meta"))
        errors.extend(meta_errors)
        warnings.extend(meta_warnings)
        
        # Validate timeline section
        timeline_errors, timeline_warnings = self._validate_timeline(data.get("timeline"))
        errors.extend(timeline_errors)
        warnings.extend(timeline_warnings)
        
        # In strict mode, warnings become errors
        if self.strict:
            errors.extend(warnings)
            warnings = []
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )
    
    def validate_json(self, json_string: str) -> ValidationResult:
        """
        Validate a JSON string.
        
        Args:
            json_string: JSON string to validate
            
        Returns:
            ValidationResult with errors and warnings
        """
        return self.validate(json_string)
    
    def validate_dict(self, data: Dict[str, Any]) -> ValidationResult:
        """
        Validate a dictionary.
        
        Args:
            data: Dictionary to validate
            
        Returns:
            ValidationResult with errors and warnings
        """
        return self.validate(data)
    
    def _validate_meta(self, meta: Any) -> Tuple[List[str], List[str]]:
        """
        Validate the meta section of timeline.
        
        Args:
            meta: Meta section data
            
        Returns:
            Tuple of (errors, warnings)
        """
        errors = []
        warnings = []
        
        if meta is None:
            errors.append("Missing 'meta' section")
            return errors, warnings
        
        if not isinstance(meta, dict):
            errors.append("'meta' must be a JSON object")
            return errors, warnings
        
        # Check required fields
        for field, expected_type in self.REQUIRED_META_FIELDS.items():
            if field not in meta:
                errors.append(f"Missing required field 'meta.{field}'")
            elif not isinstance(meta[field], expected_type):
                errors.append(
                    f"Field 'meta.{field}' must be {expected_type.__name__}, "
                    f"got {type(meta[field]).__name__}"
                )
        
        # Validate processed_at format (ISO 8601)
        if "processed_at" in meta and isinstance(meta["processed_at"], str):
            try:
                datetime.fromisoformat(meta["processed_at"].replace("Z", "+00:00"))
            except ValueError:
                warnings.append(
                    f"'meta.processed_at' is not a valid ISO 8601 timestamp: {meta['processed_at']}"
                )
        
        # Validate duration
        if "duration_seconds" in meta and isinstance(meta["duration_seconds"], int):
            if meta["duration_seconds"] < 0:
                errors.append("'meta.duration_seconds' cannot be negative")
            elif meta["duration_seconds"] == 0:
                warnings.append("'meta.duration_seconds' is 0")
        
        # Validate model_provider
        if "model_provider" in meta and isinstance(meta["model_provider"], str):
            valid_providers = ["gemini", "qwen", "error"]
            if meta["model_provider"] not in valid_providers:
                warnings.append(
                    f"'meta.model_provider' value '{meta['model_provider']}' is not recognized. "
                    f"Expected one of: {valid_providers}"
                )
        
        # Check for unknown fields
        known_fields = set(self.REQUIRED_META_FIELDS.keys()) | set(self.OPTIONAL_META_FIELDS.keys())
        for field in meta.keys():
            if field not in known_fields:
                warnings.append(f"Unknown field in meta: '{field}'")
        
        return errors, warnings
    
    def _validate_timeline(self, timeline: Any) -> Tuple[List[str], List[str]]:
        """
        Validate the timeline section.
        
        Args:
            timeline: Timeline section data
            
        Returns:
            Tuple of (errors, warnings)
        """
        errors = []
        warnings = []
        
        if timeline is None:
            errors.append("Missing 'timeline' section")
            return errors, warnings
        
        if not isinstance(timeline, list):
            errors.append("'timeline' must be a JSON array")
            return errors, warnings
        
        if len(timeline) == 0:
            warnings.append("'timeline' is empty")
            return errors, warnings
        
        # Track timestamps for ordering validation
        previous_end_seconds = -1
        seen_clip_ids = set()
        
        for i, clip in enumerate(timeline):
            clip_errors, clip_warnings = self._validate_clip(clip, i)
            errors.extend(clip_errors)
            warnings.extend(clip_warnings)
            
            # Check clip_id uniqueness
            if isinstance(clip, dict) and "clip_id" in clip:
                clip_id = clip["clip_id"]
                if clip_id in seen_clip_ids:
                    errors.append(f"Duplicate clip_id: {clip_id}")
                seen_clip_ids.add(clip_id)
            
            # Check timestamp ordering
            if isinstance(clip, dict) and "timestamp_start" in clip:
                start_seconds = self._timestamp_to_seconds(clip["timestamp_start"])
                if start_seconds is not None:
                    if start_seconds < previous_end_seconds:
                        errors.append(
                            f"Clip {i} timestamp_start ({clip['timestamp_start']}) "
                            f"is earlier than previous clip end"
                        )
                    
                    if "timestamp_end" in clip:
                        end_seconds = self._timestamp_to_seconds(clip["timestamp_end"])
                        if end_seconds is not None:
                            if end_seconds < start_seconds:
                                errors.append(
                                    f"Clip {i} timestamp_end ({clip['timestamp_end']}) "
                                    f"is earlier than timestamp_start ({clip['timestamp_start']})"
                                )
                            previous_end_seconds = end_seconds
        
        return errors, warnings
    
    def _validate_clip(self, clip: Any, index: int) -> Tuple[List[str], List[str]]:
        """
        Validate a single clip.
        
        Args:
            clip: Clip data
            index: Clip index in timeline
            
        Returns:
            Tuple of (errors, warnings)
        """
        errors = []
        warnings = []
        
        if not isinstance(clip, dict):
            errors.append(f"Clip {index} must be a JSON object")
            return errors, warnings
        
        # Check required fields
        for field, expected_type in self.REQUIRED_CLIP_FIELDS.items():
            if field not in clip:
                errors.append(f"Clip {index}: Missing required field '{field}'")
            elif not isinstance(clip[field], expected_type):
                errors.append(
                    f"Clip {index}: Field '{field}' must be {expected_type.__name__}, "
                    f"got {type(clip[field]).__name__}"
                )
        
        # Validate timestamp format
        for field in ["timestamp_start", "timestamp_end"]:
            if field in clip and isinstance(clip[field], str):
                if not self.TIMESTAMP_PATTERN.match(clip[field]):
                    errors.append(
                        f"Clip {index}: '{field}' must be in HH:MM:SS format, "
                        f"got '{clip[field]}'"
                    )
        
        # Validate clip_id is non-negative
        if "clip_id" in clip and isinstance(clip["clip_id"], int):
            if clip["clip_id"] < 0:
                errors.append(f"Clip {index}: 'clip_id' cannot be negative")
        
        # Check for empty content
        if "transcript" in clip and isinstance(clip["transcript"], str):
            if len(clip["transcript"].strip()) == 0:
                warnings.append(f"Clip {index}: 'transcript' is empty")
        
        if "visual_description" in clip and isinstance(clip["visual_description"], str):
            if len(clip["visual_description"].strip()) == 0:
                warnings.append(f"Clip {index}: 'visual_description' is empty")
        
        return errors, warnings
    
    def _timestamp_to_seconds(self, timestamp: str) -> Optional[int]:
        """
        Convert timestamp string to seconds.
        
        Args:
            timestamp: Timestamp in HH:MM:SS format
            
        Returns:
            Total seconds, or None if invalid format
        """
        match = self.TIMESTAMP_PATTERN.match(timestamp)
        if not match:
            return None
        
        hours, minutes, seconds = int(match.group(1)), int(match.group(2)), int(match.group(3))
        return hours * 3600 + minutes * 60 + seconds


def validate_timeline(data: Any, strict: bool = False) -> ValidationResult:
    """
    Validate timeline data.
    
    Convenience function that creates a validator and validates data.
    
    Args:
        data: Timeline data (string, dict, or TimelineOutput)
        strict: If True, treat warnings as errors
        
    Returns:
        ValidationResult with errors and warnings
    """
    validator = TimelineValidator(strict=strict)
    return validator.validate(data)


def is_valid_timeline(data: Any) -> bool:
    """
    Check if timeline data is valid.
    
    Args:
        data: Timeline data to check
        
    Returns:
        True if valid, False otherwise
    """
    result = validate_timeline(data)
    return result.valid


def validate_timeline_strict(data: Any) -> None:
    """
    Validate timeline data strictly.
    
    Raises ValidationError if invalid (including warnings).
    
    Args:
        data: Timeline data to validate
        
    Raises:
        ValidationError: If validation fails
    """
    result = validate_timeline(data, strict=True)
    if not result.valid:
        raise ValidationError(f"Timeline validation failed: {'; '.join(result.errors)}")
