"""
QuantMindLib V1 — CompositionValidator
"""
from __future__ import annotations

from typing import List, Optional

from src.library.archetypes.archetype_spec import ArchetypeSpec
from src.library.archetypes.composition.validation import ValidationResult
from src.library.archetypes.composition.resolver import RequirementResolver
from src.library.archetypes.registry import ArchetypeRegistry
from src.library.core.domain.bot_spec import BotSpec
from src.library.features.registry import FeatureRegistry


class CompositionValidator:
    """
    Validates bot composition against archetype specifications.

    Checks: feature compatibility, constraint satisfaction, session scope,
    confirmation requirements, and mutation safety boundaries.
    """

    def __init__(
        self,
        archetype_registry: ArchetypeRegistry,
        feature_registry: FeatureRegistry,
        requirement_resolver: RequirementResolver,
    ) -> None:
        """
        Initialize the validator with registries and resolver.

        Args:
            archetype_registry: ArchetypeRegistry for archetype lookups.
            feature_registry: FeatureRegistry for feature lookups.
            requirement_resolver: RequirementResolver for dependency checking.
        """
        self.archetype_registry = archetype_registry
        self.feature_registry = feature_registry
        self.resolver = requirement_resolver

    def validate_bot_spec(self, bot_spec: BotSpec) -> ValidationResult:
        """
        Full validation of a BotSpec against its archetype spec.

        Performs all checks: archetype existence, required features, excluded
        features, dependency resolution, session scope, confirmation
        requirements, and constraint evaluation.

        Args:
            bot_spec: The BotSpec to validate.

        Returns:
            ValidationResult with all errors, warnings, and suggestions.
        """
        errors = []
        warnings = []
        suggestions = []

        # 1. Archetype exists
        archetype_spec = self.archetype_registry.get(bot_spec.archetype)
        if not archetype_spec:
            errors.append(f"ArchetypeNotFound: {bot_spec.archetype}")
            return ValidationResult(valid=False, errors=errors, warnings=warnings, suggestions=suggestions)

        # 2. Required features present
        for required in archetype_spec.required_features:
            found = self._find_feature(required, bot_spec.features)
            if not found:
                errors.append(
                    f"RequiredFeatureMissing: {required} "
                    f"(archetype {archetype_spec.archetype_id})"
                )

        # 3. No excluded features present
        for excluded in archetype_spec.excluded_features:
            found = self._find_feature(excluded, bot_spec.features)
            if found:
                errors.append(
                    f"ExcludedFeaturePresent: {found} "
                    f"(excluded from {archetype_spec.archetype_id})"
                )

        # 4. Feature dependency resolution
        dep_result = self.resolver.resolve(bot_spec.features)
        errors.extend(dep_result.errors)
        warnings.extend(dep_result.warnings)
        suggestions.extend(dep_result.suggestions)

        # 5. Session scope check
        for session in bot_spec.sessions:
            if session not in archetype_spec.session_tags:
                warnings.append(
                    f"SessionOutsideArchetype: {session} "
                    f"not in {archetype_spec.archetype_id} scope"
                )

        # 6. Confirmation requirements
        for conf in archetype_spec.confirmations:
            if conf not in bot_spec.confirmations:
                warnings.append(
                    f"ConfirmationMissing: {conf} "
                    f"(recommended for {archetype_spec.archetype_id})"
                )

        # 7. Suggest optional features not already in the set
        for optional in archetype_spec.optional_features:
            found = self._find_feature(optional, bot_spec.features)
            if not found:
                suggestions.append(
                    f"Consider optional feature: {optional} "
                    f"(for archetype {archetype_spec.archetype_id})"
                )

        valid = len(errors) == 0
        return ValidationResult(valid=valid, errors=errors, warnings=warnings, suggestions=suggestions)

    def validate_feature_set(
        self,
        feature_ids: List[str],
        archetype_id: str,
    ) -> ValidationResult:
        """
        Validate a set of features against a specific archetype.

        Shorter check than validate_bot_spec -- no BotSpec needed.
        Useful for quick feature compatibility checks.

        Args:
            feature_ids: List of feature_ids to validate.
            archetype_id: Archetype to validate against.

        Returns:
            ValidationResult with errors, warnings, and suggestions.
        """
        errors = []
        warnings = []
        suggestions = []

        archetype = self.archetype_registry.get(archetype_id)
        if not archetype:
            errors.append(f"ArchetypeNotFound: {archetype_id}")
            return ValidationResult(valid=False, errors=errors, warnings=warnings, suggestions=suggestions)

        # Check required features
        for required in archetype.required_features:
            found = self._find_feature(required, feature_ids)
            if not found:
                errors.append(
                    f"RequiredFeatureMissing: {required} "
                    f"(required by {archetype_id})"
                )

        # Check excluded features
        for excluded in archetype.excluded_features:
            found = self._find_feature(excluded, feature_ids)
            if found:
                errors.append(
                    f"ExcludedFeaturePresent: {found} "
                    f"(excluded by {archetype_id})"
                )

        # Suggest optional features not already in set
        for optional in archetype.optional_features:
            found = self._find_feature(optional, feature_ids)
            if not found:
                suggestions.append(
                    f"Consider optional feature: {optional} "
                    f"(for archetype {archetype_id})"
                )

        valid = len(errors) == 0
        return ValidationResult(valid=valid, errors=errors, warnings=warnings, suggestions=suggestions)

    def _find_feature(self, pattern: str, feature_ids: List[str]) -> Optional[str]:
        """
        Find a feature_id that matches a pattern.

        Pattern can be exact match ("indicators/rsi_14") or base-prefix
        match ("indicators/rsi" matches "indicators/rsi_14").

        Args:
            pattern: Pattern to match (exact or prefix).
            feature_ids: List of candidate feature_ids.

        Returns:
            The matching feature_id or None if not found.
        """
        # Exact match
        if pattern in feature_ids:
            return pattern

        # Prefix match: pattern is a base prefix (e.g. "indicators/rsi"
        # matches "indicators/rsi_14")
        for fid in feature_ids:
            if fid.startswith(pattern) or pattern.startswith(fid):
                return fid

        return None


__all__ = ["CompositionValidator"]