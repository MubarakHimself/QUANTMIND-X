"""
QuantMindLib V1 — RequirementResolver
"""
from __future__ import annotations

from typing import List, Set

from src.library.archetypes.composition.validation import ValidationResult
from src.library.features.registry import FeatureRegistry


class RequirementResolver:
    """
    Resolves feature dependencies for a bot composition.

    Validates that all required features are available in the FeatureRegistry.
    Identifies missing dependencies and suggests resolution paths.
    """

    def __init__(self, feature_registry: FeatureRegistry) -> None:
        """
        Initialize the resolver with a FeatureRegistry instance.

        Args:
            feature_registry: The FeatureRegistry to query against.
        """
        self.feature_registry = feature_registry

    def resolve(self, feature_ids: List[str]) -> ValidationResult:
        """
        Resolve all dependencies for a list of feature_ids.

        Checks each feature exists in the FeatureRegistry.
        For each feature's capability spec, verifies required data sources
        are satisfiable from the set of requested features.

        Args:
            feature_ids: List of feature_ids to resolve.

        Returns:
            ValidationResult with errors (missing features) and
            warnings/suggestions (dependency opportunities).
        """
        errors = []
        warnings = []
        suggestions = []

        requested = set(feature_ids)

        for fid in feature_ids:
            if not self.feature_registry.get(fid):
                errors.append(f"FeatureNotFound: {fid}")
                continue

            # Check capability spec for dependency info
            cap = self.feature_registry.get_capability(fid)
            if cap:
                # Verify required inputs are satisfied (either in requested
                # features or assumed available market data)
                for req in cap.requires:
                    # If the required input is NOT provided by any requested
                    # feature, it must be a market data input (not a feature).
                    # We check that at least some feature provides it.
                    provided_by_requested = any(
                        f in requested
                        for f in feature_ids
                        if self.feature_registry.get_capability(f)
                        and req in self.feature_registry.get_capability(f).provides
                    )
                    # If not provided by any requested feature, check if it's
                    # a known market data input (close_prices, volume, etc.)
                    # These are not feature dependencies -- they come from feeds.
                    # We warn only if the input is NOT in standard market data set.
                    standard_market_inputs = {
                        "close_prices", "open_prices", "high_prices", "low_prices",
                        "volume", "tick_volume", "spread", "bid", "ask",
                    }
                    if not provided_by_requested and req not in standard_market_inputs:
                        warnings.append(
                            f"InputNotSourced: {fid} requires '{req}' which is "
                            f"not provided by any requested feature"
                        )

        # Suggest optional features whose dependencies are already satisfied
        all_features = set(self.feature_registry.list_all())
        for fid in all_features:
            if fid in requested:
                continue

            cap = self.feature_registry.get_capability(fid)
            if cap and set(cap.requires).issubset(requested):
                # This feature's dependencies are already satisfied -- suggest it
                suggestions.append(
                    f"AvailableFeature: '{fid}' is not in composition but its "
                    f"dependencies are satisfied"
                )

        valid = len(errors) == 0
        return ValidationResult(valid=valid, errors=errors, warnings=warnings, suggestions=suggestions)

    def check_missing_inputs(
        self,
        feature_ids: List[str],
        available_inputs: Set[str],
    ) -> List[str]:
        """
        Check which required inputs are NOT available from available_inputs.

        Args:
            feature_ids: List of feature_ids to check.
            available_inputs: Set of input keys that are currently available.

        Returns:
            List of formatted strings describing each missing input
            (e.g. "indicators/rsi_14 requires close_prices").
        """
        missing = []
        for fid in feature_ids:
            feature = self.feature_registry.get(fid)
            if feature:
                for inp in feature.required_inputs:
                    if inp not in available_inputs:
                        missing.append(f"{fid} requires {inp}")
        return missing


__all__ = ["RequirementResolver"]