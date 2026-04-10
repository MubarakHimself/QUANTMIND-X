"""
QuantMindLib V1 — Composer

Central composition engine for QuantMindLib.
Assembles a frozen BotSpec from an archetype spec + configuration.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.library.archetypes.archetype_spec import ArchetypeSpec
from src.library.archetypes.composition.result import CompositionResult
from src.library.archetypes.composition.resolver import RequirementResolver
from src.library.archetypes.composition.validator import CompositionValidator
from src.library.archetypes.composition.validation import ValidationResult
from src.library.archetypes.registry import ArchetypeRegistry
from src.library.core.domain.bot_spec import BotMutationProfile, BotSpec
from src.library.features.registry import FeatureRegistry


class Composer:
    """
    Central composition engine for QuantMindLib.

    Assembles a frozen BotSpec from an archetype spec + configuration.
    Validates composition, resolves dependencies, applies constraints,
    and attaches the mutation profile with locked/mutable surface enforcement.

    Usage:
        composer = Composer(archetype_registry, feature_registry)
        result = composer.compose(bot_id, archetype_id, config)
        # result.bot_spec -- the assembled BotSpec (frozen)
        # result.validation -- ValidationResult from CompositionValidator
    """

    def __init__(
        self,
        archetype_registry: ArchetypeRegistry,
        feature_registry: FeatureRegistry,
    ) -> None:
        """
        Initialize the Composer with registries.

        Args:
            archetype_registry: ArchetypeRegistry for archetype lookups.
            feature_registry: FeatureRegistry for feature lookups.
        """
        self.archetype_registry = archetype_registry
        self.feature_registry = feature_registry
        self.validator = CompositionValidator(
            archetype_registry,
            feature_registry,
            RequirementResolver(feature_registry),
        )

    def compose(
        self,
        bot_id: str,
        archetype_id: str,
        config: Dict[str, Any],
    ) -> CompositionResult:
        """
        Compose a BotSpec from an archetype + configuration.

        Args:
            bot_id: Unique identifier for this bot.
            archetype_id: Which archetype to base this bot on.
            config: Composition configuration dict:
                {
                    "symbol_scope": List[str],        # required
                    "sessions": List[str],            # required
                    "features": List[str],            # optional -- adds to optional_features
                    "confirmations": List[str],       # optional
                    "execution_profile": str,        # optional -- overrides archetype default
                    "mutation_profile": dict,          # optional -- overrides BotMutationProfile defaults
                }

        Returns:
            CompositionResult: {
                bot_spec: Optional[BotSpec],     # None if validation failed
                validation: ValidationResult,
                archetype: Optional[ArchetypeSpec],
                composition_notes: List[str],
            }
        """
        # 1. Get archetype spec
        archetype = self.archetype_registry.get(archetype_id)
        if not archetype:
            return CompositionResult(
                bot_spec=None,
                validation=ValidationResult(
                    valid=False,
                    errors=[f"ArchetypeNotFound: {archetype_id}"],
                ),
                archetype=None,
                composition_notes=[],
            )

        # 2. Build feature list: required + user-specified optional
        required = list(archetype.required_features)
        user_features = config.get("features", [])
        all_features = required + [f for f in user_features if f not in required]

        # 3. Build BotSpec fields
        symbol_scope = config.get("symbol_scope", [])
        sessions = config.get("sessions", [])
        confirmations = config.get("confirmations", list(archetype.confirmations))
        execution_profile = config.get(
            "execution_profile", archetype.execution_profile
        )

        # 4. Build mutation profile from archetype defaults
        mutation_config = config.get("mutation_profile", {})
        mutation = BotMutationProfile(
            bot_id=bot_id,
            allowed_mutation_areas=mutation_config.get(
                "allowed_mutation_areas", list(archetype.default_allowed_areas)
            ),
            locked_components=mutation_config.get(
                "locked_components", list(archetype.default_locked_components)
            ),
            lineage=mutation_config.get("lineage", []),
            parent_variant_id=mutation_config.get("parent_variant_id"),
            generation=mutation_config.get("generation", 0),
            unsafe_areas=mutation_config.get("unsafe_areas", []),
        )

        # 5. Build preliminary BotSpec
        prelim_spec = BotSpec(
            id=bot_id,
            archetype=archetype_id,
            symbol_scope=symbol_scope,
            sessions=sessions,
            features=all_features,
            confirmations=confirmations,
            execution_profile=execution_profile,
            mutation=mutation,
        )

        # 6. Validate
        validation = self.validator.validate_bot_spec(prelim_spec)

        # 7. Generate composition notes
        notes: List[str] = []
        if not validation.is_clean:
            notes.append(f"Composition warnings: {len(validation.warnings)}")
        for warning in validation.warnings:
            notes.append(f"  - {warning}")

        return CompositionResult(
            bot_spec=prelim_spec if validation.valid else None,
            validation=validation,
            archetype=archetype,
            composition_notes=notes,
        )

    def from_archetype_template(
        self,
        bot_id: str,
        archetype_id: str,
        symbol_scope: List[str],
        sessions: List[str],
        optional_features: Optional[List[str]] = None,
    ) -> CompositionResult:
        """
        Convenience method: compose a BotSpec from an archetype template.

        Uses archetype defaults for execution_profile, confirmations, constraints.

        Args:
            bot_id: Unique identifier for this bot.
            archetype_id: Which archetype to base this bot on.
            symbol_scope: List of symbols this bot operates on.
            sessions: List of trading session tags.
            optional_features: Optional list of feature_ids to add.

        Returns:
            CompositionResult from compose().
        """
        config: Dict[str, Any] = {
            "symbol_scope": symbol_scope,
            "sessions": sessions,
            "features": optional_features or [],
        }
        return self.compose(bot_id, archetype_id, config)


__all__ = ["Composer"]
