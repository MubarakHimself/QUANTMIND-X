"""
QuantMindLib V1 — MutationEngine

Constrained mutation engine for generating bot variants from an existing BotSpec.
Respects locked/mutable surface boundaries from BotMutationProfile.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.library.archetypes.composer import Composer
from src.library.archetypes.registry import ArchetypeRegistry
from src.library.core.domain.bot_spec import BotMutationProfile, BotSpec


class MutationResult(BaseModel):
    """Result of a mutation attempt."""

    success: bool
    mutated_spec: Optional[BotSpec] = Field(default=None)
    errors: List[str] = Field(default_factory=list)
    mutation_log: List[str] = Field(default_factory=list)


class MutationEngine:
    """
    Constrained mutation engine for generating bot variants from an existing BotSpec.

    Respects locked/mutable surface boundaries from BotMutationProfile.
    Validates that mutations only touch allowed_areas.
    Tracks lineage for the variant chain.
    """

    def __init__(
        self,
        composer: Composer,
        archetype_registry: ArchetypeRegistry,
    ) -> None:
        """
        Initialize the MutationEngine.

        Args:
            composer: Composer instance for validation of mutated specs.
            archetype_registry: ArchetypeRegistry for archetype lookups.
        """
        self.composer = composer
        self.archetype_registry = archetype_registry

    def mutate(
        self,
        bot_spec: BotSpec,
        mutations: Dict[str, Any],
    ) -> MutationResult:
        """
        Apply mutations to an existing BotSpec.

        Args:
            bot_spec: The source BotSpec to mutate.
            mutations: Dict of mutation specifications:
                {
                    "parameters.stop_loss_pips": 12.0,
                    "features.optional": ["indicators/macd"],
                    "constraints.spread_filter": 1.0,
                    ...
                }

        Returns:
            MutationResult: {
                success: bool,
                mutated_spec: Optional[BotSpec],
                errors: List[str],
                mutation_log: List[str],
            }
        """
        if bot_spec.mutation is None:
            return MutationResult(
                success=False,
                mutated_spec=None,
                errors=["BotSpec has no mutation profile -- cannot mutate"],
                mutation_log=[],
            )

        errors: List[str] = []
        mutation_log: List[str] = []
        allowed = set(bot_spec.mutation.allowed_mutation_areas)
        locked = set(bot_spec.mutation.locked_components)

        # Validate each mutation key
        for key, value in mutations.items():
            if key in locked:
                errors.append(
                    f"MutationRejected: {key} is locked (locked_components)"
                )
                continue
            if key in allowed:
                continue

            # Check if it's a nested key (e.g. "parameters.stop_loss_pips")
            # Allow if the parent area is in allowed
            area_prefix = key.split(".")[0]
            if area_prefix not in allowed and key not in allowed:
                errors.append(
                    f"MutationRejected: {key} not in allowed_mutation_areas"
                )

        if errors:
            return MutationResult(
                success=False,
                mutated_spec=None,
                errors=errors,
                mutation_log=mutation_log,
            )

        # Since BotSpec is frozen, we create a new mutable copy for mutation.
        # Library scope: structural mutations only (adding/removing features,
        # sessions, confirmations). Parameter mutations (stop_loss_pips, etc.)
        # are a runtime concern handled by the execution layer.
        new_features = list(bot_spec.features)
        new_confirmations = list(bot_spec.confirmations)

        for key, value in mutations.items():
            area = key.split(".")[0]
            if area == "features":
                sub_key = key.split(".")[1] if "." in key else None
                if sub_key == "optional" and isinstance(value, list):
                    for f in value:
                        if f not in new_features:
                            new_features.append(f)
                        mutation_log.append(f"Added optional feature: {f}")
            elif area == "confirmations":
                if isinstance(value, list):
                    for c in value:
                        if c not in new_confirmations:
                            new_confirmations.append(c)
                        mutation_log.append(f"Added confirmation: {c}")

        # Build lineage chain
        lineage = list(bot_spec.mutation.lineage)
        if bot_spec.mutation.parent_variant_id:
            lineage.append(bot_spec.mutation.parent_variant_id)

        new_gen = bot_spec.mutation.generation + 1
        new_id = f"{bot_spec.id}_v{new_gen}"

        # Create mutated spec
        mutated_mutation = BotMutationProfile(
            bot_id=new_id,
            allowed_mutation_areas=list(bot_spec.mutation.allowed_mutation_areas),
            locked_components=list(bot_spec.mutation.locked_components),
            lineage=lineage,
            parent_variant_id=bot_spec.id,
            generation=new_gen,
            unsafe_areas=list(bot_spec.mutation.unsafe_areas),
        )

        mutated_spec = BotSpec(
            id=new_id,
            archetype=bot_spec.archetype,
            symbol_scope=list(bot_spec.symbol_scope),
            sessions=list(bot_spec.sessions),
            features=new_features,
            confirmations=new_confirmations,
            execution_profile=bot_spec.execution_profile,
            capability_spec=bot_spec.capability_spec,
            runtime=bot_spec.runtime,
            evaluation=bot_spec.evaluation,
            mutation=mutated_mutation,
        )

        # Validate the mutated spec
        validation = self.composer.validator.validate_bot_spec(mutated_spec)
        if not validation.valid:
            return MutationResult(
                success=False,
                mutated_spec=None,
                errors=list(validation.errors),
                mutation_log=mutation_log,
            )

        return MutationResult(
            success=True,
            mutated_spec=mutated_spec,
            errors=[],
            mutation_log=mutation_log,
        )

    def can_mutate(self, bot_spec: BotSpec, mutation_key: str) -> bool:
        """
        Check whether a specific mutation key is allowed for a given BotSpec.

        Returns True if the key is in allowed_areas and not in locked_components.

        Args:
            bot_spec: The BotSpec to check against.
            mutation_key: The mutation key to validate (e.g. "features.optional").

        Returns:
            True if the mutation key is allowed, False otherwise.
        """
        if bot_spec.mutation is None:
            return False
        allowed = set(bot_spec.mutation.allowed_mutation_areas)
        locked = set(bot_spec.mutation.locked_components)

        if mutation_key in locked:
            return False
        if mutation_key in allowed:
            return True

        # Check prefix (e.g. "parameters.stop_loss_pips" -> check "parameters")
        area_prefix = mutation_key.split(".")[0]
        if area_prefix in locked:
            return False
        if area_prefix in allowed:
            return True

        return False


__all__ = ["MutationEngine", "MutationResult"]
