"""Deterministic orchestration engine for CEBA Core Constraint Processing."""

from __future__ import annotations

from dataclasses import dataclass, field

from ceba.core.constraints.errors import (
    ConstraintEvaluationError,
    ConstraintRegistryError,
    EmptyConstraintSetError,
    build_constraint_error_context,
)
from ceba.core.constraints.evaluator import (
    ConstraintEvaluatorPolicy,
    evaluate_constraints,
)
from ceba.core.constraints.feature_adapter import ConstraintFeatureSet
from ceba.core.constraints.registry import (
    ConstraintRegistry,
    find_active_constraints,
)
from ceba.core.constraints.schema import (
    ConstraintDefinition,
    ConstraintEvaluation,
    ConstraintEvaluationSet,
    ConstraintScope,
    ConstraintStatus,
    ensure_metadata_dict,
    ensure_non_empty_string,
    new_constraint_evaluation_set_id,
    utc_now,
)


@dataclass(frozen=True, slots=True)
class ConstraintEnginePolicy:
    """Stable engine-level policy."""

    evaluate_inactive: bool = False
    fail_on_empty_constraints: bool = False
    evaluator_policy: ConstraintEvaluatorPolicy = field(default_factory=ConstraintEvaluatorPolicy)

    def __post_init__(self) -> None:
        if not isinstance(self.evaluate_inactive, bool):
            raise TypeError("evaluate_inactive must be a bool")
        if not isinstance(self.fail_on_empty_constraints, bool):
            raise TypeError("fail_on_empty_constraints must be a bool")
        if not isinstance(self.evaluator_policy, ConstraintEvaluatorPolicy):
            raise TypeError("evaluator_policy must be ConstraintEvaluatorPolicy")


@dataclass(frozen=True, slots=True)
class ConstraintEngineResult:
    """Result of one engine execution."""

    evaluation_set: ConstraintEvaluationSet
    selected_constraint_count: int
    feature_set_count: int
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.evaluation_set, ConstraintEvaluationSet):
            raise TypeError("evaluation_set must be ConstraintEvaluationSet")
        _ensure_non_negative_int(self.selected_constraint_count, "selected_constraint_count")
        _ensure_non_negative_int(self.feature_set_count, "feature_set_count")
        ensure_metadata_dict(self.metadata)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, object]:
        """Return a shallow serializable representation."""

        return {
            "evaluation_set_id": str(self.evaluation_set.set_id),
            "observation_id": self.evaluation_set.observation_id,
            "evaluation_count": len(self.evaluation_set.evaluations),
            "selected_constraint_count": self.selected_constraint_count,
            "feature_set_count": self.feature_set_count,
            "metadata": dict(self.metadata),
        }


def select_constraints_for_feature_set(
    *,
    registry: ConstraintRegistry,
    feature_set: ConstraintFeatureSet,
    policy: ConstraintEnginePolicy,
) -> tuple[ConstraintDefinition, ...]:
    """Select constraints applicable to a feature set."""

    if not isinstance(registry, ConstraintRegistry):
        raise TypeError("registry must be ConstraintRegistry")
    if not isinstance(feature_set, ConstraintFeatureSet):
        raise TypeError("feature_set must be ConstraintFeatureSet")
    if not isinstance(policy, ConstraintEnginePolicy):
        raise TypeError("policy must be ConstraintEnginePolicy")

    candidates = registry.constraints if policy.evaluate_inactive else find_active_constraints(registry)
    feature_names = set(feature_set.names())
    feature_scope = feature_set.metadata.get("feature_scope")

    selected: list[ConstraintDefinition] = []
    for constraint in candidates:
        if not isinstance(constraint, ConstraintDefinition):
            raise TypeError("registry constraints must contain ConstraintDefinition")
        if not policy.evaluate_inactive and constraint.status is not ConstraintStatus.ACTIVE:
            continue

        if constraint.scope is ConstraintScope.GLOBAL:
            selected.append(constraint)
            continue

        if not _scope_is_compatible(
            constraint=constraint,
            feature_scope=feature_scope,
        ):
            continue

        target_names = {constraint.target.target_id}
        if constraint.target.field_path:
            target_names.add(constraint.target.field_path[-1])

        if feature_names.intersection(target_names):
            selected.append(constraint)

    return tuple(selected)


def observation_id_from_feature_sets(
    feature_sets: tuple[ConstraintFeatureSet, ...],
) -> str:
    """Resolve observation_id for an evaluation set."""

    if not isinstance(feature_sets, tuple):
        raise TypeError("feature_sets must be a tuple")
    if not feature_sets:
        raise EmptyConstraintSetError(message="feature_sets must not be empty")
    for feature_set in feature_sets:
        if not isinstance(feature_set, ConstraintFeatureSet):
            raise TypeError("feature_sets must contain ConstraintFeatureSet")
        observation_id = feature_set.input_ref.observation_id
        if isinstance(observation_id, str) and observation_id:
            return observation_id
    return "unknown_observation"


def run_constraint_engine(
    *,
    registry: ConstraintRegistry,
    feature_sets: tuple[ConstraintFeatureSet, ...] | list[ConstraintFeatureSet],
    policy: ConstraintEnginePolicy | None = None,
    metadata: dict[str, object] | None = None,
) -> ConstraintEngineResult:
    """Run registered constraints against provided feature sets."""

    try:
        if not isinstance(registry, ConstraintRegistry):
            raise TypeError("registry must be ConstraintRegistry")
        if not isinstance(feature_sets, (tuple, list)):
            raise TypeError("feature_sets must be a tuple or list")
        feature_sets_tuple = tuple(feature_sets)
        if not feature_sets_tuple:
            raise EmptyConstraintSetError(message="feature_sets must not be empty")
        for feature_set in feature_sets_tuple:
            if not isinstance(feature_set, ConstraintFeatureSet):
                raise TypeError("feature_sets must contain ConstraintFeatureSet")

        resolved_policy = ConstraintEnginePolicy() if policy is None else policy
        if not isinstance(resolved_policy, ConstraintEnginePolicy):
            raise TypeError("policy must be ConstraintEnginePolicy or None")

        evaluations: list[ConstraintEvaluation] = []
        selected_constraint_count = 0
        for feature_set in feature_sets_tuple:
            selected_constraints = select_constraints_for_feature_set(
                registry=registry,
                feature_set=feature_set,
                policy=resolved_policy,
            )
            selected_constraint_count += len(selected_constraints)
            if selected_constraints:
                evaluations.extend(
                    evaluate_constraints(
                        constraints=selected_constraints,
                        feature_set=feature_set,
                        policy=resolved_policy.evaluator_policy,
                    )
                )

        if selected_constraint_count == 0 and resolved_policy.fail_on_empty_constraints:
            raise EmptyConstraintSetError(message="no constraints selected")

        resolved_metadata = {
            "engine": "constraint_engine_v1",
            "feature_set_count": len(feature_sets_tuple),
            "selected_constraint_count": selected_constraint_count,
        }
        if metadata is not None:
            ensure_metadata_dict(metadata)
            resolved_metadata.update(metadata)
        resolved_metadata["engine"] = "constraint_engine_v1"

        evaluation_set = ConstraintEvaluationSet(
            set_id=new_constraint_evaluation_set_id(),
            observation_id=observation_id_from_feature_sets(feature_sets_tuple),
            evaluations=tuple(evaluations),
            created_at=utc_now(),
            metadata=resolved_metadata,
        )
        return ConstraintEngineResult(
            evaluation_set=evaluation_set,
            selected_constraint_count=selected_constraint_count,
            feature_set_count=len(feature_sets_tuple),
            metadata=resolved_metadata,
        )
    except EmptyConstraintSetError:
        raise
    except ConstraintRegistryError:
        raise
    except ConstraintEvaluationError:
        raise
    except Exception as exc:
        raise ConstraintEvaluationError(
            message="failed to run constraint engine",
            context=build_constraint_error_context(operation="run_constraint_engine"),
            cause=exc,
        ) from exc


def run_constraint_engine_for_feature_set(
    *,
    registry: ConstraintRegistry,
    feature_set: ConstraintFeatureSet,
    policy: ConstraintEnginePolicy | None = None,
    metadata: dict[str, object] | None = None,
) -> ConstraintEngineResult:
    """Run the constraint engine for one feature set."""

    if not isinstance(feature_set, ConstraintFeatureSet):
        raise TypeError("feature_set must be ConstraintFeatureSet")
    return run_constraint_engine(
        registry=registry,
        feature_sets=(feature_set,),
        policy=policy,
        metadata=metadata,
    )


def _scope_is_compatible(
    *,
    constraint: ConstraintDefinition,
    feature_scope: object,
) -> bool:
    if constraint.scope is constraint.target.scope:
        return True
    if isinstance(feature_scope, str) and feature_scope:
        return constraint.target.scope.name.lower() == feature_scope.lower()
    return False


def _ensure_non_negative_int(value: int, field_name: str) -> int:
    ensure_non_empty_string(field_name, "field_name")
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{field_name} must be greater than or equal to 0")
    return value


__all__ = [
    "ConstraintEnginePolicy",
    "ConstraintEngineResult",
    "select_constraints_for_feature_set",
    "observation_id_from_feature_sets",
    "run_constraint_engine",
    "run_constraint_engine_for_feature_set",
]
