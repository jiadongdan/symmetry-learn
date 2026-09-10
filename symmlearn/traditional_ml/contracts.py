"""Versioned contracts for the traditional machine-learning validation workflow.

This module is deliberately independent of the pretrained CNN. It only knows
about the validated eight-channel feature representation, user-selected support
patches, and two conventional scikit-learn classifiers. Nothing here loads,
probes, or requires a pretrained checkpoint.

The classifier parameter registry in this module is the single source of truth
for both strict validation and the Provider capability document, so Harness
never has to invent model defaults of its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
import math
from typing import Any, Mapping


TRADITIONAL_ML_OPERATION = "traditional_ml_analyze"
TRADITIONAL_ML_CAPABILITY_SCHEMA_VERSION = "symmetry-traditional-ml-capability-v1"
TRADITIONAL_ML_RECORD_SCHEMA_VERSION = "symmetry-traditional-ml-run-v1"

RAW_IMAGE_MODE = "raw_image"
IMAGE_PLUS_SYMMETRY_MAPS_MODE = "image_plus_symmetry_maps"
FEATURE_MODES = (RAW_IMAGE_MODE, IMAGE_PLUS_SYMMETRY_MAPS_MODE)

LOGISTIC_REGRESSION = "logistic_regression"
RANDOM_FOREST = "random_forest"
CLASSIFIER_IDENTIFIERS = (LOGISTIC_REGRESSION, RANDOM_FOREST)

DEFAULT_CLASSIFIER_PATCH_SIZE = 64
DEFAULT_SEED = 42
DEFAULT_STRIDE = 4
DEFAULT_BATCH_SIZE = 512
MINIMUM_PATCHES_PER_CLASS = 2
MAXIMUM_PATCHES_PER_CLASS = 5000

_MAXIMUM_SEED = 2**32 - 1


class TraditionalMLError(RuntimeError):
    """Raised when a traditional ML workflow cannot produce a valid result."""


# --- parameter specification -------------------------------------------------


POSITIVE_FLOAT = "positive_float"
POSITIVE_INT = "positive_int"
OPTIONAL_POSITIVE_INT = "optional_positive_int"
CHOICE = "choice"
FIXED = "fixed"

_NONE_LIKE_STRINGS = frozenset({"none", "null", ""})


def _is_none_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in _NONE_LIKE_STRINGS
    return False


def _coerce_number(value: Any, name: str) -> float:
    """Return a finite float for a JSON-supplied number, or raise."""
    if value is None or isinstance(value, bool):
        raise ValueError(f"{name} must be a number.")
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError as error:
            raise ValueError(f"{name} must be a number.") from error
    else:
        raise ValueError(f"{name} must be a number.")
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    return number


@dataclass(frozen=True)
class ParameterSpec:
    """One strictly validated classifier parameter.

    ``default`` is stored in its user-facing (display) form. ``mapping``
    translates a display value into the scikit-learn value when the two differ,
    for example ``"all" -> None`` or ``"none" -> None``.
    """

    name: str
    kind: str
    default: Any
    choices: tuple[str, ...] = ()
    mapping: tuple[tuple[str, Any], ...] = ()

    def resolve(self, value: Any) -> Any:
        """Resolve one supplied value into its scikit-learn form."""
        if self.kind == FIXED:
            if value is None or value == self.default:
                return self.default
            raise ValueError(
                f"{self.name} is fixed at {self.default!r} for this classifier."
            )
        if self.kind == POSITIVE_FLOAT:
            number = _coerce_number(value, self.name)
            if number <= 0:
                raise ValueError(f"{self.name} must be a positive number.")
            return float(number)
        if self.kind == POSITIVE_INT:
            number = _coerce_number(value, self.name)
            if number <= 0 or number != int(number):
                raise ValueError(f"{self.name} must be a positive integer.")
            return int(number)
        if self.kind == OPTIONAL_POSITIVE_INT:
            if _is_none_like(value):
                return None
            number = _coerce_number(value, self.name)
            if number <= 0 or number != int(number):
                raise ValueError(
                    f"{self.name} must be a positive integer or 'none'."
                )
            return int(number)
        if self.kind == CHOICE:
            options = ", ".join(self.choices)
            if _is_none_like(value):
                text = self._none_choice(options)
            else:
                text = str(value).strip()
                if text not in self.choices:
                    raise ValueError(f"{self.name} must be one of {options}.")
            return dict(self.mapping).get(text, text)
        raise RuntimeError(f"Unknown parameter kind {self.kind!r}.")

    def _none_choice(self, options: str) -> str:
        """Map a null value onto the choice that already resolves to None.

        ``class_weight`` has an explicit ``none`` choice, while
        ``max_features=None`` is scikit-learn's own spelling for ``all``.
        Requiring both to be expressible keeps JSON payloads unambiguous.
        """
        if "none" in self.choices:
            return "none"
        for display, resolved in self.mapping:
            if resolved is None:
                return display
        raise ValueError(f"{self.name} must be one of {options}.")


@dataclass(frozen=True)
class ClassifierSpecification:
    """A strict identifier-to-parameter contract for one classifier."""

    identifier: str
    parameters: tuple[ParameterSpec, ...] = field(default_factory=tuple)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.parameters)

    def defaults(self) -> dict[str, Any]:
        """Return the protocol defaults in user-facing form."""
        return {spec.name: spec.default for spec in self.parameters}

    def resolve(self, parameters: Mapping[str, Any] | None) -> dict[str, Any]:
        """Return fully resolved scikit-learn parameters.

        Unknown keys are rejected. Omitted keys fall back to the protocol
        default, which is advertised in the capability document; a supplied
        value is never silently replaced.
        """
        requested = dict(parameters or {})
        known = set(self.names)
        unknown = sorted(set(requested) - known)
        if unknown:
            raise ValueError(
                f"Unsupported {self.identifier} parameters: {unknown}. "
                f"Supported parameters are {sorted(known)}."
            )
        return {
            spec.name: spec.resolve(requested.get(spec.name, spec.default))
            for spec in self.parameters
        }


LOGISTIC_REGRESSION_SPECIFICATION = ClassifierSpecification(
    identifier=LOGISTIC_REGRESSION,
    parameters=(
        ParameterSpec("C", POSITIVE_FLOAT, 1.0),
        ParameterSpec(
            "class_weight",
            CHOICE,
            "balanced",
            choices=("balanced", "none"),
            mapping=(("balanced", "balanced"), ("none", None)),
        ),
        ParameterSpec("max_iter", POSITIVE_INT, 1000),
        ParameterSpec("solver", FIXED, "lbfgs"),
    ),
)

RANDOM_FOREST_SPECIFICATION = ClassifierSpecification(
    identifier=RANDOM_FOREST,
    parameters=(
        ParameterSpec("n_estimators", POSITIVE_INT, 300),
        ParameterSpec("max_depth", OPTIONAL_POSITIVE_INT, 12),
        ParameterSpec("min_samples_leaf", POSITIVE_INT, 2),
        ParameterSpec(
            "max_features",
            CHOICE,
            "sqrt",
            choices=("sqrt", "log2", "all"),
            mapping=(("sqrt", "sqrt"), ("log2", "log2"), ("all", None)),
        ),
        ParameterSpec(
            "class_weight",
            CHOICE,
            "balanced",
            choices=("balanced", "none"),
            mapping=(("balanced", "balanced"), ("none", None)),
        ),
        ParameterSpec("criterion", FIXED, "gini"),
        ParameterSpec("bootstrap", FIXED, True),
        ParameterSpec("n_jobs", FIXED, 1),
    ),
)

CLASSIFIER_SPECIFICATIONS: dict[str, ClassifierSpecification] = {
    specification.identifier: specification
    for specification in (
        LOGISTIC_REGRESSION_SPECIFICATION,
        RANDOM_FOREST_SPECIFICATION,
    )
}


def classifier_specification(identifier: str) -> ClassifierSpecification:
    """Return the specification for one supported classifier identifier."""
    key = str(identifier).strip()
    if key not in CLASSIFIER_SPECIFICATIONS:
        raise ValueError(
            f"Unsupported classifier {identifier!r}. "
            f"Supported classifiers are {list(CLASSIFIER_IDENTIFIERS)}."
        )
    return CLASSIFIER_SPECIFICATIONS[key]


def traditional_ml_capability() -> dict[str, Any]:
    """Return the structured traditional-ML capability block.

    Harness validates this block instead of inventing classifier defaults.
    """
    return {
        "schema_version": TRADITIONAL_ML_CAPABILITY_SCHEMA_VERSION,
        "feature_modes": list(FEATURE_MODES),
        "classifiers": [
            {
                "identifier": specification.identifier,
                "supports_predict_proba": True,
                "defaults": specification.defaults(),
            }
            for specification in CLASSIFIER_SPECIFICATIONS.values()
        ],
    }


# --- workflow options --------------------------------------------------------


def _positive_integer(value: Any, name: str) -> int:
    number = _coerce_number(value, name)
    if number <= 0 or number != int(number):
        raise ValueError(f"{name} must be a positive integer.")
    return int(number)


@dataclass(frozen=True)
class TraditionalMLOptions:
    """Validated options that fully determine one traditional ML run.

    ``classifier_patch_size`` belongs to the validation experiment, not to a
    pretrained model capability, so it is a user-editable field here.
    """

    classifier_patch_size: int = DEFAULT_CLASSIFIER_PATCH_SIZE
    seed: int = DEFAULT_SEED
    stride: int = DEFAULT_STRIDE
    batch_size: int = DEFAULT_BATCH_SIZE
    minimum_patches_per_class: int = MINIMUM_PATCHES_PER_CLASS
    maximum_patches_per_class: int = MAXIMUM_PATCHES_PER_CLASS

    def __post_init__(self) -> None:
        for name in (
            "classifier_patch_size",
            "stride",
            "batch_size",
            "minimum_patches_per_class",
            "maximum_patches_per_class",
        ):
            object.__setattr__(
                self, name, _positive_integer(getattr(self, name), name)
            )
        seed = _coerce_number(self.seed, "seed")
        if seed != int(seed) or not 0 <= int(seed) <= _MAXIMUM_SEED:
            raise ValueError(
                f"seed must be an integer between 0 and {_MAXIMUM_SEED}."
            )
        object.__setattr__(self, "seed", int(seed))
        if self.minimum_patches_per_class > self.maximum_patches_per_class:
            raise ValueError(
                "minimum_patches_per_class cannot exceed "
                "maximum_patches_per_class."
            )

    def support_options(self) -> dict[str, int]:
        """Return the option mapping expected by ``symmlearn.patches``."""
        return {
            "classifier_patch_size": self.classifier_patch_size,
            "minimum_shots_per_class": self.minimum_patches_per_class,
            "maximum_shots_per_class": self.maximum_patches_per_class,
        }

    def to_dict(self) -> dict[str, Any]:
        return {item.name: getattr(self, item.name) for item in fields(self)}

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "TraditionalMLOptions":
        """Build validated options from a JSON job payload."""
        requested = dict(payload or {})
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(requested) - known)
        if unknown:
            raise ValueError(
                f"Unsupported traditional ML options: {unknown}. "
                f"Supported options are {sorted(known)}."
            )
        return cls(**requested)


def validate_feature_mode(feature_mode: str) -> str:
    """Return one supported feature mode or raise an actionable error."""
    key = str(feature_mode).strip()
    if key not in FEATURE_MODES:
        raise ValueError(
            f"Unsupported feature mode {feature_mode!r}. "
            f"Supported feature modes are {list(FEATURE_MODES)}."
        )
    return key


__all__ = [
    "CHOICE",
    "CLASSIFIER_IDENTIFIERS",
    "CLASSIFIER_SPECIFICATIONS",
    "ClassifierSpecification",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_CLASSIFIER_PATCH_SIZE",
    "DEFAULT_SEED",
    "DEFAULT_STRIDE",
    "FEATURE_MODES",
    "FIXED",
    "IMAGE_PLUS_SYMMETRY_MAPS_MODE",
    "LOGISTIC_REGRESSION",
    "MAXIMUM_PATCHES_PER_CLASS",
    "MINIMUM_PATCHES_PER_CLASS",
    "OPTIONAL_POSITIVE_INT",
    "POSITIVE_FLOAT",
    "POSITIVE_INT",
    "ParameterSpec",
    "RANDOM_FOREST",
    "RAW_IMAGE_MODE",
    "TRADITIONAL_ML_CAPABILITY_SCHEMA_VERSION",
    "TRADITIONAL_ML_OPERATION",
    "TRADITIONAL_ML_RECORD_SCHEMA_VERSION",
    "TraditionalMLError",
    "TraditionalMLOptions",
    "classifier_specification",
    "traditional_ml_capability",
    "validate_feature_mode",
]
