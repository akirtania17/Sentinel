"""Sentinel: a semantic runtime.

Define semantic predicates over application state, evaluate them with fast calibrated models, and
branch on typed probabilities in ordinary code.
"""

from importlib import metadata as _metadata

from sentinel_runtime.errors import (
    CacheMissError,
    ConfigError,
    ErrorInfo,
    FixtureValidationError,
    PredicateValidationError,
    ProviderError,
    SentinelError,
    StateError,
)
from sentinel_runtime.predicates import (
    ChoicePredicate,
    NoulCriteria,
    NoulPredicate,
    Predicate,
    PredicateSet,
    ScorePredicate,
    choice,
    noul,
    score,
)
from sentinel_runtime.pricing import PricingTable
from sentinel_runtime.providers import (
    CachingProvider,
    FakeProvider,
    Provider,
    ProviderLimits,
    ProviderRequest,
    ProviderResponse,
    TypeSafeProvider,
)
from sentinel_runtime.results import (
    ChoiceResult,
    Evaluation,
    NoulResult,
    ProviderCall,
    Result,
    ScoreResult,
    Usage,
)
from sentinel_runtime.runtime import Runtime
from sentinel_runtime.state import State
from sentinel_runtime.telemetry import CallbackSink, EvaluationEvent, JsonlSink, LoggingSink, TelemetrySink

try:
    __version__ = _metadata.version("sentinel-runtime")
except _metadata.PackageNotFoundError:  # pragma: no cover - running from a source tree that is not installed
    __version__ = "0+unknown"

__all__ = [
    "CacheMissError",
    "CachingProvider",
    "CallbackSink",
    "ChoicePredicate",
    "ChoiceResult",
    "ConfigError",
    "ErrorInfo",
    "Evaluation",
    "EvaluationEvent",
    "FakeProvider",
    "FixtureValidationError",
    "JsonlSink",
    "LoggingSink",
    "NoulCriteria",
    "NoulPredicate",
    "NoulResult",
    "Predicate",
    "PredicateSet",
    "PredicateValidationError",
    "PricingTable",
    "Provider",
    "ProviderCall",
    "ProviderError",
    "ProviderLimits",
    "ProviderRequest",
    "ProviderResponse",
    "Result",
    "Runtime",
    "ScorePredicate",
    "ScoreResult",
    "SentinelError",
    "State",
    "StateError",
    "TelemetrySink",
    "TypeSafeProvider",
    "Usage",
    "__version__",
    "choice",
    "noul",
    "score",
]
