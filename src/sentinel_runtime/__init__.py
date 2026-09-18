"""Sentinel: a semantic runtime.

Define semantic predicates over application state, evaluate them with fast calibrated models, and
branch on typed probabilities in ordinary code.
"""

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
from sentinel_runtime.results import (
    ChoiceResult,
    Evaluation,
    NoulResult,
    ProviderCall,
    Result,
    ScoreResult,
    Usage,
)
from sentinel_runtime.state import State

__version__ = "0.1.0.dev0"

__all__ = [
    "CacheMissError",
    "ChoicePredicate",
    "ChoiceResult",
    "ConfigError",
    "ErrorInfo",
    "Evaluation",
    "FixtureValidationError",
    "NoulCriteria",
    "NoulPredicate",
    "NoulResult",
    "Predicate",
    "PredicateSet",
    "PredicateValidationError",
    "ProviderCall",
    "ProviderError",
    "Result",
    "ScorePredicate",
    "ScoreResult",
    "SentinelError",
    "State",
    "StateError",
    "Usage",
    "__version__",
    "choice",
    "noul",
    "score",
]
