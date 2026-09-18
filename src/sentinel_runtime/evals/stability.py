"""Repeat stability: how much answers move when the same state is evaluated again.

Semantic models are approximately, not exactly, deterministic. ``measure_stability`` evaluates a
sample of cases several times live (never from a cache) and reports, per predicate, how far answers
move and how often that movement crosses a decision boundary. Use it to choose regression
tolerances and to decide whether a predicate is stable enough to gate on.
"""

import asyncio
import random
from collections.abc import Mapping, Sequence

from pydantic import BaseModel, ConfigDict

from sentinel_runtime.evals.fixtures import Case
from sentinel_runtime.predicates import PredicateSet
from sentinel_runtime.results import ChoiceResult, Evaluation, NoulResult, ScoreResult
from sentinel_runtime.runtime import Runtime


class PredicateStability(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: str
    cases: int
    mean_spread: float
    """Mean over cases of (max - min) across repeats: probability for nouls, top-option probability
    for choices, score for scores."""
    max_spread: float
    changed: int
    """Cases whose decision changed across repeats: threshold side (noul), chosen option (choice),
    or nearest level (score)."""


class NoisyAnswer(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_id: str
    predicate: str
    minimum: float
    maximum: float


class StabilityReport(BaseModel):
    model_config = ConfigDict(frozen=True, title="Sentinel stability report (v1)")

    suite: str
    model_requested: str
    models_resolved: list[str]
    repeats: int
    sample: list[str]
    seed: int
    predicates: dict[str, PredicateStability]
    noisiest: list[NoisyAnswer]


def sample_cases(cases: Sequence[Case], size: int, seed: int) -> list[Case]:
    """A reproducible sample of up to ``size`` cases, in suite order."""
    if size >= len(cases):
        return list(cases)
    chosen = set(random.Random(seed).sample(range(len(cases)), size))  # noqa: S311 - sampling, not cryptography.
    return [case for index, case in enumerate(cases) if index in chosen]


def _spread(values: Sequence[float]) -> float:
    return round(max(values) - min(values), 6) if values else 0.0


async def measure_stability(
    runtime: Runtime,
    suite_name: str,
    predicate_set: PredicateSet,
    cases: Sequence[Case],
    *,
    repeats: int,
    thresholds: Mapping[str, float],
    seed: int,
    concurrency: int = 6,
) -> StabilityReport:
    """Evaluate each case ``repeats`` times and summarize the movement."""
    if repeats < 2:
        raise ValueError("stability needs at least two repeats")
    semaphore = asyncio.Semaphore(concurrency)

    async def one(case: Case) -> Evaluation:
        async with semaphore:
            return await runtime.aevaluate(case.state, predicate_set)

    rounds = [await asyncio.gather(*(one(case) for case in cases)) for _ in range(repeats)]
    models = sorted(
        {model for evaluations in rounds for e in evaluations for model in e.model_resolved.split(",")}
    )
    summaries: dict[str, PredicateStability] = {}
    noisy: list[NoisyAnswer] = []
    for predicate in predicate_set.predicates:
        spreads: list[float] = []
        changed = 0
        for index, case in enumerate(cases):
            results = [evaluations[index].results[predicate.id] for evaluations in rounds]
            if isinstance(results[0], NoulResult):
                values = [r.probability for r in results if isinstance(r, NoulResult)]
                threshold = thresholds.get(predicate.id, 0.5)
                decided = {value >= threshold for value in values}
            elif isinstance(results[0], ChoiceResult):
                choices = [r for r in results if isinstance(r, ChoiceResult)]
                values = [r.top_probability for r in choices]
                decided = {r.choice for r in choices}
            else:
                scores = [r for r in results if isinstance(r, ScoreResult)]
                values = [r.score for r in scores]
                decided = {r.nearest_level for r in scores}
            spreads.append(_spread(values))
            changed += len(decided) > 1
            if values:
                noisy.append(
                    NoisyAnswer(
                        case_id=case.id, predicate=predicate.id, minimum=min(values), maximum=max(values)
                    )
                )
        summaries[predicate.id] = PredicateStability(
            kind=predicate.kind,
            cases=len(cases),
            mean_spread=round(sum(spreads) / len(spreads), 6) if spreads else 0.0,
            max_spread=max(spreads, default=0.0),
            changed=changed,
        )
    noisy.sort(key=lambda answer: answer.maximum - answer.minimum, reverse=True)
    return StabilityReport(
        suite=suite_name,
        model_requested=rounds[0][0].model_requested if cases else "",
        models_resolved=[model for model in models if model],
        repeats=repeats,
        sample=[case.id for case in cases],
        seed=seed,
        predicates=summaries,
        noisiest=[answer for answer in noisy[:10] if answer.maximum > answer.minimum],
    )
