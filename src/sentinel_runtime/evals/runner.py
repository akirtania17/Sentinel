"""Run a fixture suite through the normal runtime.

Every case goes through ``Runtime.aevaluate`` exactly as production code would, so batching,
normalization and error handling are what the report measures. Cases run concurrently under a
semaphore (TypeSafe rate-limits shared keys at around eight concurrent requests). A provider error
on one case is recorded in the report; an offline cache miss on any case stops the run and lists
every missing case.
"""

import asyncio
from collections.abc import Sequence

from sentinel_runtime.errors import CacheMissError, ProviderError
from sentinel_runtime.evals.fixtures import Case
from sentinel_runtime.evals.report import CaseRun
from sentinel_runtime.predicates import PredicateSet
from sentinel_runtime.runtime import Runtime

DEFAULT_CONCURRENCY = 6


async def run_cases(
    runtime: Runtime,
    predicate_set: PredicateSet,
    cases: Sequence[Case],
    *,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> list[CaseRun]:
    """Evaluate every case; raise ``CacheMissError`` naming all cases missing from an offline cache."""
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    semaphore = asyncio.Semaphore(concurrency)
    missing: list[tuple[str, list[str]]] = []

    async def one(case: Case) -> CaseRun:
        async with semaphore:
            try:
                evaluation = await runtime.aevaluate(case.state, predicate_set)
            except ProviderError as error:
                return CaseRun(case=case, evaluation=error.partial, error=error.info())
            except CacheMissError as error:
                missing.append((case.id, error.keys))
                return CaseRun(case=case, evaluation=None)
            return CaseRun(case=case, evaluation=evaluation)

    runs = await asyncio.gather(*(one(case) for case in cases))
    if missing:
        order = {case.id: index for index, case in enumerate(cases)}
        missing.sort(key=lambda item: order[item[0]])
        raise CacheMissError(
            [key for _, keys in missing for key in keys], [case_id for case_id, _ in missing]
        )
    return list(runs)
