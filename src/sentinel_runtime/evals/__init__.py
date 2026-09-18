"""Evaluation and calibration: measure predicate quality against labeled fixtures."""

from sentinel_runtime.evals.compare import Comparison, compare_reports
from sentinel_runtime.evals.fixtures import Case, EvalFileV1, EvalSuite, Expectation, ScoreRange
from sentinel_runtime.evals.report import CaseRun, EvalReport, build_report, render_markdown, render_report
from sentinel_runtime.evals.runner import run_cases
from sentinel_runtime.evals.stability import StabilityReport, measure_stability

__all__ = [
    "Case",
    "CaseRun",
    "Comparison",
    "EvalFileV1",
    "EvalReport",
    "EvalSuite",
    "Expectation",
    "ScoreRange",
    "StabilityReport",
    "build_report",
    "compare_reports",
    "measure_stability",
    "render_markdown",
    "render_report",
    "run_cases",
]
