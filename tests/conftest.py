"""Shared test configuration."""

import os
from pathlib import Path

import pytest
from hypothesis import HealthCheck, settings

settings.register_profile("ci", deadline=None, max_examples=200, suppress_health_check=[HealthCheck.too_slow])
settings.register_profile("dev", deadline=None, max_examples=60)
settings.load_profile("ci" if os.environ.get("CI") else "dev")

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(autouse=True)
def _isolate_model_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests must not depend on the developer's model environment variables."""
    monkeypatch.delenv("SENTINEL_MODEL", raising=False)
    monkeypatch.delenv("TYPESAFE_DEFAULT_MODEL", raising=False)
