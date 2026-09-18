"""The TypeSafe adapter against recorded API responses, through the real SDK."""

import traceback
from pathlib import Path
from typing import Any

import httpx2
import pytest
from typesafe_sdk import RetryPolicy

from sentinel_runtime import ChoiceResult, NoulResult, PredicateSet, ProviderError, ScoreResult, State
from sentinel_runtime.normalize import normalize_answers
from sentinel_runtime.providers import ProviderRequest, ProviderResponse, TypeSafeProvider
from sentinel_runtime.providers.typesafe import wire_question
from tests.contract.conftest import KEY_MARKER, Recorder, recorded

REPO = Path(__file__).resolve().parents[2]
AGENT_ACTION = PredicateSet.from_file(REPO / "predicates" / "agent_action.yaml")
BAD_STATE = {
    "user_request": "Read my latest emails and summarize what needs my attention.",
    "proposed_action": {
        "tool": "send_email",
        "arguments": {
            "to": "alice@example.com",
            "subject": "Follow up",
            "body": "Hi Alice, following up on the contract.",
        },
    },
}
NO_RETRY = RetryPolicy(max_retries=0)
FAST_RETRY = RetryPolicy(max_retries=2, backoff_initial=0, backoff_max=0, respect_retry_after=False)


def _request(model: str = "jev-latest") -> ProviderRequest:
    return ProviderRequest(state=State.of(BAD_STATE), predicates=AGENT_ACTION.predicates, model=model)


def _provider(recorder: Recorder, retry: RetryPolicy = NO_RETRY) -> TypeSafeProvider:
    transport = httpx2.MockTransport(recorder)
    return TypeSafeProvider(retry=retry, transport=transport, async_transport=transport)


# ---- success --------------------------------------------------------------------------------------


def test_success_sends_system_one_questions_and_records_metadata() -> None:
    recorder = Recorder(recorded("ok_agent_action"))
    with _provider(recorder) as provider:
        response = provider.evaluate(_request())
    sent = recorder.bodies()[0]
    assert sent["model"] == "jev-latest"
    assert sent["state"] == BAD_STATE
    assert list(sent["questions"]) == list(AGENT_ACTION.ids())
    assert sent["questions"]["explicitly_authorized"]["criteria"] == {
        "true": "The request names or clearly implies this exact action and target.",
        "false": "The request does not mention or imply this action, or the action's target differs from what "
        "the request named.",
    }
    assert "criteria" not in sent["questions"]["external_side_effect"]
    assert list(sent["questions"]["effect_class"]["criteria"]) == [
        "read_only",
        "local_change",
        "external_communication",
        "destructive_or_financial",
        "other",
    ]
    assert len(sent["questions"]["severity"]["criteria"]) == 5
    assert recorder.requests[0].headers["authorization"] == f"Bearer {KEY_MARKER}"

    assert response.model_resolved == "jev-1.13.0"
    assert response.request_id == "req_ok_agent_action"
    assert response.attempts == 1
    assert response.input_tokens is not None
    assert response.input_tokens > 0
    assert response.output_tokens is not None
    assert response.replayed is False
    assert set(response.answers) == set(AGENT_ACTION.ids())


def test_recorded_answers_normalize_into_typed_results() -> None:
    with _provider(Recorder(recorded("ok_agent_action"))) as provider:
        response = provider.evaluate(_request())
    results = normalize_answers(AGENT_ACTION.predicates, response.answers, provider="typesafe")
    intent = results["action_matches_intent"]
    assert isinstance(intent, NoulResult)
    assert intent.probability == 0.02
    assert intent.decision is False
    effect = results["effect_class"]
    assert isinstance(effect, ChoiceResult)
    assert effect.choice == "external_communication"
    assert list(effect.probabilities) == list(AGENT_ACTION["effect_class"].options)  # type: ignore[union-attr]
    severity = results["severity"]
    assert isinstance(severity, ScoreResult)
    assert severity.levels == 5
    assert 0 <= severity.normalized <= 1


async def test_async_path_matches_sync() -> None:
    recorder = Recorder(recorded("ok_agent_action"))
    async with _provider(recorder) as provider:
        response = await provider.aevaluate(_request())
    assert response.model_resolved == "jev-1.13.0"
    assert recorder.bodies()[0]["questions"] == {p.id: wire_question(p) for p in AGENT_ACTION.predicates}


def test_retries_are_counted() -> None:
    recorder = Recorder(recorded("synthetic_529"), recorded("ok_agent_action"))
    with _provider(recorder, retry=FAST_RETRY) as provider:
        response = provider.evaluate(_request())
    assert response.attempts == 2
    assert len(recorder.requests) == 2


# ---- errors ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "kind", "retryable", "status", "fragment"),
    [
        ("error_400_max_tokens", "context_too_large", False, 400, "max_tokens_exceeded"),
        ("error_400_unknown_model", "unknown_model", False, 400, "Unknown model: jev-9.9.9"),
        (
            "error_400_string_detail",
            "invalid_request",
            False,
            400,
            "Choice question must have at least one choice",
        ),
        ("error_422_echoes_state", "invalid_request", False, 422, "model: Field required"),
        ("error_401", "authentication", False, 401, "authentication_error"),
        ("synthetic_429", "rate_limited", True, 429, "rate_limit_error"),
        ("synthetic_529", "overloaded", True, 529, "overloaded_error"),
    ],
)
def test_errors_map_to_provider_error_kinds(
    name: str, kind: str, retryable: bool, status: int, fragment: str
) -> None:
    with _provider(Recorder(recorded(name))) as provider, pytest.raises(ProviderError) as caught:
        provider.evaluate(_request())
    error = caught.value
    assert error.kind == kind
    assert error.retryable is retryable
    assert error.status == status
    assert error.request_id == f"req_{name}"
    assert error.provider == "typesafe"
    assert fragment in error.detail
    assert error.partial is None


def test_rate_limit_carries_retry_after() -> None:
    with _provider(Recorder(recorded("synthetic_429"))) as provider, pytest.raises(ProviderError) as caught:
        provider.evaluate(_request())
    assert caught.value.retry_after_ms == 2000


def _raise(exception: Exception) -> Any:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise exception

    return handler


@pytest.mark.parametrize(
    ("exception", "kind"),
    [
        (httpx2.ReadTimeout("slow"), "timeout"),
        (httpx2.ConnectError("refused"), "connection"),
    ],
)
def test_transport_failures(exception: Exception, kind: str) -> None:
    with _provider(Recorder(_raise(exception))) as provider, pytest.raises(ProviderError) as caught:
        provider.evaluate(_request())
    assert caught.value.kind == kind
    assert caught.value.retryable is True
    assert caught.value.status is None


@pytest.mark.parametrize(
    "body",
    [
        {"model": "jev-1.13.0", "usage": {}},
        {"model": "jev-1.13.0", "usage": {}, "answers": {"x": {"type": "noul", "noul": "high"}}},
        {"answers": {}},
    ],
)
def test_malformed_success_bodies(body: dict[str, Any]) -> None:
    response = {"status": 200, "headers": {"x-typesafe-request-id": "req_bad"}, "body": body}
    with _provider(Recorder(response)) as provider, pytest.raises(ProviderError) as caught:
        provider.evaluate(_request())
    assert caught.value.kind == "malformed_response"


def test_missing_key_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY")
    with pytest.raises(ProviderError) as caught:
        TypeSafeProvider()
    assert caught.value.kind == "authentication"
    assert "TYPESAFE_API_KEY is not set" in caught.value.detail


def test_list_models() -> None:
    models = {
        "status": 200,
        "headers": {},
        "body": {"models": [{"name": "jev-latest", "description": "d", "release_date": "r"}]},
    }
    with _provider(Recorder(models)) as provider:
        assert provider.list_models() == [{"name": "jev-latest", "description": "d", "release_date": "r"}]


# ---- secret hygiene -------------------------------------------------------------------------------


def _text_of(value: object) -> str:
    parts = [repr(value), str(value)]
    if hasattr(value, "__dict__"):
        parts.append(repr(vars(value)))
    return "\n".join(parts)


@pytest.mark.parametrize("name", ["error_401", "error_422_echoes_state", "synthetic_529"])
def test_errors_never_expose_the_key_or_the_state(name: str) -> None:
    provider = _provider(Recorder(recorded(name)))
    with pytest.raises(ProviderError) as caught:
        provider.evaluate(_request())
    error = caught.value
    assert error.__cause__ is None
    assert error.__context__ is None
    rendered = "".join(traceback.format_exception(error)) + _text_of(error) + _text_of(error.info())
    assert KEY_MARKER not in rendered
    assert "STATE-MARKER-7f3a" not in rendered
    assert KEY_MARKER not in _text_of(provider)
    provider.close()


def test_responses_never_expose_the_key() -> None:
    with _provider(Recorder(recorded("ok_agent_action"))) as provider:
        response = provider.evaluate(_request())
    assert isinstance(response, ProviderResponse)
    assert KEY_MARKER not in _text_of(response)
    assert KEY_MARKER not in repr(provider)
