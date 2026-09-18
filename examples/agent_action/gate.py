"""Gate an agent's proposed tool call on semantic facts plus explicit application policy.

Sentinel reports facts about the proposed action: does it follow from the request, does it reach
outside the session, did the user authorize this exact action and target, what kind of effect is it.
The policy below turns those facts into allow, confirm, or deny. The policy is yours: change the
thresholds to match your risk, and measure them with ``sentinel eval run agent_action`` rather than
trusting these illustrative values.

Semantic checks supplement authorization; they never replace it. Run your normal permission
checks first, then ask whether this specific action is what the user meant.
"""

from dataclasses import dataclass
from typing import Any, Literal

from sentinel_runtime import ChoiceResult, Evaluation, PredicateSet, ProviderError, Runtime

Action = Literal["allow", "confirm", "deny"]

# Thresholds from `sentinel eval run agent_action`: at 0.5 both nouls reached precision 1.00 and
# recall 0.94 on the labeled suite. Destructive and financial actions use a stricter bar on purpose,
# trading more confirmations for safety. Re-run the evaluation before changing these.
MATCH = 0.5
"""Below this, the action does not follow from the request."""
AUTHORIZED = 0.5
AUTHORIZED_HIGH_RISK = 0.8
"""Destructive or financial actions need stronger evidence of explicit authorization."""
SIDE_EFFECT = 0.5


@dataclass(frozen=True)
class Decision:
    """What the application does with a proposed tool call, and why."""

    action: Action
    reason: str


def decide(evaluation: Evaluation) -> Decision:
    """Apply the application's policy to the semantic facts."""
    matches = evaluation.probability("action_matches_intent")
    side_effect = evaluation.probability("external_side_effect")
    authorized = evaluation.probability("explicitly_authorized")
    effect = evaluation["effect_class"]
    high_risk = isinstance(effect, ChoiceResult) and effect.choice == "destructive_or_financial"

    if matches < MATCH and side_effect >= SIDE_EFFECT:
        return Decision("deny", "the action has effects the request did not ask for")
    if matches < MATCH:
        return Decision("confirm", "the action does not clearly follow from the request")
    if high_risk and authorized < AUTHORIZED_HIGH_RISK:
        return Decision("confirm", "destructive or financial action without clear explicit authorization")
    if side_effect >= SIDE_EFFECT and authorized < AUTHORIZED:
        return Decision("confirm", "external effect without explicit authorization of this exact action")
    return Decision("allow", "the action follows from and is authorized by the request")


def gate_tool_call(
    user_request: str, proposed_action: dict[str, Any], runtime: Runtime, predicates: PredicateSet
) -> tuple[Decision, Evaluation | None]:
    """Evaluate a proposed tool call and decide. Fails closed when the semantic check is unavailable."""
    try:
        evaluation = runtime.evaluate(
            {"user_request": user_request, "proposed_action": proposed_action}, predicates
        )
    except ProviderError as error:
        # This application chooses to fail closed; a low-stakes one might fail open instead.
        return Decision("deny", f"semantic check unavailable ({error.kind}); failing closed"), None
    return decide(evaluation), evaluation
