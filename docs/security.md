# Security

## Semantic checks supplement authorization

Sentinel tells your application what an action or an answer means: whether a tool call follows from
the user's request, whether an answer is grounded, whether a step drifts out of scope. It does not
know who the user is or what they are allowed to do, and it is not a security boundary.

- Run identity and permission checks first, in code. A semantic predicate answers a different
  question ("did this user ask for this specific action?") on top of "may this user do this?". The
  reference suite includes a request that claims to come from the CEO: the predicates correctly
  report that the request authorizes the transfer, and only ordinary authorization can decide
  whether the requester may make it.
- Keep deterministic controls deterministic: rate limits, allow lists, amount limits, schema checks.
- Use explicit confirmation for high-stakes actions, with stricter thresholds for riskier kinds of
  action. Thresholds belong to the decision, not the predicate.

## Adversarial content

State is data, and the model does not treat it as hostile by default. Text inside the state that
argues for a conclusion can move an answer. In the reference suite, an instruction injected into an
email body, a self-justifying field on the proposed action, and a memory note claiming standing
permission did not change the authorization predicates, because each predicate is scoped to the
field it judges (`user_request`), but that is evidence, not a guarantee.

- Point security-relevant predicates at the authoritative field, not at the whole conversation.
- Protect secrets with deterministic rules. In the [field test](field-test.md), agents tidying a
  repository read its credentials file 13 times; reading a file while tidying does follow the
  request, so an intent check rightly allows it. A permission rule that denies reading `.env` does not
  depend on judgment.
- Include injection attempts in your evaluation fixtures and gate on them in CI.
- Treat every result as one signal among your other controls.

## Failing open or closed

A provider failure is always a `ProviderError`, never a silent `false`. Decide per decision what
happens when judgments are unavailable. The agent example fails closed (it denies the action); a
documentation recommender might fail open.

## Secrets

- The TypeSafe key is read by the TypeSafe SDK from `TYPESAFE_API_KEY`. Sentinel never reads it into
  its own variables, never accepts it as an argument, and never stores it on its objects.
- Provider errors are raised without exception chaining, so the SDK's HTTP objects, whose request
  headers carry the key, are not reachable from them. Error details contain only the provider's
  error type and message, never the echoed request. A test injects a marker key and checks it never
  appears in errors, tracebacks, events, or object representations.
- The CLI never prints local variables in tracebacks.
- `.env` files are git-ignored. `gitleaks` runs in CI and in the optional pre-commit hook, with a rule
  for the TypeSafe key format.
- Normal CI never needs a key. The scheduled live workflow reads it from the repository secret.

## Data

Evaluating a predicate sends its state to the provider. Send only the fields your predicates need,
and use the `redact` hook on `Runtime` to remove personal data before it leaves your process:

```python
def redact(state):
    state["user"].pop("email", None)
    return state


runtime = Runtime.from_env(redact=redact)
```

Events record a digest of the state, not the state, unless you opt in. Response caches store the
provider's answers keyed by digests, never the state itself. TypeSafe documents that Jev is not
trained on customer requests; see its legal pages for data retention terms.
