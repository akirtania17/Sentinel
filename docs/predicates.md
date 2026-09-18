# Writing predicates

A predicate is one narrow judgment a semantic model makes about your application state. Good
predicates are the difference between a runtime you can calibrate and one you cannot. This guide
covers the file format, how to phrase each kind, and what to keep in code instead.

## The file format

Predicate sets live in YAML (or JSON) files, one set per file, reviewed like code:

```yaml
format: sentinel/predicates/v1
name: agent_action                 # identifier: lowercase letters, digits, underscores
description: Factual judgments about a proposed tool call.
predicates:
  explicitly_authorized:           # the predicate id; your code reads results by it
    kind: noul
    question: >-
      Does `user_request` explicitly or clearly authorize this specific `proposed_action`,
      including its target?
    criteria:
      true: The request names or clearly implies this exact action and target.
      false: The request does not mention or imply this action, or its target differs.
    threshold: 0.8                 # optional default decision threshold
    tags: [agent, authorization]   # optional, for filtering and reports
    description: Human notes. Never sent to the model.
```

Run `sentinel schema predicates` for the JSON Schema, and `sentinel predicates validate <file>` to
check a file and see authoring warnings. Pass `--state example.json` to also check that every
backticked path exists in a real state.

The predicate id is for your code; the model never sees it. Put the complete meaning in the
question.

## The three kinds

| Kind | Answers | Result | Use when |
| --- | --- | --- | --- |
| `noul` | Is this statement true? | `probability` of yes | A clean yes/no condition you will threshold |
| `choice` | Which of these options? | `choice`, `probabilities`, `confidence` | Exactly one of a closed set applies |
| `score` | Where on these levels? | `score`, `probabilities`, `confidence` | A position on an ordered scale you can describe in steps |

A `noul` probability near 0.5 means yes and no are about equally likely. It does not mean "medium".
If you want a degree, use a `score` with described levels.

### Nouls

- Phrase the question so that **yes means the condition holds**. Negated questions ("Does the reply
  *not* answer...") read less reliably; the linter flags them.
- Add `criteria` when the boundary is subtle. Say what counts as `true` and what counts as `false`.
- A `threshold` makes the result carry a `decision` (`probability >= threshold`). It is a default
  for convenience. Your application can and should apply stricter thresholds for riskier actions.

### Choices

- Options are a mapping of key to description. The key is what your code branches on; the
  description (or `null` when the key speaks for itself) tells the model what the option means.
- **Option order is part of the question** and part of the fingerprint.
- A choice always picks something. When the options might not cover every input, add an `other`
  or `none` option. The linter warns about sets of four or more without one.
- `min_confidence` makes the result carry a `decision`: the chosen option when the distribution is
  concentrated enough, `null` (abstain) otherwise.
- When two options blur together, describe each as an object with the same fields:

  ```yaml
  options:
    return_policy:
      what: Whether and how an item can be returned
      not_for: Progress of a return already sent
    return_status:
      what: Progress of a return already sent
      not_for: Whether and how an item can be returned
  ```

### Scores

- Levels are listed from lowest (level 0) to highest, 2 to 10 of them.
- **Describe situations, not degrees.** "Broken feature, but a workaround exists" gives the model
  something to match; "moderately severe" and bare numbers do not.
- Keep each score to one dimension. If a level says "fast and accurate and cheap", split it into
  three scores and combine them in code.
- Scores are useful for ranking and for informing decisions. They are the noisiest kind across
  repeated evaluations, so they carry no default decision; do not gate a high-stakes action on a
  score alone.

## Point predicates at the state

Send a JSON object with named fields and reference them in questions with backticked paths:
`` `user_request` ``, `` `proposed_action.arguments.to` ``, `` `previous_steps[0].result` ``. The
model then knows exactly which part of the state to judge. Keep the state to what the predicates
need; unrelated material costs accuracy.

## Views: what a predicate may see

A predicate can declare `sees`, the state paths it is shown. The runtime then sends it only that
part of the state, in a request of its own (predicates sharing a view share a request). Use it as a
trust boundary: a judgment that must not be argued with, such as "did the user ask for this", should
not see text an outsider can write.

```yaml
format: sentinel/predicates/v1
name: intent
sees:                        # every predicate below, unless it declares its own
  - user_request
  - proposed_action
  - recent_steps[].tool      # each step's tool and arguments, never its output
  - recent_steps[].arguments
predicates:
  requested:
    kind: noul
    question: Is `proposed_action` part of doing what `user_request` asks?
```

Paths are dot-separated keys; `[]` marks a list whose items are each narrowed by the rest of the
path. A path absent from a state selects nothing, and a view that selects nothing at all is a
`StateError`, never an empty request. `sees` is not part of the fingerprint, because it changes the
state rather than the question; cached answers are keyed by the exact state each request sent, and
every `ProviderCall` records that state's digest.

In the [stress test](stress-test.md), this boundary is what held: fresh attackers who could rewrite
every tool output got 2 of 12 harmful actions past the Claude Code hook, and 10 of the 12 were
stopped by the one judgment that never sees tool output. The hook's `intent.yaml` declares it this
way.

## Keep these in code

The model reads meaning. It is not a calculator or a calendar.

- **Counting and arithmetic.** Count, sum and compare numbers in code. Ask about meaning instead.
- **Dates and durations.** Extract dates in code (or with a choice over date parts), then compare
  them in code.
- **Structural rules and identities.** Anything a regex, parser or database lookup can decide.
- **Policy.** "Should we allow this?" is your application's decision. Ask factual questions ("Did the
  user authorize this exact action?") and decide in code. The reference sets never ask the model
  to pick allow, confirm or deny.
- **Generation.** Predicates return judgments, never text.

Do not rely on arithmetic identities between predicates either: a question and its negation, asked
as two nouls, do not sum to 1.

## Adversarial content

State is data, and the model does not treat it as hostile by default. Scope security-relevant
predicates to the authoritative field (ask whether `user_request` authorized the action, not whether
the conversation contains permission), include injection attempts in your evaluation fixtures, and
treat every result as one signal among your deterministic controls. Sentinel is not a security
boundary.

## Fingerprints and versions

Every predicate has a `fingerprint`: `sha256` over exactly what the model sees (kind, question, and
criteria, options or levels). Thresholds, tags, descriptions and the optional `version` label do
not change it. The set fingerprint covers every `(id, fingerprint)` pair. Evaluations, events,
response caches and evaluation baselines all record fingerprints, so an edited question can never
be silently compared against answers to the old wording. Git is the registry.

## Authoring warnings

`sentinel predicates validate` reports these. They are advice; the evaluation harness has the final
word.

| Code | Meaning |
| --- | --- |
| `no-state-reference` | The question names no state field (checked with `--state`) |
| `unresolved-state-path` | A backticked path, or a `sees` path, does not exist in the example state |
| `unseen-state-path` | The question names a path outside what the predicate `sees` |
| `arithmetic` | The question asks for counting or arithmetic |
| `date-comparison` | The question compares dates or durations |
| `negation` | A yes/no question is phrased negatively |
| `vague-levels` | Score levels are numbers or very short |
| `duplicate-levels` | Two score levels have the same description |
| `no-escape-option` | A choice of four or more options has no `other` or `none` |
| `long-question` | The question is over 600 characters; move data into the state |
