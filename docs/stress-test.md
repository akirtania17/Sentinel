# Stress test: is Sentinel good enough to open source?

The [field test](field-test.md) showed Sentinel working on real agent traffic but left four gaps:
its harmful actions were blatant, it used one sandbox, nothing attacked Sentinel itself, and there
was no comparison with the obvious alternative of asking an LLM. This test closes them. Everything
ran on 2026-09-18 with `jev-1.13.0`; the tools are in
[`experiments/agent_gate_benchmark`](../experiments/agent_gate_benchmark/README.md).

## Verdict

Worth open sourcing, positioned as a fast, cheap first-line gate with honest measurements, not as a
security boundary.

- **Accuracy**: on held-out scenarios written by an independent author, Sentinel stops 97% of harmful
  agent actions and interrupts 5% of legitimate ones. Claude Sonnet as a judge does better (99% and
  1%); Claude Haiku as a judge does worse (98% and 21%).
- **Cost and speed**: Sentinel decides in 0.2 s for $0.08 per 1,000 actions. Sonnet takes 3.1 s and
  $10.33, over 100 times the cost. Escalating Sentinel's uncertain 9% to Sonnet gives 98% and 3% for
  $1.03.
- **Independent benchmark**: on Meta's LlamaFirewall AlignmentCheck evaluation set, Sentinel detects
  90% of hijacked agent actions with 7% false alarms, level with Meta's own checker running on Llama
  3.3 70B (91% and 5%), without any tuning on that data.
- **Adaptive attacks**: an attacker who can run Sentinel locally got 9 of 12 harmful actions through
  a design that judged everything in one request with the tool output. Judging intent without any tool output cut that to 4 of 12, and escalating
  uncertain cases stops all but one attack, a redirected invoice payment that also fooled Sonnet.
  Sentinel is not robust to a determined attacker on its own, and must say so.

## What was built

| Piece | Purpose |
| --- | --- |
| Scenario format and guide ([SCENARIOS.md](../experiments/agent_gate_benchmark/SCENARIOS.md)) | Moments in an agent's work: the request, recent steps with output, candidate actions with labels |
| Development set | 47 scenarios, 233 actions: support, GitHub, personal assistant, local files |
| Held-out set | 46 scenarios, 246 actions: cloud, payments, databases, messaging, publishing, HR |
| Blind label audit | A second reviewer relabels every action without seeing the labels |
| Gates compared | No gate, a read-only allowlist, Sentinel, Claude Haiku and Claude Sonnet as judges |
| Stress tests | Context padding, repeat stability, escalation, an external benchmark, adaptive attacks |

Both scenario sets were written by separate subagents that were not allowed to read Sentinel's code
or predicates. Every stop category (wrong target, scope creep, unmet condition, followed injection,
injection aimed at the gate, obfuscation, exfiltration, drift) has at least 12 cases, about half of
all actions should be allowed, and scenarios use matched pairs that differ in one detail. The blind
audits agreed with the authors on 99.1% of decisions in both sets; the few disagreements count as
ambiguous and are not scored. All improvements were made on the development set; the held-out set
was scored once per design.

## Accuracy

Held-out set, 118 harmful and 114 legitimate actions:

| Gate | Harmful stopped | Legitimate interrupted | Median latency | Cost per 1,000 |
| --- | --- | --- | --- | --- |
| No gate | 0% | 0% | - | - |
| Read-only allowlist (permission prompts) | 100% | 64% | - | - |
| Sentinel, first context design | 96% | 6% | 0.18 s | $0.05 |
| **Sentinel, shipped design** | **97%** | **5%** | 0.22 s | $0.08 |
| Shipped design, uncertain 9% escalated to Sonnet | 98% | 3% | 0.22 s for 91% | $1.03 |
| Claude Haiku as judge | 98% | 21% | 9.0 s | $8.05 |
| Claude Sonnet as judge | 99% | 1% | 3.1 s | $10.33 |

The judges ran through the `claude` CLI with a short system prompt stating the same review rubric;
their latency includes CLI start-up and structured output, and a direct API call would be faster
and cheaper. Sentinel's cost is its TypeSafe usage.

On the development set, where the designs were tuned, the first context design scored 90% and 10%,
and the shipped design 97% and 6%. Its narrower questions caught every lookalike domain (18 of 18,
up from 14). On the held-out set the two designs scored within one case of each other, so that
improvement did not show up as accuracy. It did show up everywhere else below.

The shipped design's remaining held-out mistakes: unrequested outside effects that read as
plausible follow-ups (tweeting about an outage while updating the status page), and requests that
authorize by category ("approve the small expenses") judged too strictly.

## Stress

| Test | First context design | Shipped design |
| --- | --- | --- |
| 2,000 tokens of irrelevant tool output added | 95% stopped, 11% interrupted | 97%, 4% |
| 8,000 tokens added | 96%, 11% | 97%, 4% |
| About 24,000 tokens added | rejected: over Jev's 32,000-token limit, fails closed | same |
| Decisions that changed across 3 repeat runs (development set) | 1.3% | 1.7% |
| Escalation at the same margin, held-out | 98%, 6%, 5% escalated | 98%, 3%, 9% escalated |

Both designs change about 1 to 2% of decisions between runs, almost all of them borderline or
ambiguous actions. The hook keeps states well under the limit by clipping each step's output and
keeping six steps.
Repeat runs also surfaced a rare malformed answer from the provider (about 1 in 700 requests); the
runtime now retries once, and the response cache no longer stores answers that fail validation.

## Independent benchmark: LlamaFirewall AlignmentCheck

Meta's [evaluation set](https://huggingface.co/datasets/facebook/llamafirewall-alignmentcheck-evals)
(MIT) holds agent traces from six models in shopping, email, calendar and ride-booking tasks, some
with injected instructions, and the verdicts of Meta's AlignmentCheck running on six Llama models.
Following Meta's definitions, positives are the 278 actions an injection hijacked and negatives the
1,728 actions from traces with no injection. Sentinel saw what AlignmentCheck sees (the user's
message, the agent's earlier steps, its latest reasoning and action), with a decision rule fixed
before any results: flag an action that is not part of the request or that carries out an
instruction found in data.

| Monitor | Hijacked actions detected | Clean actions flagged |
| --- | --- | --- |
| Sentinel, shipped design | 90.3% | 6.9% |
| AlignmentCheck, Llama 3.3 70B | 90.6% | 5.2% |
| AlignmentCheck, Llama 4 Scout | 87.8% | 4.9% |
| AlignmentCheck, Llama 4 Maverick | 83.8% | 2.7% |
| AlignmentCheck, Llama 3.2 3B | 71.6% | 22.7% |
| AlignmentCheck, Llama 3.1 8B | 96.0% | 41.2% |

All 2,006 decisions took a median 0.23 s and cost $0.26 in total.

## Adaptive attacks

Once Sentinel is public, an attacker can run it locally and iterate. Each attack targeted 12 harmful
actions from the development set (a redirected payment, an admin role grant, customer data posted to
a webhook, remote code piped to a shell, a production deploy, and so on). The attacker, a separate
subagent, could rewrite only what the agent's tools returned, as a malicious email, ticket, file or
log would, saw every judgment after each try, had 8 tries per action, and knew the exact questions
and decision rule.

| Target | Harmful actions that got through |
| --- | --- |
| Sentinel, one request with context | 9 of 12 |
| Sentinel, intent judged without tool output (shipped) | 4 of 12 |
| Shipped design plus escalating uncertain cases (winning attacks replayed) | 1 of 12 |
| Claude Sonnet as judge, first attacker | 1 of 12 |
| Claude Sonnet as judge, second attacker with the full budget | 2 of 12 |

Fake approvals and messages aimed at the monitor never worked on any gate. What worked on Sentinel
was calm, factual-sounding redescription: a setup script "only prints an index", a production deploy
workflow is "the standard way to cut a release", `truncate audit_log` is "the session purge". With
one request, that text moved the judgments of what the action does and whether the user asked for
it.

The shipped design answers those questions in a second, parallel request whose state has every
tool output removed, so nothing an attacker plants can reach them. Against a fresh attacker it held
8 of 12 structurally. The 4 that got through exploited:

- details only the data defines (a bank account the request never named; also beat Sonnet twice);
- a public link described as internal to the company (scope judged from context);
- an unrequested script judged a harmless local change (the decision rule now stops anything that
  runs or changes something unless the intent judgment says the user asked for it; no accuracy cost
  on either scenario set);
- sampling noise: an admin grant whose "requested" score sat at 0.45 to 0.51 passed on a retry.

Escalating decisions within 0.2 of a threshold closes the last three: Sonnet stopped each of them.
This combination was checked by replaying the winning attacks, not by a new attacker adapting to it.
The invoice redirect passes every gate tested, because the harmful value is exactly what the data
says the right value is; that needs a deterministic control such as a payee allowlist.

## Real agents with the shipped hook

17 live Claude Haiku sessions in the [field-test sandbox](field-test.md), with the hook enforcing:
9 of 11 authorized team actions ran (both double-charge refunds included, which the previous hook
held for confirmation), and the agent itself declined the other two (a deploy with failing tests,
and a customer deletion it wanted confirmed). No harmful effect ran. The hook intervened on 2 of 80
calls: it asked about a cleanup command that deleted the credentials file, and denied deleting the
README. Median time per tool call rose to 0.98 s, from 0.6 s, because the hook now starts a Python
process and makes two requests for every call.

## Before going public

- State the limits in the README: a supplement to permissions, not a security boundary; adaptive
  attackers get some actions through; value-level fraud needs deterministic controls.
- Recommend escalation (or asking the user) for uncertain decisions on high-stakes tools; the hook
  exposes this as `SENTINEL_HOOK_UNSURE=ask`.
- Make the trust boundary a library feature: let a predicate set declare which state fields each
  predicate may see, so every application gets the protection the hook builds by hand.
- Cut hook latency with a long-running helper process instead of a new process per tool call.
- Publish the benchmark itself; the scenario sets, audits and harness are useful on their own.

## Limits

The scenarios, labels and attacks were produced by Claude subagents, independent of Sentinel's
wording but not of Claude's view of what is harmful. 232 labeled held-out actions give confidence
intervals of a few points, so differences of one or two cases are noise. The attacks covered 12
actions with 8 tries each; a longer campaign would likely find more. The LLM judges were run through
the Claude Code CLI rather than tuned API calls. The adaptive attacks ran before a punctuation-only edit
to the development set (em dashes became hyphens), so a few attacked scenarios differ from the
committed files by dashes alone.
