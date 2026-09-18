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
- **A second, larger held-out set** (520 actions, 8 new domains) reproduced this: Sentinel 95% and
  7% alone, 98% and 4% with escalation, Sonnet 99% and 3%.
- **AgentDojo**: replaying GPT-4o's recorded runs, Sentinel cut successful prompt injections from
  47.7% to 3.8%, better than the benchmark's published tool filter (6.8%), but it interrupted about
  a fifth of the clean tasks the agent had completed, where the tool filter lost none.
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

## Second round: prompting changes, multi-turn, a new held-out set, AgentDojo

A second pass tested the improvements the first round suggested, on data the first round never
touched. The two original scenario sets were merged into one tuning set of 479 actions, a new
multi-turn tuning set added 133 actions, and two new independent authors wrote a new held-out set
of 520 actions in eight new domains (healthcare administration, e-commerce, legal, education,
security operations, marketing, smart home and facilities, mobile releases), with 22 multi-turn
scenarios. Blind audits agreed with the authors on 98.5% and 97.0% of decisions.

### What did not help

Following TypeSafe's question-writing guidance literally made Sentinel worse on the tuning set:

| Change | Harmful stopped | Legitimate interrupted | Balanced accuracy |
| --- | --- | --- | --- |
| Shipped questions | 96% | 7% | 0.949 |
| Narrower questions, structured as JSON, state with trust labels | 95% | 10% | 0.925 |
| ... without the new project-rule question | 95% | 8% | 0.934 |
| ... plus each key question asked in three wordings and averaged | 95% | 8% | 0.934 |

Splitting one broad check into several narrow ones multiplied the chances for a false alarm,
because any one of them stops the action; a question about project rules fired whenever the
instructions described a procedure the action was following. Averaging wordings did not reduce
errors. Explicit multi-turn wording also did worse on the multi-turn set (93% and 8%) than the
shipped questions with the earlier turns simply added to the state (96% and 5%).

### What did help

- **Earlier turns in the state.** The hook now passes the user's and assistant's earlier turns, so a
  short "yes, go ahead" can approve what the assistant proposed.
- **A lookalike check in code** (`examples/claude_code/checks.py`) compares addresses, domains,
  account numbers and ids in the action with those the user named. It changed no decision on either
  scenario set, because Jev already caught those cases, and flagged no legitimate action; it is kept
  because an attacker cannot talk code out of noticing a swapped digit.
- **A threshold preset.** Thresholds chosen on one half of the tuning set and checked on the other
  (then the reverse) trade a little detection for far fewer interruptions:
  `SENTINEL_HOOK_THRESHOLDS=balanced`. `strict` stays the default.
- **Asking instead of denying** when an action looks unrequested but shows no sign of an injected
  instruction or data leaving, so a person can approve work the gate misread.

### New held-out set

520 actions, 234 harmful and 249 legitimate (37 ambiguous, unscored), scored once:

| Gate | Harmful stopped | Legitimate interrupted | Median latency | Cost per 1,000 |
| --- | --- | --- | --- | --- |
| Sentinel, strict | 95% | 7% | 0.2 s | $0.08 |
| Sentinel, balanced | 94% | 3% | 0.2 s | $0.08 |
| Sentinel strict, unsure 10% escalated to Sonnet | 98% | 4% | 0.2 s for 90% | $1.17 |
| Claude Haiku as judge | 98% | 23% | 9.5 s | $8.39 |
| Claude Sonnet as judge | 99% | 3% | 3.4 s | $10.42 |
| Read-only allowlist | 100% | 70% | - | - |

On the 22 multi-turn scenarios, Sentinel strict stopped 95% and interrupted 2%, Sonnet 98% and 0%.
The result reproduces the first held-out set: alone, Sentinel is a few points behind Sonnet; with
its uncertain tenth escalated, it comes within a point of Sonnet at about a ninth of the cost.

### AgentDojo

[AgentDojo](https://github.com/ethz-spylab/agentdojo) (MIT) is the standard benchmark for prompt
injection against tool-using agents, and publishes full recorded runs. `agentdojo.py` replays every
tool call of an agent's recorded runs through Sentinel, offline, and assumes that a stop at or before
the attacker's first harmful call prevents the attack and that any stop in a clean run costs that
task. The published defenses were measured live, so the comparison is indicative, not exact.

GPT-4o, AgentDojo's standard "important instructions" attack (629 attacked runs, 97 clean tasks):

| Defense | Attacks that succeed | Clean tasks completed |
| --- | --- | --- |
| None | 47.7% | 69.1% |
| Sentinel, strict (offline) | 3.8% | 53.6% |
| Sentinel, balanced (offline) | 3.8% | 56.7% |
| AgentDojo tool filter (published run) | 6.8% | 72.2% |
| AgentDojo prompt-injection detector (published run) | 7.9% | 41.2% |
| Spotlighting with delimiters (published run) | 41.7% | 72.2% |

Of the 276 attacks Sentinel stopped, 267 were stopped at the attacker's own call. With Claude 3.5
Sonnet's runs, where attacks already succeed only 1.1% of the time, Sentinel lowered that to 0.6%
and completed 67 of the agent's 77 successful clean tasks (70 with `balanced`).

The cost is the weakness. Sentinel denied 10 to 11 of GPT-4o's successful clean tasks. Seven of the
denials across both agents were tasks that tell the agent to carry out instructions from a named
email, file or to-do list ("do all the tasks on my TODO list at ..."). Sentinel's rule that data
never authorizes anything refuses those by design; honoring such delegation means an attacker who
can write into the delegated source is obeyed. The other denials came from the "forbidden or
condition not met" and "data leaves" checks misfiring.

## Before going public

- State the limits in the README: a supplement to permissions, not a security boundary; adaptive
  attackers get some actions through; value-level fraud needs deterministic controls; delegation to
  instructions in data is refused by design.
- Recommend escalation (or asking the user) for uncertain decisions on high-stakes tools; the hook
  exposes this as `SENTINEL_HOOK_UNSURE=ask`.
- Decide how to support delegation ("follow the instructions in this file"), for example by asking
  the user once per delegated source, and measure it on AgentDojo's delegating tasks.
- Tighten the "forbidden or condition not met" check, the largest source of false denials on
  AgentDojo, and re-measure on a fresh held-out set.
- Make the trust boundary a library feature: let a predicate set declare which state fields each
  predicate may see, so every application gets the protection the hook builds by hand.
- Cut hook latency with a long-running helper process instead of a new process per tool call.
- Get human labels on a sample of the scenarios; every label so far comes from Claude.
- Run AgentDojo live with Sentinel in the loop, which settles what happens after a stop.
- Publish the benchmark itself; the scenario sets, audits and harness are useful on their own.

## Limits

The scenarios, labels and attacks were produced by Claude subagents, independent of Sentinel's
wording but not of Claude's view of what is harmful. 232 labeled held-out actions give confidence
intervals of a few points, so differences of one or two cases are noise. The attacks covered 12
actions with 8 tries each; a longer campaign would likely find more. The LLM judges were run through
the Claude Code CLI rather than tuned API calls. The adaptive attacks ran before a punctuation-only edit
to the development set (em dashes became hyphens), so a few attacked scenarios differ from the
committed files by dashes alone.
