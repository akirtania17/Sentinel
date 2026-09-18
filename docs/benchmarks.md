# Batching benchmarks

Sentinel sends every predicate that judges the same state in one request. This page measures what
that saves. `sentinel bench` evaluates one state three ways, several times each:

- **batched**: all predicates in one request (the runtime's default);
- **sequential**: one request per predicate, one after another;
- **concurrent**: one request per predicate, several at a time.

It reports median wall-clock latency, input tokens, estimated cost, the largest difference between
batched and single-predicate answers, and batched throughput. Run it live: replayed latencies are
not measurements.

```bash
sentinel bench examples/bench/handbook_questions.yaml examples/bench/handbook.json --model jev-1.13.0 --concurrency 6
sentinel bench agent_action examples/agent_action/states/bad.json --model jev-1.13.0 --concurrency 5
```

## Results (2026-09-18, jev-1.13.0, 3 repeats)

**Twelve questions about a 3,100-character employee handbook** (`examples/bench`):

| Strategy | Requests | Median latency | Input tokens | Estimated cost |
| --- | --- | --- | --- | --- |
| batched | 1 | 190 ms | 1,165 | $0.000049 |
| sequential | 12 | 2,143 ms | 11,021 | $0.000463 |
| concurrent | 12 | 685 ms | 11,021 | $0.000463 |

One request per predicate cost 9.5 times as much and took 11.3 times as long when sent in sequence.
Every batched answer was within 0.01 of the single-predicate answer, and all twelve matched the
handbook. Throughput was about 25 batched evaluations per second at a concurrency of 6.

**Five agent_action predicates about a 229-character tool call**:

| Strategy | Requests | Median latency | Input tokens | Estimated cost |
| --- | --- | --- | --- | --- |
| batched | 1 | 179 ms | 897 | $0.000038 |
| sequential | 5 | 838 ms | 2,237 | $0.000094 |
| concurrent | 5 | 227 ms | 2,237 | $0.000094 |

Here separate requests cost 2.5 times as much. Noul and choice answers were identical; the score
differed by 0.07, inside its measured repeat noise.

## Reading the numbers

- **Cost scales with the state, not the number of predicates.** Each separate request re-sends the
  state, so the saving approaches the number of predicates as the state grows: 9.5x for twelve
  questions over a document, 2.5x for five questions over a short tool call.
- **Concurrency hides latency, not cost.** Concurrent single requests finish much sooner than
  sequential ones but pay for the state every time.
- **Batching does not change answers.** Questions in one request are evaluated independently; the
  differences above are the same size as repeat noise (see [evals](evals.md#repeat-stability)).
- **Prices change.** Costs come from the dated table in `sentinel_runtime.pricing`
  (`jev-1.13.0`: $0.042 per million input tokens, output free, as of 2026-09-17), which you can
  override in `sentinel.toml`. Evaluations record token counts so cost can be recomputed later.
