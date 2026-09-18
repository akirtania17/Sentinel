# Recorded TypeSafe HTTP responses

Real `POST /v1/systemone` responses captured on 2026-09-18 against `jev-1.13.0`, used by the
contract tests through an injected mock transport. Each file holds the status, a minimal header
subset, and the JSON body. Request ids are replaced with readable placeholders; no request headers
or credentials are stored.

| File | How it was produced |
| --- | --- |
| `ok_agent_action.json` | The `agent_action` reference set on the "summarize emails / send email" state |
| `error_400_max_tokens.json` | A state of about 315,000 characters |
| `error_400_unknown_model.json` | `model: jev-9.9.9` |
| `error_400_string_detail.json` | A choice question with empty criteria |
| `error_422_echoes_state.json` | A request without `model`; the body echoes the request, including a marker state |
| `error_401.json` | An invalid API key |
| `synthetic_429.json` | **Synthetic.** Shaped after the documented error format; rate limits are not triggered on purpose |
| `synthetic_529.json` | **Synthetic.** Same reason |
