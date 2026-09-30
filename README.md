# prompton-sdk (Python)

The official [PromptOn](https://app.prompton.ai) SDK for Python.

PromptOn is the control plane for your app's LLM prompts. For each **use case** (one place in your
code that calls an LLM) and each environment, it holds one **pin**: a prompt version, one model,
and its parameters. Your app fetches a use-case document for those pins, renders the pinned prompt with this
call's variables, calls the provider **itself, with your own key and your own HTTP client**, and
sends monitoring logs back in batches.

PromptOn is config-fetch, **not a proxy**. It is never in the request path, it never sees your
provider key, and if it goes down your app keeps running on the last use cases it received. This SDK
is deliberately thin: a cache, a template renderer, and a log batcher.

```
prompton.use_case("support_reply")  ──▶  UseCase(model, params, deployment, prompt_names, …)
use_case.messages(vars)             ──▶  rendered chat messages
use_case.track(...)                  ──▶  your provider call, timed and logged
```

## Install

Not published yet, so install from git:

```sh
pip install "prompton-sdk @ git+https://github.com/polimo-dev/prompton-python.git"
```

Once it is on PyPI:

```sh
pip install prompton-sdk
```

Python 3.10 or newer. **No runtime dependencies** - it uses `urllib` from the standard library.

## Quick start

```python
import prompton

prompton.configure(api_key="ptn_myproject_...")  # or set PTN_API_KEY

use_case = prompton.use_case("support_reply")  # which prompt, model and params
messages = use_case.messages({"question": question})


def call():
    answer = openai.chat.completions.create(  # your client, your key
        model=use_case.model, messages=messages, **use_case.params
    )
    return prompton.Result.from_openai(answer)


result = use_case.track(call, variables={"question": question})
print(result.content)
```

`track` times the call, builds the monitoring log (which deployment revision, which
prompt version, which model, how long, how many tokens, what it cost, why it stopped) and queues it.
It returns exactly what your function returned, and re-raises any exception unchanged after
recording it.

Call `prompton.close()` at shutdown so the last batch is sent. A `PromptOn` instance is also a
context manager.

## Configuration

Precedence is always **explicit option > environment variable > default**.

| Option | Environment variable | Default | What it does |
|---|---|---|---|
| `api_key` | `PTN_API_KEY` | — | `ptn_<project>_…`, one per project. Without it the SDK makes no remote calls and works from disk and bundle only, saying so once |
| `host` | `PTN_HOST` | `https://app.prompton.ai` | The SDK appends `/api/v1` itself |
| `environment` | `PTN_ENVIRONMENT` | `production` | Sent as `?environment=` and used as the disk/bundle guard |
| `project` | `PTN_PROJECT` | parsed from the key | Names the disk cache file and guards against a foreign use-case document |
| `timeout` | `PTN_TIMEOUT` | `5.0` | Per-request timeout for non-config API calls. Config fetches are capped at 1 second |
| `cache_ttl` | `PTN_CACHE_TTL` | `10.0` | Deprecated compatibility option. Normal config fetch freshness and attempt gating are fixed at 10 seconds per use-case key |
| `poll` | `PTN_POLL` | `false` | Deprecated compatibility option. Normal runtime config fetches do not poll in the background |
| `disk_cache` | `PTN_DISK_CACHE` | on | `True`, `False`, or a path. Default is `<os cache dir>/prompton/<project>-<environment>.json` |
| `bundle` | `PTN_BUNDLE` | — | A use-case JSON shipped inside the app, used when memory and disk are empty |
| `mode` | `PTN_MODE` | `live` | `test` (no HTTP, no disk cache or bundle, logs captured) or `offline` (disk and bundle only) |
| `hash_end_user` | `PTN_HASH_END_USER` | `false` | Send `sha256(end_user_ref)` instead of the raw value |
| `redact` | — | — | `fn(record) -> record`, applied last to every monitoring log |
| `flush_interval` | `PTN_FLUSH_INTERVAL` | `2.0` | Time trigger for a batch, seconds |
| `flush_size` | `PTN_FLUSH_SIZE` | `100` | Size trigger for a batch |
| `flush_bytes` | `PTN_FLUSH_BYTES` | `1000000` | Byte trigger for a batch |
| `max_queue` | `PTN_MAX_QUEUE` | `10000` | Queue cap; over it the oldest records are dropped and counted |
| `max_send_attempts` | `PTN_MAX_SEND_ATTEMPTS` | `8` | Retries per batch before it is dropped and counted |
| `max_backoff` | `PTN_MAX_BACKOFF` | `300.0` | Backoff ceiling, seconds |

```python
from prompton import PromptOn

client = PromptOn(
    api_key=os.environ["PTN_API_KEY"],
    environment="staging",
    disk_cache="/var/lib/myapp/prompton.json",
    bundle="myapp/prompton/use-cases.production.json",
    redact=lambda record: record,
)
```

## Resilience: what happens when PromptOn is down

The use-case document lives in three local tiers - **memory, one local file, a bundled file** - and
nothing else. The SDK never requires or optionally integrates a database, Redis, or any other shared
store. Several processes on one host may share the disk file: writes are atomic (tmp + rename),
readers tolerate a concurrent rename, and a corrupt or partial file is ignored rather than raised.

Runtime config fetch is demand-driven per use-case key:

* Startup and idle periods load only local files. They do not fetch remote config and do not poll.
* `client.use_case("support_reply")` checks that key's cache. If it is fresh within the fixed 10-second config freshness window, no HTTP call is made.
* If the key is missing or stale, the SDK tries one
  `GET /api/v1/prompts/support_reply?environment=...` request with that key's ETag when available.
  Concurrent calls for the same key share the same in-flight request; different keys are independent.
* The same fixed 10-second window is also the attempt gate. A failed attempt counts, so the SDK will not retry
  that key again until the TTL has elapsed. There is no config retry loop.
* Each config fetch has a one-second total deadline, including the response body. If PromptOn cannot
  answer in that budget, the SDK immediately serves the last valid value for that key, even if it is
  expired. With no cached value for that key, it raises `UseCaseDocumentUnavailableError`.
* A document for **another environment or project is never used**. The file records both; a mismatch
  is ignored with a warning, and if it leaves the client with nothing the error names both sides
  rather than blaming the network.
* Prefork servers keep log sending safe after `fork()`. Config polling is disabled, so there is no
  config poll thread to restart in children.

### How it fails

| Situation | What your call sees |
|---|---|
| Use-case cache is fresh | Served from memory, no HTTP call |
| Use-case cache is stale or missing | The resolved use case after one keyed fetch, or the cached fallback if the fetch fails |
| Same use case requested concurrently | The callers share one in-flight config fetch |
| PromptOn returns `304 Not Modified` | The previous document for that key |
| PromptOn returns `429` / `5xx`, times out, or is unreachable | The previous document for that key, and no retry for that key until 10 seconds elapse |
| PromptOn unreachable, disk cache present for that key | The disk document, `source="disk"` |
| PromptOn unreachable, only a bundle present for that key | The bundled document, `source="bundle"` |
| PromptOn unreachable and **nothing cached for that key** | `UseCaseDocumentUnavailableError` saying exactly that |
| A use-case document arrives for another project or environment | It is ignored; if nothing else is cached, the error names both sides rather than blaming the network |
| `filled_prompt()` in `mode="test"`, `mode="offline"`, or with no key | No request is made: `ConfigurationError` in test mode, otherwise the cached answer, or `UseCaseDocumentUnavailableError` |

## The monitoring log record

`track` fills all of this in. `log()` lets you build it yourself - for a streaming call,
or a background job that does not wrap the provider call.

| Field | Notes |
|---|---|
| `id` | UUIDv7 generated by the SDK before the call; the idempotency key. A resend is counted as a duplicate, never stored twice |
| `use_case`, `model`, `status`, `started_at` | Required. `status` is `ok` or `error` |
| `kind` | `chat`, `text` or `embedding` |
| `deployment_id`, `deployment_revision`, `prompt`, `prompt_version_id`, `model_id` | The use-case evidence: which pin produced this call |
| `source` | `remote`, `disk`, `bundle` or `manual` |
| `provider`, `model_used`, `upstream_provider` | Who actually served it |
| `params` | The effective params, plus any per-call override you passed |
| `input` | `{"variables", "messages"}` or `{"text"}` |
| `output` | `{"content", "tool_calls"}` |
| `finish_reason`, `stop_kind` | `stop_kind` is `stop`, `length`, `tool_call`, `content_filter` or `other`, derived from `finish_reason` when absent |
| `error` | On `status: "error"`: `kind` (`http_4xx`, `http_5xx`, `rate_limited`, `timeout`, `transport`, `parse`, `app`), `status`, `message` |
| `usage` | `input_tokens`, `output_tokens`, `cost_usd`, `cost_source`, `raw` |
| `latency_ms`, `trace_id`, `sequence`, `end_user_ref` | Correlation |
| `context`, `metadata` | Free-form. Keep them small: over 2 KB and 4 KB the server rejects the record |
| `sdk` | `{"name": "prompton-python", "version": "0.2.0"}` |

Before sending, the SDK applies the use case's payload policy: sampling (deterministic on the id,
with errors and truncated answers always kept), `hash` or `none` modes, and truncation to the
contract's caps, UTF-8 safe. `redact` runs last.

### Errors from your provider

```python
from prompton import Result, ProviderError


def call():
    response = requests.post(url, json=body, timeout=60)
    if response.status_code == 429:
        raise ProviderError("rate limited", kind="rate_limited", status=429)
    data = response.json()
    result = Result(
        content=data["choices"][0]["message"]["content"],
        finish_reason=data["choices"][0]["finish_reason"],
        input_tokens=data["usage"]["prompt_tokens"],
        output_tokens=data["usage"]["completion_tokens"],
    )
    try:
        result.result = json.loads(result.content)
    except ValueError as error:
        # the provider answered, so keep the usage and the text as a quality signal
        raise ProviderError(str(error), kind="parse", result=result) from error
    return result
```

## Other entry points

```python
client.prompt_names("support_reply")  # ["default", "ko"] - exactly what use_case() accepts
client.filled_prompt("support_reply", variables={...})  # prompt endpoint smoke test
client.refresh(key="support_reply")  # fetch one key now, subject to the 10 s attempt gate
client.export_use_cases("app/prompton/use-cases.production.json")  # build a bundle
client.use_cases_info()  # {"source", "etag", "age_seconds", "stale", ...}
client.log(record)  # a record you built yourself
client.flush()  # send the queue now and wait, including the batch already on the wire
client.stats  # enqueued / sent / accepted / duplicates / dropped_*
```

`flush()` waits for everything the buffer still holds - the queue, a batch waiting out a retry, and
the request in flight - so the counters it returns describe what actually happened; `close(timeout)`
spends whatever the flush leaves on finishing that last batch. `stats.queued` counts all three, and
`stats.batches_sent` counts only batches the server accepted.

Keyed `refresh(key=...)` follows the same one-attempt-per-10-seconds gate as `use_case()`.
`refresh(force=True, key=...)` is the explicit escape hatch for tools that really mean now.

From the command line, for CI:

```sh
python -m prompton export --out app/prompton/use-cases.production.json
prompton-use-cases export --out app/prompton/use-cases.production.json
```

## Testing your app

```python
from prompton import PromptOn
from prompton.testing import make_use_case_document

client = PromptOn(mode="test")  # no HTTP at all
client.load_use_cases(
    make_use_case_document(
        support_reply={
            "model": "openai/gpt-4o-mini",
            "messages": [{"role": "user", "content": "Answer: {{ question }}"}],
            "params": {"temperature": 0.2},
        },
    )
)

use_case = client.use_case("support_reply")
...
assert client.captured[0]["status"] == "ok"
```

A test-mode client starts **empty**: it reads neither the machine's disk cache nor a bundle, so it
behaves the same on a laptop with a warm cache as it does in CI, and `load_use_cases` is the only way
to put a document in it. It also makes no HTTP call at all - `filled_prompt()` raises
`ConfigurationError` rather than quietly reaching the network.

`mode="offline"` is the other half: real lookup from the disk cache or the bundle, still no
network - useful in CI and on a plane.

## Conformance

`tests/conformance/` is the cross-language contract, copied from the reference implementation:
template rendering, use-case lookup, monitoring-log truncation, `stop_kind` normalisation and golden
records. Every case runs in this SDK's test suite, so a Python service and a Go service select the
same prompt to the same bytes.

## License

Copyright 2026 Polimo

Licensed under the Apache License, Version 2.0 - see [LICENSE](LICENSE).

PromptOn is a trademark of Polimo. The license does not grant permission to use the PromptOn name or
logo; forks and derived services must use a different name.


## Trace events

Use `log_events()` when your app has already observed tool calls or completion events and wants them available for eval evidence. The SDK does not execute tools and does not infer these events from provider requests. In live mode it immediately posts `{"logs": [], "events": [...]}` to the logs endpoint; in test mode the submitted events are available on `client.captured_events`.

```python
client.log_events(
    [
        {
            "event_id": "evt_1",
            "trace_id": "trace_1",
            "event_kind": "tool_attempt",
            "status": "ok",
            "observed_at": "2026-09-28T00:00:00Z",
            "tool_call_id": "call_1",
            "tool_name": "search_diary",
            "arguments": {"query": "Ada"},
            "result": {"matches": []},
        }
    ]
)
```
