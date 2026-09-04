# prompton-sdk (Python)

The official [PromptOn](https://app.prompton.ai) SDK for Python.

PromptOn is the control plane for your app's LLM prompts. For each **use case** (one place in your
code that calls an LLM) and each environment, it holds one **pin**: a prompt version, one model,
and its parameters. Your app fetches a snapshot of those pins, renders the pinned prompt with this
call's variables, calls the provider **itself, with your own key and your own HTTP client**, and
sends monitoring logs back in batches.

PromptOn is config-fetch, **not a proxy**. It is never in the request path, it never sees your
provider key, and if it goes down your app keeps running on the last snapshot it received. This SDK
is deliberately thin: a cache, a template renderer, and a log batcher.

```
resolve("support_reply")  ──▶  Resolution(model, params, messages, deployment_id, prompt, …)
render(resolution, vars)  ──▶  messages
with_generation(...)      ──▶  your provider call, timed and logged
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

resolution = prompton.resolve("support_reply")  # which prompt, model and params
messages = prompton.render(resolution, {"question": question})


def call():
    answer = openai.chat.completions.create(  # your client, your key
        model=resolution.model, messages=messages, **resolution.effective_params
    )
    choice = answer.choices[0]
    return prompton.Outcome(
        content=choice.message.content,
        finish_reason=choice.finish_reason,
        input_tokens=answer.usage.prompt_tokens,
        output_tokens=answer.usage.completion_tokens,
    )


outcome = prompton.with_generation(resolution, call, variables={"question": question})
print(outcome.content)
```

`with_generation` times the call, builds the monitoring log (which deployment revision, which
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
| `project` | `PTN_PROJECT` | parsed from the key | Names the disk cache file and guards against a foreign snapshot |
| `timeout` | `PTN_TIMEOUT` | `5.0` | Per-request timeout, seconds |
| `cache_ttl` | `PTN_CACHE_TTL` | `10.0` | How long a snapshot is served from memory with no HTTP call |
| `poll` | `PTN_POLL` | `true` | Refresh in a background thread. `false` refreshes on the next call instead (stale-while-revalidate) |
| `disk_cache` | `PTN_DISK_CACHE` | on | `True`, `False`, or a path. Default is `<os cache dir>/prompton/<project>-<environment>.json` |
| `bundle` | `PTN_BUNDLE` | — | A snapshot JSON shipped inside the app, used when memory and disk are empty |
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
    bundle="myapp/prompton/snapshot.production.json",
    redact=lambda record: record,
)
```

## Resilience: what happens when PromptOn is down

The snapshot lives in three tiers - **memory, one local file, a bundled file** - and nothing else.
The SDK never requires or optionally integrates a database, Redis, or any other shared store.
Instances never coordinate; each keeps its own copy, which ETag polling makes cheap. Several
processes on one host may share the disk file: writes are atomic (tmp + rename), readers tolerate a
concurrent rename, and a corrupt or partial file is ignored rather than raised.

* Within `cache_ttl` (10 seconds) every `resolve` is answered from memory with **no HTTP call**.
* Past the TTL the SDK refreshes with `If-None-Match`; a `304` costs nothing. The refresh happens in
  the background - a poll thread, or a stale-while-revalidate refresh triggered by the next call.
  **A refresh never blocks or fails a generation**: while it is in flight, and if it fails, the
  previous document is used.
* On `429` the SDK reads `Retry-After` (falling back to `error.details.retry_after`, then to
  backoff) and does not contact the server again before it has elapsed. On `5xx`, timeouts and
  transport errors it backs off ×2 from the TTL up to five minutes. The caller sees none of it.
* On start the load order is **memory → disk → bundle → remote**, and the tier that answered is
  reported as `resolution_source` on the `Resolution` and on every monitoring log.
* A document for **another environment or project is never used**. The file records both; a mismatch
  is ignored with a warning, and if it leaves the client with nothing the error says so - the
  server answered, it just answered for someone else's project.
* **Prefork servers work.** Threads do not survive `fork()`, so a client built before gunicorn
  `--preload` or uWSGI forks would otherwise never refresh again. The SDK restarts its poll thread
  and its log sender in the child, and falls back to stale-while-revalidate if it cannot.

### How it fails

| Situation | What your call sees |
|---|---|
| Snapshot is fresh | Served from memory, no HTTP call |
| Snapshot is stale, refresh in flight | The previous document, immediately |
| PromptOn returns 429 / 5xx, or is unreachable | The previous document; the SDK backs off quietly |
| PromptOn unreachable, disk cache present | The disk document, `resolution_source="disk"` |
| PromptOn unreachable, only a bundle present | The bundled document, `resolution_source="bundle"` |
| PromptOn unreachable and **nothing cached** | `SnapshotUnavailableError` saying exactly that |
| A snapshot arrives for another project or environment | It is ignored; if nothing else is cached, the error names both sides rather than blaming the network |
| `resolve_remote()` in `mode="test"`, `mode="offline"`, or with no key | No request is made: `ConfigurationError` in test mode, otherwise the cached answer, or `SnapshotUnavailableError` |
| Use case key not in the snapshot | `UnknownUseCaseError` - a bug in the app |
| Use case has no live deployment | `UnresolvedError` - deploy it; never a silent fallback |
| Prompt name not pinned by the live revision | `UnknownPromptError` with `available_prompts`; never falls back to `default` |
| A template variable was not supplied | `MissingVariableError` naming the variable |
| Monitoring log queue full | The oldest records are dropped and counted in `client.stats` |
| No API key, or `mode="offline"` | Nothing is sent; records are dropped and counted as `dropped_no_remote`, with one log line |
| `/generations` returns 429 or 5xx | The same batch, with the same ids, is retried; duplicates are absorbed server-side |
| `/generations` returns 413 | The batch is split in half and resent |
| `/generations` returns another 4xx | The batch is dropped, counted, and logged once - a rejected record only gets rejected again |

Nothing in the monitoring-log path can raise into your request. The one exception is `log()` with a
required field missing, which is a bug in the calling code and raises `ValueError` straight away.

## The monitoring log record

`with_generation` fills all of this in. `log()` lets you build it yourself - for a streaming call,
or a background job that does not wrap the provider call.

| Field | Notes |
|---|---|
| `id` | UUIDv7 generated by the SDK before the call; the idempotency key. A resend is counted as a duplicate, never stored twice |
| `use_case`, `model`, `status`, `started_at` | Required. `status` is `ok` or `error` |
| `kind` | `chat`, `text` or `embedding` |
| `deployment_id`, `deployment_revision`, `prompt`, `prompt_version_id`, `model_id` | The resolution evidence: which pin produced this call |
| `resolution_source` | `remote`, `disk`, `bundle` or `manual` |
| `provider`, `model_used`, `upstream_provider` | Who actually served it |
| `params` | The effective params, plus any per-call override you passed |
| `input` | `{"variables", "messages"}` or `{"text"}` |
| `output` | `{"content", "tool_calls"}` |
| `finish_reason`, `stop_kind` | `stop_kind` is `stop`, `length`, `tool_call`, `content_filter` or `other`, derived from `finish_reason` when absent |
| `error` | On `status: "error"`: `kind` (`http_4xx`, `http_5xx`, `rate_limited`, `timeout`, `transport`, `parse`, `app`), `status`, `message` |
| `usage` | `input_tokens`, `output_tokens`, `cost_usd`, `cost_source`, `raw` |
| `latency_ms`, `trace_id`, `sequence`, `end_user_ref` | Correlation |
| `context`, `metadata` | Free-form. Keep them small: over 2 KB and 4 KB the server rejects the record |
| `sdk` | `{"name": "prompton-python", "version": "0.1.0"}` |

Before sending, the SDK applies the use case's payload policy: sampling (deterministic on the id,
with errors and truncated answers always kept), `hash` or `none` modes, and truncation to the
contract's caps, UTF-8 safe. `redact` runs last.

### Errors from your provider

```python
from prompton import Outcome, ProviderError


def call():
    response = requests.post(url, json=body, timeout=60)
    if response.status_code == 429:
        raise ProviderError("rate limited", kind="rate_limited", status=429)
    data = response.json()
    outcome = Outcome(
        content=data["choices"][0]["message"]["content"],
        finish_reason=data["choices"][0]["finish_reason"],
        input_tokens=data["usage"]["prompt_tokens"],
        output_tokens=data["usage"]["completion_tokens"],
    )
    try:
        outcome.result = json.loads(outcome.content)
    except ValueError as error:
        # the provider answered, so keep the usage and the text as a quality signal
        raise ProviderError(str(error), kind="parse", outcome=outcome) from error
    return outcome
```

## Other entry points

```python
client.prompt_names("support_reply")  # ["default", "ko"] - exactly what resolve() accepts
client.resolve_remote("support_reply", variables={...})  # POST /resolve: the smoke test
client.refresh()  # fetch once, now, synchronously (refresh(force=True) ignores a Retry-After pause)
client.export_snapshot("app/prompton/snapshot.production.json")  # build a bundle
client.snapshot_info()  # {"source", "etag", "age_seconds", "stale", ...}
client.log(record)  # a record you built yourself
client.flush()  # send the queue now and wait, including the batch already on the wire
client.stats  # enqueued / sent / accepted / duplicates / dropped_*
```

`flush()` waits for everything the buffer still holds - the queue, a batch waiting out a retry, and
the request in flight - so the counters it returns describe what actually happened; `close(timeout)`
spends whatever the flush leaves on finishing that last batch. `stats.queued` counts all three, and
`stats.batches_sent` counts only batches the server accepted.

`refresh()` respects an active `Retry-After`: calling it in a readiness-probe loop cannot become the
thing that keeps a rate-limited server busy. Pass `force=True` when you really mean now.

From the command line, for CI:

```sh
python -m prompton export --out app/prompton/snapshot.production.json
```

## Testing your app

```python
from prompton import PromptOn
from prompton.testing import make_snapshot

client = PromptOn(mode="test")  # no HTTP at all
client.load_snapshot(
    make_snapshot(
        support_reply={
            "model": "openai/gpt-4o-mini",
            "messages": [{"role": "user", "content": "Answer: {{ question }}"}],
            "params": {"temperature": 0.2},
        },
    )
)

resolution = client.resolve("support_reply")
...
assert client.captured[0]["status"] == "ok"
```

A test-mode client starts **empty**: it reads neither the machine's disk cache nor a bundle, so it
behaves the same on a laptop with a warm cache as it does in CI, and `load_snapshot` is the only way
to put a document in it. It also makes no HTTP call at all - `resolve_remote()` raises
`ConfigurationError` rather than quietly reaching the network.

`mode="offline"` is the other half: real resolution from the disk cache or the bundle, still no
network - useful in CI and on a plane.

## Conformance

`tests/conformance/` is the cross-language contract, copied from the reference implementation:
template rendering, resolution, monitoring-log truncation, `stop_kind` normalisation and golden
records. Every case runs in this SDK's test suite, so a Python service and a Go service resolve the
same prompt to the same bytes.

## License

Copyright 2026 Polimo

Licensed under the Apache License, Version 2.0 - see [LICENSE](LICENSE).

PromptOn is a trademark of Polimo. The license does not grant permission to use the PromptOn name or
logo; forks and derived services must use a different name.
