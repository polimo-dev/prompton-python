# Changelog

## Unreleased

- Suppress the exact closed Req connection transport failure from generation logs and matching
  error completion trace events before they reach redaction, test capture, buffering, or HTTP.
- Retired message-slot expansion. A prompt message with `{"type": "slot"}` now raises
  `Message slots are not supported; compose conversation history in app code.` even when variables
  include a matching list; `{{ history }}` remains an ordinary template variable.
- Updated examples and tests so apps compose PromptOn-managed messages, app-owned conversation
  history and the current user message before calling the provider, then pass those final messages
  to `track(input_messages=...)`.

## 0.5.0

- Changed normal config fetch to be demand-driven per use-case/prompt key. Startup and idle
  periods no longer fetch or poll PromptOn.
- `use_case(key)` fetches `GET /api/v1/prompts/:key?environment=...` only when that key has no
  fresh cache, shares same-key concurrent fetches, and keeps prompt keys cached independently.
- Config fetches have a 10-second freshness/attempt gate, a 1-second total fetch deadline, no
  retry, per-key ETags, and stale fallback to the last valid disk, bundle, manual, or remote value.

## 0.4.1

- Fixed runtime compatibility with the current PromptOn prompt API: snapshot fetches now use `GET /api/v1/prompts`, remote render uses `POST /api/v1/prompts/{key}/render` with `template`, and monitoring logs use canonical `prompt_key`/`template` fields.
- Updated active conformance fixtures to the schema7 prompt contract with native tool messages while keeping existing public use-case aliases.

All notable changes to `prompton-sdk` for Python. This project follows
[semantic versioning](https://semver.org/).

## 0.4.0

- Added schema-v7 prompt documents with chat tool definitions and message slots. Full provider chat messages, including `content: None`, array content, `tool_calls`, `tool_call_id` and native extra fields, are preserved when slot histories are spliced.
- Prepared runtime evidence now carries deployment `api`, `request_path` and prompt `tools` metadata for provider request construction by applications.
- Added `PromptOn.log_events()` and module-level `prompton.log_events()` for application-observed trace events. It posts `{"logs": [], "events": [...]}` to `/api/v1/logs?environment=...` and captures events in test mode.

## 0.2.0

Breaking rename to the final PromptOn runtime vocabulary.

- Replaced the public `resolve()` / `render()` flow with `use_case()` returning `UseCase`, whose
  `.messages()` and `.text()` methods validate the use-case kind before rendering.
- Replaced `Outcome` with `Result`, including `Result.from_openai()` and
  `Result.from_anthropic()` helpers for provider response extraction.
- Replaced public generation tracking with `UseCase.track(call, ...)` and the context-manager form
  `with use_case.track(...) as log: ... log.result(result)`.
- Updated runtime HTTP paths to `GET /api/v1/use-cases`, `POST /api/v1/use-cases/{key}/prompt`,
  and `POST /api/v1/logs` with the `{"logs": [...]}` envelope.
- Updated public fields to `params`, `provider_options` and `source`, schema version 4, and the
  default bundle filename `use-cases.production.json`.
- Renamed conformance fixtures to `use_case.json` and `log_record.json`.

## 0.1.0

Initial release: the PromptOn runtime contract for Python, with no runtime dependencies.

- **Use case lookup** a use case to its pinned prompt version, model and parameters, entirely locally from
  a cached use-case document. Prompt name is the only selection axis, and an unpinned name is an error
  rather than a silent fall back to `default`.
- **Rendering** the pinned prompt with the Liquid subset PromptOn allows - `for`, `if`/`elsif`/`else`,
  `unless`, `assign`, and the `size`, `join` and `default` filters - including Liquid's blank-body
  rule and the missing-variable semantics the contract specifies. Plus `lint` and `variables_of`.
- **Use-case document store** with the mandated caching rules: a 10-second memory cache, `If-None-Match`
  revalidation in a poll thread or stale-while-revalidate, `Retry-After` on 429, ×2 backoff to five
  minutes on 5xx and transport failures, and three tiers - memory, an atomically written disk cache
  with a sidecar, and an optional committed bundle. No database, no Redis, no coordination between
  instances. A document for another environment or project is never used, and the error says which
  project answered rather than blaming the network. `refresh()` honours an active `Retry-After`
  too; `refresh(force=True)` is the escape hatch the CLI uses.
- **Monitoring logs**: `log()`, `flush()` and the `track()` wrapper (plus a `track()`
  context manager), backed by a batching buffer with app-generated UUIDv7 ids, size/time/byte flush
  triggers, batches of at most 200 records and 5 MB, partial-acceptance handling, retries of the
  same batch with the same ids on 429 and 5xx, 413 splitting, a bounded drop-oldest queue, a
  redaction hook, `hash_end_user`, and flush on shutdown. `flush()` waits for the batch already on
  the wire as well as for the queue, and `close(timeout)` spends the rest of that budget draining a
  scheduled retry, so a busy app loses nothing at shutdown. `stats.queued` counts the queue, the
  retry backlog and the in-flight batch; `stats.batches_sent` counts only accepted batches.
- **Prefork runtimes**: the poll thread and the log sender are restarted in the child after
  `fork()`, and stale-while-revalidate takes over whenever no poll thread is alive - so a client
  built before gunicorn `--preload` or uWSGI forks keeps refreshing in every worker.
- **Payload policy** applied before anything leaves the process: deterministic sampling on the
  record id, `hash` and `none` modes, and UTF-8-safe truncation to the contract's caps.
- **Prompt endpoint client** as the simple path and the smoke test, with the same caching and
  fallback rules. It is a network call, so it makes none in test mode, in offline mode or without
  an API key: it serves a cached answer if it has one and otherwise says why it cannot.
- **Test mode** (no HTTP, no disk cache or bundle, logs captured for assertions) and **offline
  mode** (disk and bundle only), plus `prompton.testing.make_use_case_document` for building a use-case document in a
  test. A test-mode client starts empty, so it behaves the same on a developer's machine as in CI.
- **`python -m prompton export`** to fetch use cases in CI and commit it as a bundle.
- The cross-language conformance suite from the reference implementation runs in this SDK's tests:
  template rendering, use-case lookup, truncation, `stop_kind` and the golden monitoring-log records.
