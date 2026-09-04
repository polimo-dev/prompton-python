# Changelog

All notable changes to `prompton-sdk` for Python. This project follows
[semantic versioning](https://semver.org/).

## 0.1.0

Initial release: the PromptOn runtime contract for Python, with no runtime dependencies.

- **Resolve** a use case to its pinned prompt version, model and parameters, entirely locally from
  a cached snapshot. Prompt name is the only selection axis, and an unpinned name is an error
  rather than a silent fall back to `default`.
- **Render** the pinned prompt with the Liquid subset PromptOn allows - `for`, `if`/`elsif`/`else`,
  `unless`, `assign`, and the `size`, `join` and `default` filters - including Liquid's blank-body
  rule and the missing-variable semantics the contract specifies. Plus `lint` and `variables_of`.
- **Snapshot store** with the mandated caching rules: a 10-second memory cache, `If-None-Match`
  revalidation in a poll thread or stale-while-revalidate, `Retry-After` on 429, ×2 backoff to five
  minutes on 5xx and transport failures, and three tiers - memory, an atomically written disk cache
  with a sidecar, and an optional committed bundle. No database, no Redis, no coordination between
  instances. A document for another environment or project is never used.
- **Monitoring logs**: `log()`, `flush()` and the `with_generation()` wrapper (plus a `track()`
  context manager), backed by a batching buffer with app-generated UUIDv7 ids, size/time/byte flush
  triggers, batches of at most 200 records and 5 MB, partial-acceptance handling, retries of the
  same batch with the same ids on 429 and 5xx, 413 splitting, a bounded drop-oldest queue, a
  redaction hook, `hash_end_user`, and flush on shutdown.
- **Payload policy** applied before anything leaves the process: deterministic sampling on the
  record id, `hash` and `none` modes, and UTF-8-safe truncation to the contract's caps.
- **`POST /resolve` client** as the simple path and the smoke test, with the same caching and
  fallback rules.
- **Test mode** (no HTTP, logs captured for assertions) and **offline mode** (disk and bundle only),
  plus `prompton.testing.make_snapshot` for building a snapshot in a test.
- **`python -m prompton export`** to fetch a snapshot in CI and commit it as a bundle.
- The cross-language conformance suite from the reference implementation runs in this SDK's tests:
  template rendering, resolution, truncation, `stop_kind` and the golden monitoring-log records.
