"""The ``POST /resolve`` client: the simple path, and the smoke test.

The server runs the same algorithm as :mod:`prompton.resolver`, so this is the quickest way to
prove a deployment is live and to see exactly how a prompt renders. It is *not* the hot path: it
costs a request per call. Ask for the raw template once (no ``variables``), cache it for the same
ten seconds the snapshot uses, and render locally - which is what :meth:`ResolveClient.resolve`
does for you.

When the server rate-limits, fails or cannot be reached, a cached answer is served instead.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .config import Config
from .errors import APIError, ConfigurationError, SnapshotUnavailableError, TransportError
from .http import Transport, build_headers, parse_api_error, retry_after_seconds
from .template import render as render_template
from .template import render_messages

__all__ = ["RemoteResolution", "ResolveClient"]

log = logging.getLogger("prompton")


@dataclass(frozen=True)
class RemoteResolution:
    """The body of a ``POST /resolve`` answer, with the messages already rendered."""

    use_case: str
    kind: str
    deployment: dict[str, Any]
    prompt: str | None
    prompts: list[str]
    model: str | None
    model_id: str | None
    provider: str | None
    effective_params: dict[str, Any]
    effective_provider_options: dict[str, Any]
    prompt_version: dict[str, Any] | None
    messages: list[dict[str, Any]] | None = None
    text: str | None = None
    warnings: list[str] = field(default_factory=list)
    etag: str | None = None

    @classmethod
    def from_body(cls, body: Mapping[str, Any]) -> RemoteResolution:
        return cls(
            use_case=body.get("use_case", ""),
            kind=body.get("kind", "chat"),
            deployment=body.get("deployment") or {},
            prompt=body.get("prompt"),
            prompts=list(body.get("prompts") or []),
            model=body.get("model"),
            model_id=body.get("model_id"),
            provider=body.get("provider"),
            effective_params=dict(body.get("effective_params") or {}),
            effective_provider_options=dict(body.get("effective_provider_options") or {}),
            prompt_version=body.get("prompt_version"),
            messages=body.get("messages"),
            text=body.get("text"),
            warnings=list(body.get("warnings") or []),
            etag=body.get("etag"),
        )

    def rendered(self, variables: Mapping[str, Any] | None) -> RemoteResolution:
        """This answer with its raw templates rendered against ``variables``."""
        if variables is None:
            return self
        messages = render_messages(self.messages, variables) if self.messages is not None else None
        text = render_template(self.text, variables) if self.text is not None else None
        return RemoteResolution(
            **{**self.__dict__, "messages": messages, "text": text},
        )


@dataclass
class _CacheEntry:
    value: RemoteResolution
    at: float


class ResolveClient:
    """Calls ``POST /resolve``, caching the raw answer for the configured TTL."""

    def __init__(self, config: Config, transport: Transport) -> None:
        self._config = config
        self._transport = transport
        self._lock = threading.Lock()
        self._cache: dict[tuple[str, str, str], _CacheEntry] = {}
        self._not_before = 0.0
        self._failures = 0

    def resolve(
        self,
        use_case: str,
        *,
        prompt: str | None = None,
        variables: Mapping[str, Any] | None = None,
        environment: str | None = None,
        render_locally: bool = True,
    ) -> RemoteResolution:
        """Resolve on the server and render locally.

        The raw answer (no ``variables`` sent) is cached per use case, prompt and environment for
        ``cache_ttl`` seconds, so repeated calls with different variables cost one request.

        Pass ``render_locally=False`` to send the variables and let the server render them. That
        costs a request every time and skips the cache, but it is the exact reference behaviour -
        useful when you want to see what the server itself produces, including its ``400`` for a
        missing variable.

        This is a network call, so it is refused in test mode, and in offline mode or without an
        API key it serves a cached answer if there is one and otherwise says why it cannot.
        """
        environment = environment or self._config.environment
        key = (use_case, prompt or "default", environment)
        if not self._config.remote_enabled:
            return self._without_remote(key, variables)
        if not render_locally:
            return self._request(use_case, prompt, environment, variables)

        with self._lock:
            entry = self._cache.get(key)
            fresh = entry is not None and (time.monotonic() - entry.at) < self._config.cache_ttl
            blocked = time.monotonic() < self._not_before

        if entry is not None and (fresh or blocked):
            return entry.value.rendered(variables)

        try:
            answer = self._request(use_case, prompt, environment)
        except TransportError as error:
            self._note_failure(error)
            if entry is not None:
                log.warning("prompton: /resolve failed, serving the cached answer: %s", error)
                return entry.value.rendered(variables)
            raise
        except APIError as error:
            # a retryable status already scheduled its own pause in _request
            if entry is not None:
                log.warning("prompton: /resolve failed, serving the cached answer: %s", error)
                return entry.value.rendered(variables)
            raise

        with self._lock:
            self._cache[key] = _CacheEntry(answer, time.monotonic())
            self._failures = 0
            self._not_before = 0.0
        return answer.rendered(variables)

    def _without_remote(
        self, key: tuple[str, str, str], variables: Mapping[str, Any] | None
    ) -> RemoteResolution:
        """No remote calls are allowed: serve the cached answer, or say why there is none."""
        if self._config.mode == "test":
            raise ConfigurationError(
                "resolve_remote() is a network call and test mode makes none; load a document "
                "with load_snapshot() and use resolve() instead"
            )
        with self._lock:
            entry = self._cache.get(key)
        if entry is not None:
            return entry.value.rendered(variables)
        if self._config.mode == "offline":
            raise SnapshotUnavailableError(
                "offline mode makes no remote calls, and no /resolve answer is cached for "
                f"{key[0]!r}; use resolve() against the disk cache or the bundle instead"
            )
        raise SnapshotUnavailableError(
            "no API key configured: set PTN_API_KEY or pass api_key= to use the network"
        )

    def _request(
        self,
        use_case: str,
        prompt: str | None,
        environment: str,
        variables: Mapping[str, Any] | None = None,
    ) -> RemoteResolution:
        payload: dict[str, Any] = {"use_case": use_case, "environment": environment}
        if prompt is not None:
            payload["prompt"] = prompt
        if variables is not None:
            payload["variables"] = dict(variables)
        headers = build_headers(self._config.api_key, self._config.user_agent)
        headers["content-type"] = "application/json"
        response = self._transport.request(
            "POST",
            f"{self._config.base_url}/resolve",
            headers=headers,
            body=json.dumps(payload).encode("utf-8"),
            timeout=self._config.timeout,
        )
        if response.status == 200:
            body = response.json()
            if not isinstance(body, dict):
                raise APIError(response.status, message="the /resolve answer was not an object")
            return RemoteResolution.from_body(body)
        error = parse_api_error(response)
        if response.status == 429 or response.status >= 500:
            self._note_failure(error, retry_after_seconds(response))
        raise error

    def _note_failure(self, error: BaseException, retry_after: float | None = None) -> None:
        with self._lock:
            self._failures += 1
            delay = retry_after
            if delay is None:
                base = max(self._config.cache_ttl, 1.0)
                delay = min(base * (2 ** (self._failures - 1)), self._config.max_backoff)
            self._not_before = time.monotonic() + delay

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._failures = 0
            self._not_before = 0.0
