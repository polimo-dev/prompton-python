"""PromptOn SDK for Python.

PromptOn is the control plane for your app's LLM prompts. For each use case and environment it
holds one **pin**: a prompt version, one model and its parameters. Your app fetches a snapshot of
those pins, renders the pinned prompt with this call's variables, calls the provider **itself with
its own key and HTTP client**, and sends monitoring logs back in batches. PromptOn is never in the
request path; if it is down your app keeps running on the last snapshot it received.

    import prompton

    prompton.configure(api_key="ptn_myproject_...")

    resolution = prompton.resolve("support_reply")
    messages = prompton.render(resolution, {"question": question})

    def call():
        answer = openai_client.chat.completions.create(
            model=resolution.model, messages=messages, **resolution.effective_params
        )
        return prompton.Outcome(
            content=answer.choices[0].message.content,
            finish_reason=answer.choices[0].finish_reason,
            input_tokens=answer.usage.prompt_tokens,
            output_tokens=answer.usage.completion_tokens,
        )

    outcome = prompton.with_generation(resolution, call, variables={"question": question})

A module-level default client covers the common case of one client per process. Build a
:class:`PromptOn` yourself when you want several, or when you want to control its lifetime.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from typing import Any

from ._version import SDK_NAME, VERSION
from .buffer import BufferStats
from .client import PromptOn
from .config import Config
from .errors import (
    APIError,
    ConfigurationError,
    MissingVariableError,
    NoTemplateError,
    PromptOnError,
    ProviderError,
    RenderError,
    ResolutionError,
    SnapshotUnavailableError,
    TemplateError,
    TemplateSyntaxError,
    TransportError,
    UnknownPromptError,
    UnknownUseCaseError,
    UnresolvedError,
)
from .generation import Outcome
from .http import HttpResponse, Transport, UrllibTransport
from .resolve_client import RemoteResolution
from .resolver import Resolution
from .snapshot_data import SnapshotData
from .stop_kind import normalize as normalize_stop_kind
from .stop_kind import truncated as output_truncated
from .template import LintReason, variables_of
from .template import lint as lint_template
from .uuidv7 import uuid7

__all__ = [
    "SDK_NAME",
    "VERSION",
    "APIError",
    "BufferStats",
    "Config",
    "ConfigurationError",
    "HttpResponse",
    "LintReason",
    "MissingVariableError",
    "NoTemplateError",
    "Outcome",
    "PromptOn",
    "PromptOnError",
    "ProviderError",
    "RemoteResolution",
    "RenderError",
    "Resolution",
    "ResolutionError",
    "SnapshotData",
    "SnapshotUnavailableError",
    "TemplateError",
    "TemplateSyntaxError",
    "Transport",
    "TransportError",
    "UnknownPromptError",
    "UnknownUseCaseError",
    "UnresolvedError",
    "UrllibTransport",
    "__version__",
    "close",
    "configure",
    "flush",
    "generation_id",
    "get_client",
    "lint_template",
    "log",
    "normalize_stop_kind",
    "output_truncated",
    "prompt_names",
    "render",
    "resolve",
    "resolve_remote",
    "set_client",
    "snapshot_info",
    "uuid7",
    "variables_of",
    "with_generation",
]

__version__ = VERSION

_default: PromptOn | None = None
_default_lock = threading.Lock()


def configure(**options: Any) -> PromptOn:
    """Create (or replace) the module-level default client and return it.

    Calling it again closes the previous default, so a test suite can reconfigure freely.
    """
    global _default
    client = PromptOn(**options)
    with _default_lock:
        previous, _default = _default, client
    if previous is not None:
        previous.close(timeout=1.0)
    return client


def get_client() -> PromptOn:
    """The default client, built from the environment on first use."""
    global _default
    with _default_lock:
        if _default is None:
            _default = PromptOn()
        return _default


def set_client(client: PromptOn | None) -> None:
    """Replace the default client outright - handy in tests. ``None`` clears it."""
    global _default
    with _default_lock:
        _default = client


def resolve(use_case: str, prompt: str | None = None) -> Resolution:
    """:meth:`PromptOn.resolve` on the default client."""
    return get_client().resolve(use_case, prompt)


def render(
    resolution: Resolution, variables: Mapping[str, Any] | None = None
) -> list[dict[str, Any]] | str:
    """:meth:`PromptOn.render` on the default client."""
    return get_client().render(resolution, variables)


def prompt_names(use_case: str) -> list[str]:
    """:meth:`PromptOn.prompt_names` on the default client."""
    return get_client().prompt_names(use_case)


def resolve_remote(
    use_case: str,
    *,
    prompt: str | None = None,
    variables: Mapping[str, Any] | None = None,
    environment: str | None = None,
    render_locally: bool = True,
) -> RemoteResolution:
    """:meth:`PromptOn.resolve_remote` on the default client."""
    return get_client().resolve_remote(
        use_case,
        prompt=prompt,
        variables=variables,
        environment=environment,
        render_locally=render_locally,
    )


def log(record: Mapping[str, Any], **options: Any) -> str:
    """:meth:`PromptOn.log` on the default client."""
    return get_client().log(record, **options)


def flush(timeout: float = 5.0) -> BufferStats:
    """:meth:`PromptOn.flush` on the default client."""
    return get_client().flush(timeout=timeout)


def with_generation(resolution: Resolution, call: Callable[[], Any], **meta: Any) -> Any:
    """:meth:`PromptOn.with_generation` on the default client."""
    return get_client().with_generation(resolution, call, **meta)


def generation_id() -> str:
    """A UUIDv7 record id, issued before the provider call."""
    return uuid7()


def snapshot_info() -> dict[str, Any]:
    """:meth:`PromptOn.snapshot_info` on the default client."""
    return get_client().snapshot_info()


def close(timeout: float = 5.0) -> None:
    """Flush and stop the default client, if one was built."""
    global _default
    with _default_lock:
        client, _default = _default, None
    if client is not None:
        client.close(timeout=timeout)
