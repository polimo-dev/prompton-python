"""PromptOn SDK for Python.

PromptOn is the control plane for your app's LLM prompts. For each prompt and environment it
holds one **pin**: a prompt version, one model and its parameters. Your app fetches prompts for
those pins, renders the pinned prompt with this call's variables, calls the provider **itself with
its own key and HTTP client**, and sends monitoring logs back in batches. PromptOn is never in the
request path; if it is down your app keeps running on the last prompts it received.

    import prompton

    prompton.configure(api_key="ptn_myproject_...")

    prompt_key = prompton.prompt("support_reply")
    messages = prompt.messages({"question": question})

    def call():
        answer = openai_client.chat.completions.create(
            model=prompt_key.model, messages=messages, **prompt_key.params
        )
        return prompton.Result.from_openai(answer)

    result = prompt_key.track(call, variables={"question": question})

A module-level default client covers the common case of one client per process. Build a
:class:`PromptOn` yourself when you want several, or when you want to control its lifetime.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from typing import Any

from ._version import SDK_NAME, VERSION
from .buffer import BufferStats
from .client import Prompt, PromptOn
from .config import Config
from .errors import (
    APIError,
    ConfigurationError,
    MissingVariableError,
    NoTemplateError,
    PromptDocumentUnavailableError,
    PromptLookupError,
    PromptOnError,
    ProviderError,
    RenderError,
    TemplateError,
    TemplateSyntaxError,
    TransportError,
    UnknownPromptError,
    UnknownTemplateError,
    UnresolvedError,
)
from .generation import Result
from .http import HttpResponse, Transport, UrllibTransport
from .prompt_render_client import RenderedPrompt
from .snapshot_data import PromptDocument
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
    "Prompt",
    "PromptDocument",
    "PromptDocumentUnavailableError",
    "PromptLookupError",
    "PromptOn",
    "PromptOnError",
    "ProviderError",
    "RenderError",
    "RenderedPrompt",
    "Result",
    "TemplateError",
    "TemplateSyntaxError",
    "Transport",
    "TransportError",
    "UnknownPromptError",
    "UnknownTemplateError",
    "UnresolvedError",
    "UrllibTransport",
    "__version__",
    "close",
    "configure",
    "flush",
    "get_client",
    "lint_template",
    "log",
    "log_events",
    "log_id",
    "normalize_stop_kind",
    "output_truncated",
    "prompt",
    "prompts_info",
    "render_prompt",
    "set_client",
    "template_names",
    "uuid7",
    "variables_of",
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


def prompt(key: str, template: str | None = None) -> Prompt:
    """:meth:`PromptOn.prompt` on the default client."""
    return get_client().prompt(key, template)


def template_names(prompt_key: str) -> list[str]:
    """:meth:`PromptOn.template_names` on the default client."""
    return get_client().template_names(prompt_key)


def render_prompt(
    prompt_key: str,
    *,
    template: str | None = None,
    variables: Mapping[str, Any] | None = None,
    environment: str | None = None,
    render_locally: bool = True,
) -> RenderedPrompt:
    """:meth:`PromptOn.render_prompt` on the default client."""
    return get_client().render_prompt(
        prompt_key,
        template=template,
        variables=variables,
        environment=environment,
        render_locally=render_locally,
    )


def log(record: Mapping[str, Any], **options: Any) -> str:
    """:meth:`PromptOn.log` on the default client."""
    return get_client().log(record, **options)


def log_events(events: list[Mapping[str, Any]]) -> dict[str, Any]:
    """:meth:`PromptOn.log_events` on the default client."""
    return get_client().log_events(events)


def log_id() -> str:
    """A UUIDv7 record id, issued before the provider call."""
    return uuid7()


def flush(timeout: float = 5.0) -> BufferStats:
    """:meth:`PromptOn.flush` on the default client."""
    return get_client().flush(timeout=timeout)


def prompts_info() -> dict[str, Any]:
    """:meth:`PromptOn.prompts_info` on the default client."""
    return get_client().prompts_info()


def close(timeout: float = 5.0) -> None:
    """Flush and stop the default client, if one was built."""
    global _default
    with _default_lock:
        client, _default = _default, None
    if client is not None:
        client.close(timeout=timeout)
