"""Every exception the SDK raises.

The rule of thumb: configuration mistakes and prompt lookup mistakes raise, logging never does.
``PromptOn.log`` and ``PromptOn.flush`` swallow their own failures (they count them instead), so a
monitoring problem can never take down a request that already produced an answer.
"""

from __future__ import annotations

from typing import Any


class PromptOnError(Exception):
    """Base class for every error raised by this SDK."""


class ConfigurationError(PromptOnError):
    """An option or environment variable holds a value the SDK cannot use."""


class PromptDocumentUnavailableError(PromptOnError):
    """No prompt document in memory, on disk or in the bundle, and the server is unreachable.

    This is the only prompt lookup error that means "PromptOn is unreachable and nothing is
    cached". Every other lookup error is a bug in the app or in the deployment.
    """


class PromptLookupError(PromptOnError):
    """Base class for local prompt lookup failures described by the runtime contract."""

    code = "prompt_lookup_error"


class UnknownPromptError(PromptLookupError):
    """The prompt document holds no prompt with this key."""

    code = "unknown_prompt"

    def __init__(self, prompt_key: str) -> None:
        super().__init__(f"unknown prompt: {prompt_key}")
        self.prompt_key = prompt_key


class UnresolvedError(PromptLookupError):
    """The prompt exists but has no live deployment in this environment."""

    code = "unresolved"

    def __init__(self, prompt_key: str) -> None:
        super().__init__(f"no live deployment for prompt: {prompt_key}")
        self.prompt_key = prompt_key


class UnknownTemplateError(PromptLookupError):
    """The live deployment pins no prompt template version under the requested template name.

    There is deliberately no fallback to ``default``: shipping English to a request that asked for
    ``ko`` is worse than an error.
    """

    code = "unknown_template"

    def __init__(self, prompt_key: str, template: str, template_names: list[str]) -> None:
        super().__init__(
            f'the live deployment for {prompt_key} pins no template named "{template}" - '
            f"available template names: {', '.join(template_names) or '(none)'}"
        )
        self.prompt_key = prompt_key
        self.template = template
        self.template_names = template_names


class TemplateError(PromptOnError):
    """Base class for prompt template failures."""

    code = "template_error"


class TemplateSyntaxError(TemplateError):
    """The template uses a construct outside the allowed Liquid subset, or is malformed."""

    code = "parse_error"


class MissingVariableError(TemplateError):
    """A variable the template reads was not supplied.

    A key present with a ``None`` value is *not* missing: it renders as the empty string.
    """

    code = "missing_variable"

    def __init__(self, variable: str) -> None:
        super().__init__(f"missing variable: {variable}")
        self.variable = variable


class RenderError(TemplateError):
    """The template parsed but rendering failed for another reason."""

    code = "render_error"


class NoTemplateError(TemplateError):
    """The selected prompt carries no template (``kind: embedding``)."""

    code = "no_template"


class APIError(PromptOnError):
    """A PromptOn HTTP endpoint answered with an error status."""

    def __init__(
        self,
        status: int,
        code: str | None = None,
        message: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(f"PromptOn API error {status} {code or ''}: {message or ''}".strip())
        self.status = status
        self.code = code
        self.message = message
        self.details = details or {}


class TransportError(PromptOnError):
    """The PromptOn server could not be reached at all (DNS, connection, timeout)."""


class ProviderError(PromptOnError):
    """Raise this from inside ``track`` to record a typed provider failure.

    ``kind`` is one of ``http_4xx``, ``http_5xx``, ``rate_limited``, ``timeout``, ``transport``,
    ``parse`` and ``app``; anything else is recorded as ``app``. Pass ``result`` when the provider
    did answer and you want its usage and output kept as a quality signal (a parse failure, say).
    The exception propagates unchanged after the monitoring log has been built.
    """

    def __init__(
        self,
        message: str,
        *,
        kind: str = "app",
        status: int | None = None,
        result: Any = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.status = status
        self.result = result
