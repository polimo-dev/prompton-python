"""Local resolution: snapshot + prompt key (+ template name) into a :class:`Resolution`.

A pure function over a decoded snapshot, and the same algorithm the server runs behind
the prompt endpoint::

    deployment       = snapshot.deployments[prompt_key]      # absent -> unresolved, not a fallback
    version          = snapshot.prompt_versions[deployment.template_pins[template or "default"]]
    model            = snapshot.models[deployment.model_id]
    params           = prompt_key.default_params      <- deployment.params
    provider_options = model.provider_options       <- deployment.provider_options

There is no fallback to ``default`` when the requested template name is not pinned: shipping English
to a request that asked for ``ko`` is worse than an error.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .errors import UnknownPromptError, UnknownTemplateError, UnresolvedError
from .params import merge
from .snapshot_data import PayloadPolicy, PromptDocument

__all__ = ["DEFAULT_TEMPLATE", "template_names"]

DEFAULT_TEMPLATE = "default"

ResolutionSource = Literal["remote", "disk", "bundle", "manual"]


@dataclass(frozen=True)
class Resolution:
    """What to use for this call: the model, the params and the prompt, plus the evidence.

    Hand ``model``, ``params``, ``provider_options`` and the rendered messages to your provider
    client; hand the whole object to PromptOn's tracking helpers so the monitoring log records
    which deployment revision and prompt version produced the call.
    """

    prompt_key: str
    kind: str
    template: str | None
    deployment_id: str | None
    deployment_revision: str | None
    prompt_version_id: str | None
    prompt_version_number: int | None
    engine: str
    model_id: str | None
    model: str | None
    provider: str | None
    params: dict[str, Any] = field(default_factory=dict)
    provider_options: dict[str, Any] = field(default_factory=dict)
    messages: tuple[dict[str, Any], ...] | None = None
    text_template: str | None = None
    decision: dict[str, Any] | None = None
    available_templates: tuple[str, ...] = ()
    api: str | None = None
    request_path: str | None = None
    tools: dict[str, Any] | None = None
    variables: tuple[Any, ...] = ()
    payload_policy: PayloadPolicy | None = None
    source: ResolutionSource = "remote"
    etag: str | None = None
    warnings: tuple[str, ...] = ()

    @property
    def prompt_version(self) -> dict[str, Any] | None:
        """``{"id", "number"}`` of the pinned version, or ``None`` for an embedding prompt."""
        if self.prompt_version_id is None and self.prompt_version_number is None:
            return None
        return {"id": self.prompt_version_id, "number": self.prompt_version_number}

    @property
    def key(self) -> str:
        """The prompt key."""
        return self.prompt_key

    @property
    def deployment(self) -> dict[str, Any]:
        """``{"id", "revision"}`` of the deployment that produced this resolution."""
        return {"id": self.deployment_id, "revision": self.deployment_revision}

    @property
    def template_names(self) -> tuple[str, ...]:
        """Template names pinned by this deployment."""
        return self.available_templates


def resolve(
    snapshot: PromptDocument,
    prompt_key: str,
    template: str | None = None,
    *,
    source: ResolutionSource = "remote",
    etag: str | None = None,
) -> Resolution:
    """Resolve one prompt. Raises the contract's three resolution errors."""
    entry = snapshot.prompts.get(prompt_key)
    if entry is None:
        raise UnknownPromptError(prompt_key)

    deployment = snapshot.deployments.get(prompt_key)
    if deployment is None:
        raise UnresolvedError(prompt_key)

    available = tuple(sorted(deployment.template_pins))

    if entry.kind == "embedding":
        # An embedding prompt has no template at all; a template name is ignored, not rejected.
        prompt_name: str | None = None
        version_id: str | None = None
        available = ()
    else:
        prompt_name = template or DEFAULT_TEMPLATE
        if prompt_name not in deployment.template_pins:
            raise UnknownTemplateError(prompt_key, prompt_name, list(available))
        version_id = deployment.template_pins[prompt_name]

    warnings: list[str] = []
    version = None
    if version_id is not None:
        version = snapshot.prompt_versions.get(version_id)
        if version is None:
            raise UnresolvedError(prompt_key)

    model = None
    if deployment.model_id is not None:
        model = snapshot.models.get(deployment.model_id)
        if model is None:
            raise UnresolvedError(prompt_key)

    runtime_kind = version.kind or entry.kind if version else entry.kind
    messages = version.messages if (version and runtime_kind == "chat") else None
    text_template = version.text_template if (version and runtime_kind == "text") else None
    decision = version.decision if (version and runtime_kind == "decision") else None

    return Resolution(
        prompt_key=entry.key,
        kind=runtime_kind,
        template=prompt_name,
        deployment_id=deployment.id,
        deployment_revision=deployment.revision,
        prompt_version_id=version.id if version else None,
        prompt_version_number=version.number if version else None,
        engine=version.engine if version else "liquid",
        model_id=model.id if model else None,
        model=model.model_id if model else None,
        provider=model.provider if model else None,
        params=merge(entry.default_params, deployment.params),
        provider_options=merge(
            model.provider_options if model else None, deployment.provider_options
        ),
        messages=messages,
        text_template=text_template,
        decision=decision,
        available_templates=available,
        variables=entry.variables,
        payload_policy=entry.payload_policy,
        api=deployment.api,
        request_path=deployment.request_path,
        tools=version.tools if (version and runtime_kind == "chat") else None,
        source=source,
        etag=etag,
        warnings=tuple(warnings),
    )


def template_names(snapshot: PromptDocument, prompt_key: str) -> list[str]:
    """The prompt names this prompt's live deployment pins, sorted. ``[]`` when undeployed."""
    if prompt_key not in snapshot.prompts:
        raise UnknownPromptError(prompt_key)
    deployment = snapshot.deployments.get(prompt_key)
    return sorted(deployment.template_pins) if deployment else []
