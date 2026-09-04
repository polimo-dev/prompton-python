"""Local resolution: snapshot + use case key (+ prompt name) into a :class:`Resolution`.

A pure function over a decoded snapshot, and the same algorithm the server runs behind
``POST /resolve``::

    deployment       = snapshot.deployments[use_case]      # absent -> unresolved, not a fallback
    version          = snapshot.prompt_versions[deployment.prompt_pins[prompt or "default"]]
    model            = snapshot.models[deployment.model_id]
    params           = use_case.default_params      <- deployment.params
    provider_options = model.provider_options       <- deployment.provider_options

There is no fallback to ``default`` when the requested prompt name is not pinned: shipping English
to a request that asked for ``ko`` is worse than an error.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .errors import UnknownPromptError, UnknownUseCaseError, UnresolvedError
from .params import merge
from .snapshot_data import PayloadPolicy, SnapshotData

__all__ = ["DEFAULT_PROMPT", "Resolution", "prompt_names", "resolve"]

DEFAULT_PROMPT = "default"

ResolutionSource = Literal["remote", "disk", "bundle", "manual"]


@dataclass(frozen=True)
class Resolution:
    """What to use for this call: the model, the params and the prompt, plus the evidence.

    Hand ``model``, ``effective_params``, ``effective_provider_options`` and the rendered messages
    to your provider client; hand the whole object to :meth:`prompton.PromptOn.with_generation` so
    the monitoring log records which deployment revision and prompt version produced the call.
    """

    use_case: str
    kind: str
    prompt: str | None
    deployment_id: str | None
    deployment_revision: int | None
    prompt_version_id: str | None
    prompt_version_number: int | None
    engine: str
    model_id: str | None
    model: str | None
    provider: str | None
    effective_params: dict[str, Any] = field(default_factory=dict)
    effective_provider_options: dict[str, Any] = field(default_factory=dict)
    messages: tuple[dict[str, Any], ...] | None = None
    text_template: str | None = None
    available_prompts: tuple[str, ...] = ()
    input_schema: tuple[Any, ...] = ()
    payload_policy: PayloadPolicy | None = None
    resolution_source: ResolutionSource = "remote"
    etag: str | None = None
    warnings: tuple[str, ...] = ()

    @property
    def prompt_version(self) -> dict[str, Any] | None:
        """``{"id", "number"}`` of the pinned version, or ``None`` for an embedding use case."""
        if self.prompt_version_id is None and self.prompt_version_number is None:
            return None
        return {"id": self.prompt_version_id, "number": self.prompt_version_number}


def resolve(
    snapshot: SnapshotData,
    use_case: str,
    prompt: str | None = None,
    *,
    resolution_source: ResolutionSource = "remote",
    etag: str | None = None,
) -> Resolution:
    """Resolve one use case. Raises the contract's three resolution errors."""
    entry = snapshot.use_cases.get(use_case)
    if entry is None:
        raise UnknownUseCaseError(use_case)

    deployment = snapshot.deployments.get(use_case)
    if deployment is None:
        raise UnresolvedError(use_case)

    available = tuple(sorted(deployment.prompt_pins))

    if entry.kind == "embedding":
        # An embedding use case has no prompt at all; a prompt name is ignored, not rejected.
        prompt_name: str | None = None
        version_id: str | None = None
        available = ()
    else:
        prompt_name = prompt or DEFAULT_PROMPT
        if prompt_name not in deployment.prompt_pins:
            raise UnknownPromptError(use_case, prompt_name, list(available))
        version_id = deployment.prompt_pins[prompt_name]

    warnings: list[str] = []
    version = None
    if version_id is not None:
        version = snapshot.prompt_versions.get(version_id)
        if version is None:
            warnings.append(f"missing_prompt_version: {version_id}")

    model = None
    if deployment.model_id is not None:
        model = snapshot.models.get(deployment.model_id)
        if model is None:
            warnings.append(f"missing_model: {deployment.model_id}")

    messages = version.messages if (version and entry.kind == "chat") else None
    text_template = version.text_template if (version and entry.kind == "text") else None

    return Resolution(
        use_case=entry.key,
        kind=entry.kind,
        prompt=prompt_name,
        deployment_id=deployment.id,
        deployment_revision=deployment.revision,
        prompt_version_id=version.id if version else None,
        prompt_version_number=version.number if version else None,
        engine=version.engine if version else "liquid",
        model_id=model.id if model else None,
        model=model.model_id if model else None,
        provider=model.provider if model else None,
        effective_params=merge(entry.default_params, deployment.params),
        effective_provider_options=merge(
            model.provider_options if model else None, deployment.provider_options
        ),
        messages=messages,
        text_template=text_template,
        available_prompts=available,
        input_schema=entry.input_schema,
        payload_policy=entry.payload_policy,
        resolution_source=resolution_source,
        etag=etag,
        warnings=tuple(warnings),
    )


def prompt_names(snapshot: SnapshotData, use_case: str) -> list[str]:
    """The prompt names this use case's live deployment pins, sorted. ``[]`` when undeployed."""
    if use_case not in snapshot.use_cases:
        raise UnknownUseCaseError(use_case)
    deployment = snapshot.deployments.get(use_case)
    return sorted(deployment.prompt_pins) if deployment else []
