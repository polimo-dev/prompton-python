"""A runnable end-to-end example: resolve a prompt, call a provider, log the generation.

The "provider" here is a fake function, so this runs with no API keys and no network::

    python examples/quickstart.py

Point it at a real PromptOn instead by setting ``PTN_API_KEY`` (and ``PTN_HOST``); the code below
does not change - only where the snapshot comes from does::

    PTN_HOST=http://localhost:4000 PTN_API_KEY=ptn_yourproject_... python examples/quickstart.py
"""

from __future__ import annotations

import json
import os
import random
import time

from prompton import Outcome, PromptOn, ProviderError
from prompton.testing import make_snapshot

USE_CASE = "greeting"


def fake_provider(model: str, messages: list[dict], **params) -> dict:
    """Stands in for openai / anthropic / openrouter. Returns their usual answer shape."""
    time.sleep(0.05)
    if random.random() < 0.15:  # a demo, not a security decision
        raise TimeoutError("the provider took too long")
    name = messages[-1]["content"].rsplit(" ", 1)[-1].rstrip(".")
    return {
        "choices": [
            {"message": {"content": f"Hello, {name}! Lovely to see you."}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 38, "completion_tokens": 9},
        "model": model,
    }


def build_client() -> PromptOn:
    """A live client when a key is configured, otherwise a self-contained offline one."""
    if os.environ.get("PTN_API_KEY"):
        print("using the PromptOn server at", os.environ.get("PTN_HOST", "https://app.prompton.ai"))
        return PromptOn()

    print("no PTN_API_KEY set - running against an in-memory snapshot")
    client = PromptOn(mode="offline", api_key=None, disk_cache=False)
    client.load_snapshot(
        make_snapshot(
            project="example",
            environment="production",
            greeting={
                "model": "openai/gpt-4o-mini",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a friendly greeter. Answer in one line.",
                    },
                    {"role": "user", "content": "Say hello to {{ name }}."},
                ],
                "prompts": {"ko": [{"role": "user", "content": "{{ name }}님에게 인사해줘."}]},
                "params": {"temperature": 0.2, "max_tokens": 256},
            },
        )
    )
    return client


def main() -> None:
    client = build_client()
    with client:
        resolution = client.resolve(USE_CASE)
        print(
            f"resolved {USE_CASE}: model={resolution.model} "
            f"revision={resolution.deployment_revision} prompt={resolution.prompt} "
            f"source={resolution.resolution_source}"
        )
        print("prompts pinned by the live revision:", client.prompt_names(USE_CASE))

        variables = {"name": "Ada"}
        messages = client.render(resolution, variables)
        print("rendered prompt:", json.dumps(messages, ensure_ascii=False, indent=2))

        def call() -> Outcome:
            try:
                answer = fake_provider(resolution.model, messages, **resolution.effective_params)
            except TimeoutError as error:
                raise ProviderError(str(error), kind="timeout") from error
            choice = answer["choices"][0]
            return Outcome(
                content=choice["message"]["content"],
                finish_reason=choice["finish_reason"],
                input_tokens=answer["usage"]["prompt_tokens"],
                output_tokens=answer["usage"]["completion_tokens"],
                cost_source="unknown",
                model_used=answer["model"],
            )

        try:
            outcome = client.with_generation(
                resolution,
                call,
                variables=variables,
                input_messages=messages,
                trace_id="example:1",
                end_user_ref="user-42",
                context={"language": "en"},
                metadata={"example": True},
            )
            print("the model said:", outcome.content)
        except ProviderError as error:
            # the failure was logged before it reached you, with its kind
            print("the provider failed:", error, f"(kind={error.kind})")

        stats = client.flush(timeout=5).as_dict()
        print("monitoring logs:", {k: v for k, v in stats.items() if v})
        print("snapshot:", client.snapshot_info())


if __name__ == "__main__":
    main()
