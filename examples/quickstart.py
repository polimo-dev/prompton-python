"""A runnable end-to-end example: load a use case, call a provider, log the result.

The "provider" here is a fake function, so this runs with no API keys and no network::

    python examples/quickstart.py

Point it at a real PromptOn instead by setting ``PTN_API_KEY`` (and ``PTN_HOST``); the code below
does not change - only where the use-case document comes from does::

    PTN_HOST=http://localhost:4000 PTN_API_KEY=ptn_yourproject_... python examples/quickstart.py
"""

from __future__ import annotations

import json
import os
import random
import time

from prompton import PromptOn, ProviderError, Result
from prompton.testing import make_use_case_document

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

    print("no PTN_API_KEY set - running against an in-memory use-case document")
    client = PromptOn(mode="offline", api_key=None, disk_cache=False)
    client.load_use_cases(
        make_use_case_document(
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
        use_case = client.use_case(USE_CASE)
        print(
            f"loaded {USE_CASE}: model={use_case.model} "
            f"revision={use_case.deployment['revision']} prompt={use_case.prompt} "
            f"source={use_case.source}"
        )
        print("prompts pinned by the live revision:", client.prompt_names(USE_CASE))

        variables = {"name": "Ada"}
        messages = use_case.messages(variables)
        print("rendered prompt:", json.dumps(messages, ensure_ascii=False, indent=2))

        def call() -> Result:
            try:
                answer = fake_provider(use_case.model, messages, **use_case.params)
            except TimeoutError as error:
                raise ProviderError(str(error), kind="timeout") from error
            result = Result.from_openai(answer)
            result.cost_source = "unknown"
            return result

        try:
            result = use_case.track(
                call,
                variables=variables,
                input_messages=messages,
                trace_id="example:1",
                end_user_ref="user-42",
                context={"language": "en"},
                metadata={"example": True},
            )
            print("the model said:", result.content)
        except ProviderError as error:
            # the failure was logged before it reached you, with its kind
            print("the provider failed:", error, f"(kind={error.kind})")

        stats = client.flush(timeout=5).as_dict()
        print("monitoring logs:", {k: v for k, v in stats.items() if v})
        print("use cases:", client.use_cases_info())


if __name__ == "__main__":
    main()
