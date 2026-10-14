"""A runnable end-to-end example: load a prompt, call a provider, log the result.

The "provider" here is a fake function, so this runs with no API keys and no network::

    python examples/quickstart.py

Point it at a real PromptOn instead by setting ``PTN_API_KEY`` (and ``PTN_HOST``); the code below
does not change - only where the prompt document comes from does::

    PTN_HOST=http://localhost:4000 PTN_API_KEY=ptn_yourproject_... python examples/quickstart.py
"""

from __future__ import annotations

import json
import os
import random
import time

from prompton import PromptOn, ProviderError, Result
from prompton.testing import make_prompt_document

PROMPT_KEY = "greeting"


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

    print("no PTN_API_KEY set - running against an in-memory prompt document")
    client = PromptOn(mode="offline", api_key=None, disk_cache=False)
    client.load_prompts(
        make_prompt_document(
            project="example",
            environment="production",
            greeting={
                "model": "openai/gpt-4o-mini",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a friendly greeter. Answer in one line.",
                    }
                ],
                "prompts": {
                    "ko": [
                        {
                            "role": "system",
                            "content": "너는 친절한 인사 도우미다. 한 줄로 답한다.",
                        }
                    ]
                },
                "params": {"temperature": 0.2, "max_tokens": 256},
            },
        )
    )
    return client


def main() -> None:
    client = build_client()
    with client:
        prompt_key = client.prompt(PROMPT_KEY)
        print(
            f"loaded {PROMPT_KEY}: model={prompt_key.model} "
            f"revision={prompt_key.deployment['revision']} template={prompt_key.template} "
            f"source={prompt_key.source}"
        )
        print("prompts pinned by the live revision:", client.template_names(PROMPT_KEY))

        variables = {"name": "Ada"}
        managed_messages = prompt_key.messages(variables)
        app_history = [{"role": "assistant", "content": "Ready to greet."}]
        messages = [
            *managed_messages,
            *app_history,
            {"role": "user", "content": f"Say hello to {variables['name']}."},
        ]
        print("provider messages:", json.dumps(messages, ensure_ascii=False, indent=2))

        def call() -> Result:
            try:
                answer = fake_provider(prompt_key.model, messages, **prompt_key.params)
            except TimeoutError as error:
                raise ProviderError(str(error), kind="timeout") from error
            result = Result.from_openai(answer)
            result.cost_source = "unknown"
            return result

        try:
            result = prompt_key.track(
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
        print("prompts:", client.prompts_info())


if __name__ == "__main__":
    main()
