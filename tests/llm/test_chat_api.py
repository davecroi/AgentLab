import os
from types import SimpleNamespace

import pytest

from agentlab.llm.chat_api import (
    AnthropicModelArgs,
    AzureModelArgs,
    ChatModel,
    OpenAIModelArgs,
    make_system_message,
    make_user_message,
)

# TODO(optimass): figure out a good model for all tests


def _client_returning(content):
    message = SimpleNamespace(content=content)
    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
    )

    class FakeClient:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kw: completion)
            )

    return FakeClient


def test_chat_model_empty_content_becomes_empty_string():
    # Reasoning models can end their turn after the reasoning, with content=None.
    # It must reach the parser as "" (a retryable ParseError), not None (a TypeError).
    model = ChatModel("some/model", api_key="x", client_class=_client_returning(None))
    assert model([make_user_message("hi")])["content"] == ""


if "AGENTLAB_LOCAL_TEST" in os.environ:
    skip_tests = os.environ["AGENTLAB_LOCAL_TEST"] != "1"
else:
    skip_tests = False


@pytest.mark.pricy
@pytest.mark.skipif(skip_tests, reason="Skipping on remote as Azure is pricy")
@pytest.mark.skipif(
    not os.getenv("AZURE_OPENAI_API_KEY"), reason="Skipping as Azure API key not set"
)
def test_api_model_args_azure():
    model_args = AzureModelArgs(
        model_name="gpt-4.1-nano",
        deployment_name="gpt-4.1-nano",
        max_total_tokens=8192,
        max_input_tokens=8192 - 512,
        max_new_tokens=512,
        temperature=1e-1,
    )
    model = model_args.make_model()

    messages = [
        make_system_message("You are an helpful virtual assistant"),
        make_user_message("Give the third prime number"),
    ]
    answer = model(messages)

    assert "5" in answer.get("content")


@pytest.mark.pricy
@pytest.mark.skipif(skip_tests, reason="Skipping on remote as Azure is pricy")
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Skipping as OpenAI API key not set")
def test_api_model_args_openai():
    model_args = OpenAIModelArgs(
        model_name="gpt-4o-mini",
        max_total_tokens=8192,
        max_input_tokens=8192 - 512,
        max_new_tokens=512,
        temperature=1e-1,
    )
    model = model_args.make_model()

    messages = [
        make_system_message("You are an helpful virtual assistant"),
        make_user_message("Give the third prime number"),
    ]
    answer = model(messages)

    assert "5" in answer.get("content")


@pytest.mark.pricy
@pytest.mark.skipif(skip_tests, reason="Skipping on remote as Anthropic is pricy")
@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"), reason="Skipping as Anthropic API key not set"
)
def test_api_model_args_anthropic():
    model_args = AnthropicModelArgs(
        model_name="claude-3-haiku-20240307",
        max_total_tokens=8192,
        max_input_tokens=8192 - 512,
        max_new_tokens=512,
        temperature=1e-1,
    )
    model = model_args.make_model()

    messages = [
        make_system_message("You are an helpful virtual assistant"),
        make_user_message("Give the third prime number. Just the number, no explanation."),
    ]
    answer = model(messages)
    assert "5" in answer.get("content")


if __name__ == "__main__":
    test_api_model_args_anthropic()
