import pydantic
import pytest

from ooba_api.prompts import ChatPrompt, InstructPrompt, LlamaInstructPrompt, Prompt


class TestPrompt:
    def test_full_prompt_excludes_negative_prompt(self) -> None:
        prompt = Prompt(prompt="prompt", negative_prompt="negative_prompt")
        assert prompt.full_prompt() == "prompt"


class TestInstructPrompt:
    def test_full_prompt_uses_instruct_template(self) -> None:
        prompt = InstructPrompt(prompt="prompt", negative_prompt="negative_prompt")
        prompt.instruct_template = "<DO> {prompt} </DO>"

        assert prompt.full_prompt() == "<DO> prompt </DO>"


class TestChatPrompt:
    def test_requires_at_least_one_message(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            ChatPrompt(prompt="prompt", negative_prompt="negative_prompt", messages=[])


class TestLlamaInstructPrompt:
    def test_generates_expected_instruct_template(self) -> None:
        prompt = LlamaInstructPrompt(
            prompt="Generate a library for ooba booga. Don't laugh at the name."
        )
        prompt.system_prompt = "You are a talented, experienced software engineer."

        assert prompt.full_prompt() == (
            "[INST] <<SYS>> You are a talented, experienced software engineer. <</SYS>> "
            "Generate a library for ooba booga. Don't laugh at the name. [/INST]"
        )
