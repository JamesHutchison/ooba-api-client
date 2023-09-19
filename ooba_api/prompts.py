import textwrap
from typing import TYPE_CHECKING

import pydantic

from ooba_api.pydantic_compat import USING_PYDANTIC_LEGACY


class Prompt(pydantic.BaseModel):
    negative_prompt: str | None = None
    prompt: str

    def full_prompt(self) -> str:
        return self.prompt


class InstructPrompt(Prompt):
    """
    Used for instructions
    """

    instruct_template: str = "{prompt}"

    def full_prompt(self) -> str:
        return self.instruct_template.format(prompt=self.prompt)


if not TYPE_CHECKING:
    if USING_PYDANTIC_LEGACY:
        Messages = pydantic.conlist(dict, min_items=1)
    else:
        Messages = pydantic.conlist(dict, min_length=1)
else:
    Messages = list


class ChatPrompt(Prompt):
    """
    Used for chat

    Not tested, not implemented, just placeholder
    """

    messages: Messages


class LlamaInstructPrompt(Prompt):
    """
    Used for llama, llama 2, code llama, etc
    """

    system_prompt: str = ""
    instruct_template: str = textwrap.dedent(
        """
        [INST] <<SYS>> {system_prompt} <</SYS>> {user_prompt} [/INST]
        """
    ).strip()

    def full_prompt(self) -> str:
        return self.instruct_template.format(
            system_prompt=self.system_prompt, user_prompt=self.prompt
        )
