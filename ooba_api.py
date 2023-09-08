import json
import logging
import textwrap

import pydantic
import requests

logger = logging.getLogger("ooba_api")
prompt_logger = logging.getLogger("ooba_api.prompt")


class Parameters(pydantic.BaseModel):
    add_bos_token: bool = True  # add beginning token, highly recommended
    auto_max_new_tokens: bool = (  # ignore max_new_tokens and instead max out the context length
        False
    )
    ban_eos_token: bool = False  # do not allow the model to end
    guidance_scale: pydantic.conint(  # knob, how much it will listen to you
        ge=0, le=2.5
    ) = 1
    max_new_tokens: pydantic.conint(gt=1) = 128  # limit output tokens
    min_length: pydantic.conint(gt=0) = 0  # force generation of a minimum length
    repetition_penalty_range: pydantic.conint(gt=0) = 1  # TODO: what does this do?
    repetition_penalty: pydantic.confloat(  # knob, penalize repeating tokens. Recommended
        gt=0
    ) = 1.1
    seed: int = -1  # knob, random seed, use -1 for a random... random seed
    skip_special_tokens: bool = (  # do not include special tokens. Some models have to disable this
        True
    )
    stopping_strings: pydantic.conlist(  # tell model to stop if it spits out one of these strings
        str, min_items=0
    ) = []
    temperature: pydantic.confloat(  # knob, with sample, random selection. High is more random or creative
        gt=0, lt=1.0
    ) = 0.6
    top_k: pydantic.conint(gt=0) = 20  # knob, number of samples
    top_p: pydantic.confloat(  # knob, probability threshold for considering a sample
        gt=0, lt=1
    ) = 0.9
    typical_p: pydantic.conint(gt=0) = 1  # TODO: what does this do?
    truncation_length: pydantic.conint(gt=1) = 4096  # length to truncate the prompt


DEFAULT_PARAMETERS = Parameters()


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


class ChatPrompt(Prompt):
    """
    Used for chat

    Not tested, not implemented, just placeholder
    """

    messages = pydantic.conlist(dict, min_items=1)


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


class OobaApiClient:
    """
    Client for the Ooba Booga text generation web UI
    """

    def __init__(
        self,
        url: str | None = None,
        *,
        host: str = "http://localhost",
        port: int = 5000,
        api_key: str | None = None,
    ):
        if url:
            self.url = url
        else:
            self.url = f"{host}:{port}"
        self._chat_url = f"{self.url}/api/v1/chat"
        self._generate_url = f"{self.url}/api/v1/generate"
        self.api_key = api_key

        assert not self.api_key, "API keys are not yet supported"

    def instruct(
        self,
        prompt: Prompt,
        parameters: Parameters = DEFAULT_PARAMETERS,
        timeout: int = 500,
        print_prompt: bool = False,
    ) -> str:
        """
        Provide an instruction, get a response

        :param messages: Message to provide an instruction
        :param max_tokens: Maximum tokens to generate
        :param timeout: When to timeout
        :param print_prompt: Print the prompt being used. Use case is debugging
        """
        prompt_to_use = prompt.full_prompt()
        if print_prompt:
            print(prompt_to_use)
        prompt_logger.info(prompt_to_use)
        response = requests.post(
            self._generate_url,
            timeout=timeout,
            json=prompt.dict() | parameters.dict(),
        )
        response.raise_for_status()
        data = response.json()
        if __debug__:
            logger.debug(json.dumps(data, indent=2))

        return data["results"][0]["text"]
