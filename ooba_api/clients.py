import json
import logging

import requests

from ooba_api.parameters import DEFAULT_PARAMETERS, Parameters
from ooba_api.prompts import Prompt

logger = logging.getLogger("ooba_api")
prompt_logger = logging.getLogger("ooba_api.prompt")


class OobaApiClient:
    """
    Client for the Ooba Booga text generation web UI
    """

    # full URL to chat endpoint
    _chat_url: str

    # full URL to generate endpoint
    _generate_url: str

    # API Key, not yet used
    api_key: str | None

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

        if self.api_key:
            logger.warning("API keys are not yet supported")

    def _post(self, target_url: str, timeout: float, data: dict) -> requests.Response:
        return requests.post(target_url, timeout=timeout, json=data)

    def instruct(
        self,
        prompt: Prompt,
        parameters: Parameters = DEFAULT_PARAMETERS,
        timeout: int | float = 500,
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
        response = self._post(
            self._generate_url,
            timeout=timeout,
            data=(
                {"prompt": prompt_to_use, "negative_prompt": prompt.negative_prompt or ""}
                | parameters.model_dump()
            ),
        )
        response.raise_for_status()
        data = response.json()
        if __debug__:
            logger.debug(json.dumps(data, indent=2))

        return data["results"][0]["text"]
