import copy
from threading import Lock
from typing import Any
from langchain.llms.base import LLM
from ooba_api import OobaApiClient, Parameters, Prompt

from langchain.callbacks.manager import (
    CallbackManagerForLLMRun,
)


class BlockingLangChainOobaLLM(LLM):
    """
    A LangChain LLM that ensures only one call to Ooba is going out at a time.
    This ensures the ooba server isn't overwhelmed from concurrent requests.

    This will block and wait for any other calls to finish.
    """

    base_prompt: Prompt

    parameters: Parameters = Parameters()

    api_client: OobaApiClient

    print_prompt: bool = False

    _lock = Lock()

    @property
    def _llm_type(self) -> str:
        return "ooba"

    def _call(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        actual_prompt = copy.copy(self.base_prompt)
        actual_prompt.prompt = prompt

        with self._lock:
            return self.api_client.instruct(
                actual_prompt, self.parameters, print_prompt=self.print_prompt
            )
