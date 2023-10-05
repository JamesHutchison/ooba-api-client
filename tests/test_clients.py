import json
import logging

import pytest
import requests
from megamock import Mega, MegaMock

from ooba_api.clients import OobaApiClient
from ooba_api.model_info import OobaModelInfo, OobaModelNotLoaded
from ooba_api.prompts import InstructPrompt


class TestOobaApiClient:
    def test_default_init(self) -> None:
        client = OobaApiClient()

        assert client.url == "http://localhost:5000"
        assert client._chat_url == "http://localhost:5000/api/v1/chat"
        assert client._generate_url == "http://localhost:5000/api/v1/generate"
        assert client.api_key is None

    class TestInstruct:
        @pytest.fixture(autouse=True)
        def setup(self) -> None:
            self.client = MegaMock.it(OobaApiClient)
            Mega(self.client.instruct).use_real_logic()
            self.client._generate_url = "http://host/api/v1/generate"

        def test_returns_text_body(self, generate_output: dict) -> None:
            prompt = InstructPrompt(prompt="a prompt")
            response = MegaMock.it(requests.Response)
            response.json.return_value = generate_output
            self.client._post.return_value = response

            value = self.client.instruct(prompt=prompt)

            assert value == "output text"

        def test_prints_nothing_by_default(
            self, generate_output: dict, capsys: pytest.CaptureFixture
        ) -> None:
            prompt = InstructPrompt(prompt="a prompt")
            response = MegaMock.it(requests.Response)
            response.json.return_value = generate_output
            self.client._post.return_value = response

            self.client.instruct(prompt=prompt)

            assert capsys.readouterr().out == ""

        def test_print_prompt(self, generate_output: dict, capsys: pytest.CaptureFixture) -> None:
            prompt = InstructPrompt(prompt="a prompt")
            response = MegaMock.it(requests.Response)
            response.json.return_value = generate_output
            self.client._post.return_value = response

            self.client.instruct(prompt=prompt, print_prompt=True)

            assert capsys.readouterr().out == "a prompt\n"

        def test_raises_for_bad_status(self) -> None:
            prompt = InstructPrompt(prompt="a prompt")
            response = MegaMock.it(requests.Response, spec_set=False)
            response.status_code = 400
            response.reason = MegaMock()
            response.url = MegaMock()
            Mega(response.raise_for_status).use_real_logic()
            self.client._post.return_value = response

            with pytest.raises(requests.HTTPError):
                self.client.instruct(prompt=prompt)

        def test_logs_prompt(
            self, generate_output: dict, caplog: pytest.LogCaptureFixture
        ) -> None:
            prompt = InstructPrompt(prompt="a prompt")
            response = MegaMock.it(requests.Response)
            response.json.return_value = generate_output
            self.client._post.return_value = response
            caplog.set_level(logging.DEBUG, "ooba_api")

            self.client.instruct(prompt=prompt)

            records = caplog.get_records("call")
            assert records[0].name == "ooba_api.prompt"
            assert records[0].msg == "a prompt"
            assert records[1].name == "ooba_api"
            assert records[1].msg == json.dumps(generate_output, indent=2)

    class TestModelInfo:
        @pytest.fixture(autouse=True)
        def setup(self) -> None:
            self.client = MegaMock.it(OobaApiClient)
            Mega(self.client.model_info).use_real_logic()
            Mega(self.client._model_api).use_real_logic()
            self.client._model_url = "http://host/api/v1/model"

        def test_when_not_loaded(self, model_not_loaded_output: dict) -> None:
            response = MegaMock.it(requests.Response)
            response.json.return_value = model_not_loaded_output
            self.client._post.return_value = response

            result: OobaModelNotLoaded = self.client.model_info()

            assert isinstance(result, OobaModelNotLoaded)
            assert result.shared_args
            assert result.shared_settings

        def test_when_loaded(self, model_loaded_output: dict) -> None:
            response = MegaMock.it(requests.Response)
            response.json.return_value = model_loaded_output
            self.client._post.return_value = response

            result: OobaModelInfo = self.client.model_info()

            assert result.model_name == "codellama-7b-instruct.Q4_K_M.gguf"
            # lora, at least creating one, is broken at the time of this writing
            assert result.lora_names == ["todo-actual-value"]
            assert result.shared_args
            assert result.shared_settings

    class TestLoadModel:
        @pytest.fixture(autouse=True)
        def setup(self) -> None:
            self.client = MegaMock.it(OobaApiClient)
            Mega(self.client.load_model).use_real_logic()
            Mega(self.client._model_api).use_real_logic()
            self.client._model_url = "http://host/api/v1/model"

        def test_load_model(self, load_model_output) -> None:
            response = MegaMock.it(requests.Response)
            response.json.return_value = load_model_output
            self.client._post.return_value = response

            result: OobaModelInfo = self.client.load_model(
                "codellama-7b-instruct.Q4_K_M.gguf",
                args_dict={
                    "loader": "ctransformers",
                    "n-gpu-layers": 100,
                    "n_ctx": 2500,
                    "threads": 0,
                    "n_batch": 512,
                    "model_type": "llama",
                },
            )

            assert result.model_name == "codellama-7b-instruct.Q4_K_M.gguf"
            assert result.lora_names == []
            assert result.shared_args
            assert result.shared_settings
