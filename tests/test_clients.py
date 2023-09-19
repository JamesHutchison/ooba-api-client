import json
import logging

import pytest
import requests
from megamock import Mega, MegaMock

from ooba_api.clients import OobaApiClient
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
