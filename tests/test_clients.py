from ooba_api.clients import OobaApiClient


class TestOobaApiClient:
    def test_default_init(self) -> None:
        client = OobaApiClient()

        assert client.url == "http://localhost:5000"
        assert client._chat_url == "http://localhost:5000/api/v1/chat"
        assert client._generate_url == "http://localhost:5000/api/v1/generate"
        assert client.api_key is None

    class TestInstruct:
        def test_returns_text_body(self) -> None:
            pass

        def test_print_prompt(self) -> None:
            pass

        def test_raises_for_bad_status(self) -> None:
            pass
