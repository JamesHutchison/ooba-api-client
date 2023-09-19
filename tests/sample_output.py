import pytest


@pytest.fixture()
def generate_output():
    return {"results": [{"text": "output text"}]}
