# Python API Client for Ooba-Booga's Text Generation Web UI

An API client for the text generation UI, with sane defaults.

Motivation: documentation isn't great, examples are gnarly, not seeing an existing library.

Supported use cases:
- [x] generate / instruct
- [ ] chat
- [ ] streaming instruct
- [ ] streaming chat
- [ ] model info

Example:
```python
import logging

from ooba_api.clients import OobaApiClient
from ooba_api.parameters import Parameters
from ooba_api.prompts import LlamaInstructPrompt

logger = logging.getLogger("ooba_api")
logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG)

client = OobaApiClient()  # defaults to http://localhost:5000

response = client.instruct(
    LlamaInstructPrompt(
        system_prompt=(
            "Generate only the requested code from the user. Do not generate anything else. "
            "Be succint. Generate markdown of the code, and give the correct type. "
            "If the code is python use ```python for the markdown. Do not explain afterwards"
        ),
        prompt="Generate a Python function to reverse the contents of a file",
    ),
    parameters=Parameters(temperature=0.2, repetition_penalty=0.7),
)
print(response)
```

~~~
  ```python
def reverse_file(file_path):
    with open(file_path, "r") as f:
        content = f.read()
    return content[::-1]
```
~~~
