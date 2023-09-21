# Python API Client for Ooba-Booga's Text Generation Web UI

An API client for the text generation UI, with sane defaults.

Motivation: documentation isn't great, examples are gnarly, not seeing an existing library.

Supported use cases:
- [x] generate / instruct
- [ ] chat
- [ ] streaming instruct
- [ ] streaming chat
- [ ] model info

## What model should I use?
If you're new to LLMs, you may be unsure what model to use for your use case.

In general, models tend to come in three flavors:
- a foundational model
- a chat model
- an instruct model

The foundational model typically is used for text prediction (typically suggestions), if its even good for that. You probably don't want this. Foundamational models often need behavior training to be useful.

The chat model is used for conversation histories. This would be the preferred model if you're trying to create a chat bot who replies to the user.

The instruct models are tuned towards following instructions. If your interest is in creating autonomous agents, this is probably what you want. Note that you can always roll up chat histories into a single prompt.

## Example
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
    parameters=Parameters(temperature=0.2, repetition_penalty=1.05),
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

## Appendix

### Specific Model Help

```python
# Code Llama Instruct
from ooba_api.prompts import LlamaInstructPrompt

response = client.instruct(
    LlamaInstructPrompt(
        system_prompt=(
            "Generate only the requested code from the user. Do not generate anything else. "
            "Be succint. Generate markdown of the code, and give the correct type. "
            "If the code is python use ```python for the markdown. Do not explain afterwards"
        ),
        prompt="Generate a Python function to reverse the contents of a file",
    ),
    parameters=Parameters(temperature=0.2, repetition_penalty=1.05),
)
```

```python
# falcon instruct
from ooba_api.prompts import InstructPrompt

response = client.instruct(
    InstructPrompt(
        prompt=(
            "Generate only the requested code from the user. Do not generate anything else. "
            "Be succint. Generate markdown of the code, and give the correct type. "
            "If the code is python use ```python for the markdown. Do not explain afterwards.\n"
            "Generate a Python function to reverse the contents of a file"
        )
    ),
    parameters=Parameters(temperature=0.2, repetition_penalty=1.05),
)
```
