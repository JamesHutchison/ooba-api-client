# Python API Client for Ooba-Booga's Text Generation Web UI

An API client for the text generation UI, with sane defaults.

Motivation: documentation isn't great, examples are gnarly, not seeing an existing library.

Supported use cases:
- [x] generate / instruct
- [ ] chat
- [ ] streaming instruct
- [ ] streaming chat
- [x] model info
- [x] model loading

## Installation

`pip install ooba-api-client`

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

from ooba_api import OobaApiClient, Parameters, LlamaInstructPrompt

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

## Model Information and Loading
To get the currently loaded model:

```python
from ooba_api import OobaApiClient, OobaModelInfo, OobaModelNotLoaded

model_info: OobaModelInfo = client.model_info()

print(model_info)

# model_info will be OobaModelNotLoaded if no model is currently loaded
assert not isinstance(model_info, OobaModelNotLoaded)
```

To load a model:

```python
load_model_response: OobaModelInfo = client.load_model(
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
```

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

### Running Ooba-Booga
The Text Generation Web UI can be found here:

https://github.com/oobabooga/text-generation-webui/

This start-up config gives very good performance on a RTX 3060-TI with 8 GB of VRAM on an i5 system with 32 GB of DDR4. In many cases this produced over 20 tokens / sec. There is no delay before producing tokens.

`python server.py --model codellama-7b-instruct.Q4_K_M.gguf --api --listen --gpu-memory 8 --cpu-memory 1 --n-gpu-layers 1000000 --loader ctransformer --n_ctx 2500`

Increasing the context size (2500) will slow it down.
