from dataclasses import dataclass


@dataclass
class ModelInfo:
    model_name: str
    lora_names: list[str]
