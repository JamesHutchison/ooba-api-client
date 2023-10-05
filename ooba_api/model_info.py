from dataclasses import dataclass


@dataclass
class OobaModelInfo:
    model_name: str | None
    lora_names: list[str]
    shared_settings: dict
    shared_args: dict


class OobaModelNotLoaded(OobaModelInfo):
    def __init__(self, **kwargs) -> None:
        super().__init__(model_name=None, lora_names=[], **kwargs)

    def __repr__(self) -> str:
        return "OobaModelNotLoaded()"
