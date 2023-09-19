from typing import TYPE_CHECKING

import pydantic

if not TYPE_CHECKING:
    GuidanceScale = pydantic.confloat(ge=0, le=2.5)
    NewMaxTokens = pydantic.conint(gt=1)
    MinLength = pydantic.conint(gt=0)
    RepetitionPenaltyRange = pydantic.conint(gt=0)
    RepetitionPenalty = pydantic.confloat(gt=0)
    StoppingStrings = pydantic.conlist(str, min_length=0)
    Temperature = pydantic.confloat(gt=0, lt=1.0)
    TopK = pydantic.conint(gt=0)
    TopP = pydantic.confloat(gt=0, lt=1)
    TypicalP = pydantic.conint(gt=0, lt=1)
    TruncationLength = pydantic.conint(gt=1)
else:
    GuidanceScale = float
    NewMaxTokens = int
    MinLength = int
    RepetitionPenaltyRange = int
    RepetitionPenalty = float
    StoppingStrings = list
    Temperature = float
    TopK = int
    TopP = float
    TypicalP = int
    TruncationLength = int


class Parameters(pydantic.BaseModel):
    # add beginning token, highly recommended but maybe some models don't want this
    add_bos_token: bool = True

    # ignore max_new_tokens and instead use math to set max tokens and max out the context length
    auto_max_new_tokens: bool = False

    # disables the "end of sentence" token. Probably would only enable this if combined with custom stopping strings.
    ban_eos_token: bool = False

    # knob, how much it will listen to you
    guidance_scale: GuidanceScale = 1.0

    # limit output tokens
    max_new_tokens: NewMaxTokens = 128

    # force generation of a minimum length
    min_length: MinLength = 0

    # knob, set the range, in tokens, to consider when applying the repetition penalty. 0 means disabled
    # a value like 4 helps prevent repeated junk (like newline comments)
    # a larger value or disabled helps prevent repeating itself, but may influence the output
    repetition_penalty_range: RepetitionPenaltyRange = 0

    # knob, penalize repeating tokens. Recommended
    # note that this may interfere with correct syntax in code generation.
    # Consider "foo(bar(1, 2))" becomes "foo(bar(1, 2)"
    repetition_penalty: RepetitionPenalty = 1.1

    # knob, random seed, use -1 for a random... random seed
    seed: int = -1

    # do not include special tokens. Some models have to disable this
    skip_special_tokens: bool = True

    # tell model to stop if it spits out one of these strings
    stopping_strings: StoppingStrings = []

    # knob, with sample, random selection. High is more random or creative
    temperature: Temperature = 0.6

    # knob, number of samples.
    top_k: TopK = 20

    # knob, probability threshold for considering a sample
    top_p: TopP = 0.5

    # knob, a value of 1 disables it. Tokens too far from this are removed.
    # Typically used if a high temperature starts producing gibberish.
    typical_p: TypicalP = 1

    # length to truncate the prompt. Should always be lower than the max context size
    truncation_length: TruncationLength = 4096


DEFAULT_PARAMETERS = Parameters()
