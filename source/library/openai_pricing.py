from functools import singledispatch
from enum import Enum
from pydantic import BaseModel
import tiktoken


class InstructModels(Enum):
    BABBAGE = 'text-babbage-001'
    CURIE = 'text-curie-001'
    DAVINCI = 'text-davinci-003'


class InstructPricing(BaseModel):
    price_per_tokens: float
    per_x_tokens: int

    def cost(self, n_tokens: int) -> float:
        return self.price_per_tokens * (n_tokens / self.per_x_tokens)


PRICING_LOOKUP = {
    InstructModels.BABBAGE: InstructPricing(price_per_tokens=0.0005, per_x_tokens=1_000),
    InstructModels.CURIE: InstructPricing(price_per_tokens=0.002, per_x_tokens=1_000),
    InstructModels.DAVINCI: InstructPricing(price_per_tokens=0.02, per_x_tokens=1_000),
}

MODEL_NAME_TO_ENUM_LOOKUP = {
    'text-babbage-001': InstructModels.BABBAGE,
    'text-curie-001': InstructModels.CURIE,
    'text-davinci-003': InstructModels.DAVINCI,
}


def _encode(value: str, model: InstructModels) -> list[int]:
    encoding = tiktoken.encoding_for_model(model.value)
    return encoding.encode(value)


@singledispatch
def cost(value, model: InstructModels | str):
    raise NotImplementedError("Unsupported data type")


@cost.register
def _(n_tokens: int, model: InstructModels | str):
    if isinstance(model, str):
        model = MODEL_NAME_TO_ENUM_LOOKUP[model]
    return PRICING_LOOKUP[model].cost(n_tokens=n_tokens)


@cost.register
def _(value: str, model: InstructModels | str):
    return cost(len(_encode(value=value, model=model)), model=model)
