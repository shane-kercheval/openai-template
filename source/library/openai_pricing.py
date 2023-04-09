from functools import singledispatch
from pydantic import BaseModel
import tiktoken


class InstructPricing(BaseModel):
    price_per_tokens: float
    per_x_tokens: int

    def cost(self, n_tokens: int) -> float:
        return self.price_per_tokens * (n_tokens / self.per_x_tokens)


PRICING_LOOKUP = {
    'text-babbage-001': InstructPricing(price_per_tokens=0.0005, per_x_tokens=1_000),
    'text-curie-001': InstructPricing(price_per_tokens=0.002, per_x_tokens=1_000),
    'text-davinci-003': InstructPricing(price_per_tokens=0.02, per_x_tokens=1_000),
}


def _encode(value: str, model: str) -> list[int]:
    encoding = tiktoken.encoding_for_model(model)
    return encoding.encode(value)


@singledispatch
def cost(value, model: str):
    raise NotImplementedError("Unsupported data type")


@cost.register
def _(n_tokens: int, model: str):
    return PRICING_LOOKUP[model].cost(n_tokens=n_tokens)


@cost.register
def _(value: str, model: str):
    return cost(len(_encode(value=value, model=model)), model=model)
