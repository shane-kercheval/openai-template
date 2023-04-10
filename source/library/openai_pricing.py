"""Helper functions used to calculate the cost of API calls based on https://openai.com/pricing"""
from abc import ABC, abstractmethod
from functools import cache, singledispatch
from enum import Enum
from pydantic import BaseModel
import tiktoken


class OpenAIModels(Enum):
    """
    Enum used to define the types of OpenAI Models available. This Base enum is used for type
    hinting only.
    """


class InstructModels(OpenAIModels):
    BABBAGE = 'text-babbage-001'
    CURIE = 'text-curie-001'
    DAVINCI = 'text-davinci-003'


class EmbeddingModels(OpenAIModels):
    ADA = 'text-embedding-ada-002'


class ModelPricing(BaseModel, ABC):
    @abstractmethod
    def cost(self, n_tokens: int) -> float:
        """Functiont that takes the number of tokens and returns the estimated cost."""


class PerXTokensPricing(ModelPricing):
    price_per_tokens: float
    per_x_tokens: int

    def cost(self, n_tokens: int) -> float:
        return self.price_per_tokens * (n_tokens / self.per_x_tokens)


PRICING_LOOKUP = {
    EmbeddingModels.ADA: PerXTokensPricing(price_per_tokens=0.0004, per_x_tokens=1_000),
    InstructModels.BABBAGE: PerXTokensPricing(price_per_tokens=0.0005, per_x_tokens=1_000),
    InstructModels.CURIE: PerXTokensPricing(price_per_tokens=0.002, per_x_tokens=1_000),
    InstructModels.DAVINCI: PerXTokensPricing(price_per_tokens=0.02, per_x_tokens=1_000),
}

MODEL_NAME_TO_ENUM_LOOKUP = {
    EmbeddingModels.ADA.value: EmbeddingModels.ADA,
    InstructModels.BABBAGE.value: InstructModels.BABBAGE,
    InstructModels.CURIE.value: InstructModels.CURIE,
    InstructModels.DAVINCI.value: InstructModels.DAVINCI,
}


@cache
def _get_encoding(model: OpenAIModels):
    """Helper function that returns an encoding method for a given model."""
    return tiktoken.encoding_for_model(model.value)


def _encode(value: str, model: OpenAIModels) -> list[int]:
    """Helper function that takes a string/model and returns an encoding list."""
    encoding = _get_encoding(model=model)
    return encoding.encode(value)


def num_tokens(value: str, model: OpenAIModels) -> int:
    """
    Determines the number of tokens that the str value will be converted into, based on a given
    model.

    Args:
        value: the string
        model: the OpenAI model used in the API call.
    """
    return len(_encode(value=value, model=model))


@singledispatch
def cost(value, model: OpenAIModels | str):
    raise NotImplementedError("Unsupported data type")


@cost.register
def _(n_tokens: int, model: OpenAIModels | str):
    """
    Calculates the cost of an OpenAI API call based on the number of tokens and the model.

    Args:
        n_tokens: the total number of tokens used in the API call and cost model.
        model: the OpenAI model used in the API call.
    """
    if isinstance(model, str):
        model = MODEL_NAME_TO_ENUM_LOOKUP[model]
    return PRICING_LOOKUP[model].cost(n_tokens=n_tokens)


@cost.register
def _(value: str, model: OpenAIModels | str):
    """
    Calculates the cost of an OpenAI API call based on a given string (the string is converted into
    the corresponding number of tokens) and the model.

    Args:
        value: the string, which is used to determine the number of tokens
        model: the OpenAI model used in the API call.
    """
    return cost(num_tokens(value=value, model=model), model=model)
