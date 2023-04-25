"""Defines the OpenAI models that are supported."""
from enum import Enum


class OpenAIModels(Enum):
    """
    Enum used to define the types of OpenAI Models available. This Base enum is used for type
    hinting only.
    """


class InstructModels(OpenAIModels):
    """Defines the models that can be used with InstructGPT."""

    BABBAGE = 'text-babbage-001'
    CURIE = 'text-curie-001'
    DAVINCI = 'text-davinci-003'


class EmbeddingModels(OpenAIModels):
    """Defines the embedding models available."""

    ADA = 'text-embedding-ada-002'
