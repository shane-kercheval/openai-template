import pytest
from source.library.openai_pricing import PRICING_LOOKUP, EmbeddingModels, _encode, num_tokens, \
    cost, InstructModels


def test__cost__instruct_models():
    test_value = "This is a string"
    encoding = _encode(test_value, model=InstructModels.BABBAGE)
    assert len(encoding) > 0
    assert isinstance(encoding, list)
    assert all([isinstance(x, int) for x in encoding])
    assert num_tokens(test_value, model=InstructModels.BABBAGE) == len(encoding)

    assert cost(1_000_000, model=InstructModels.BABBAGE) == 0.5
    assert cost(1_000_000, model=InstructModels.CURIE) == 2.0
    assert cost(1_000_000, model=InstructModels.DAVINCI) == 20.0
    assert cost(1_000_000, model=InstructModels.BABBAGE.value) == 0.5
    assert cost(1_000_000, model=InstructModels.CURIE.value) == 2.0
    assert cost(1_000_000, model=InstructModels.DAVINCI.value) == 20.0

    def _verify_cost(model):
        pricing = PRICING_LOOKUP[model]
        assert cost(test_value, model=model) == len(encoding) * pricing.price_per_tokens / pricing.per_x_tokens  # noqa

    _verify_cost(model=InstructModels.BABBAGE)
    _verify_cost(model=InstructModels.CURIE)
    _verify_cost(model=InstructModels.DAVINCI)


def test__cost__embedding_models():
    test_value = "This is a string"
    encoding = _encode(test_value, model=EmbeddingModels.ADA)
    assert len(encoding) > 0
    assert isinstance(encoding, list)
    assert all([isinstance(x, int) for x in encoding])
    assert num_tokens(test_value, model=EmbeddingModels.ADA) == len(encoding)

    assert cost(1_000_000, model=EmbeddingModels.ADA) == 0.4
    assert cost(1_000_000, model=EmbeddingModels.ADA.value) == 0.4

    def _verify_cost(model):
        pricing = PRICING_LOOKUP[model]
        assert cost(test_value, model=model) == len(encoding) * pricing.price_per_tokens / pricing.per_x_tokens  # noqa

    _verify_cost(model=EmbeddingModels.ADA)


def test__cost__invalid_type():
    with pytest.raises(NotImplementedError):
        cost([], model=InstructModels.BABBAGE)
