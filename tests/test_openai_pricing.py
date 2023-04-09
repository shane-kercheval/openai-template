from source.library.openai_pricing import PRICING_LOOKUP, _encode, cost, InstructModels


def test__costs():
    test_value = "This is a string"
    encoding = _encode(test_value, model=InstructModels.BABBAGE)
    assert len(encoding) > 0
    assert isinstance(encoding, list)
    assert all([isinstance(x, int) for x in encoding])

    assert cost(1_000_000, model=InstructModels.BABBAGE) == 0.5
    assert cost(1_000_000, model=InstructModels.CURIE) == 2.0
    assert cost(1_000_000, model=InstructModels.DAVINCI) == 20.0
    assert cost(1_000_000, model='text-babbage-001') == 0.5
    assert cost(1_000_000, model='text-curie-001') == 2.0
    assert cost(1_000_000, model='text-davinci-003') == 20.0

    def _verify_cost(model):
        pricing = PRICING_LOOKUP[model]
        assert cost(test_value, model=model) == len(encoding) * pricing.price_per_tokens / pricing.per_x_tokens  # noqa

    _verify_cost(model=InstructModels.BABBAGE)
    _verify_cost(model=InstructModels.CURIE)
    _verify_cost(model=InstructModels.DAVINCI)
