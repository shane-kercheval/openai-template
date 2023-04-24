"""Tests that we are pulling data from config file correctly."""
from source.config import config


def assert_string_not_empty(value: str) -> None:
    """Helper function to verify str is not empty."""
    assert isinstance(value, str)
    assert value is not None
    assert value.strip() != ''


def test_config_values_are_not_empty() -> None:
    """Tests each function of the config file to ensure expected keys are in yaml."""
    assert_string_not_empty(config.dir_data_processed())
    assert_string_not_empty(config.dir_ouput())
    assert_string_not_empty(config.dir_data_raw())
    assert_string_not_empty(config.dir_data_interim())
    assert_string_not_empty(config.dir_data_external())
    assert_string_not_empty(config.dir_data_processed())
    assert_string_not_empty(config.dir_notebooks())
    assert_string_not_empty(config.openai_token())
    pricing = config.openai_pricing()
    assert isinstance(pricing, list)
    assert isinstance(pricing[0], dict)
    assert 'model' in pricing[0]
    assert 'price_per_tokens' in pricing[0]
    assert 'per_x_tokens' in pricing[0]
