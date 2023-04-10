"""This file defines test fixtures for pytest unit-tests."""
import pytest
import re
from unittest.mock import MagicMock
from source.library.openai import OpenAIResponse, InstructModels
from source.service.dataset_types import DatasetsBase, PickledDataLoader, CsvDataLoader


def is_valid_datetime_format(datetime_str):
    pattern = r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$"
    match = re.match(pattern, datetime_str)
    return bool(match)


def verify_openai_instruct_response(response: OpenAIResponse, expected_model):
    assert response.result.reply
    assert response.result.has_data
    assert response.has_data
    assert response.response_status == 200
    assert response.response_reason == 'OK'
    assert not response.has_error
    assert len(response.result.choices) == 1
    assert response.result.timestamp > 1680995788
    assert is_valid_datetime_format(response.result.timestamp_utc)
    assert response.result.model == expected_model.value
    assert response.result.type == 'text_completion'
    assert response.result.usage_total_tokens > 0
    assert 0 < response.result.usage_prompt_tokens < response.result.usage_total_tokens
    assert 0 < response.result.usage_completion_tokens < response.result.usage_total_tokens
    assert 0 < response.result.cost_total < 0.1
    assert response.result.error_code is None
    assert response.result.error_type is None
    assert response.result.error_message is None


def verify_openai_instruct_response_on_error(response: OpenAIResponse):
    assert response.result.error_code
    assert response.result.error_type
    assert response.result.error_message
    assert response.result.reply == ''
    assert not response.result.reply
    assert not response.has_data
    assert response.has_error
    assert response.response_status != 200
    assert response.response_reason != 'OK'
    assert len(response.result.choices) == 1
    assert response.result.model is None
    assert response.result.type is None
    assert response.result.timestamp is None
    assert response.result.timestamp_utc is None
    assert response.result.usage_total_tokens is None
    assert response.result.usage_prompt_tokens is None
    assert response.result.usage_completion_tokens is None
    assert response.result.cost_total is None


def verify_openai_embedding_response(response: OpenAIResponse, expected_model):
    assert len(response.result.embedding) > 0
    assert all(isinstance(x, float) for x in response.result.embedding)
    assert response.result.has_data
    assert response.has_data
    assert response.response_status == 200
    assert response.response_reason == 'OK'
    assert not response.has_error
    # assert response.result.timestamp > 1680995788
    # assert is_valid_datetime_format(response.result.timestamp_utc)
    assert response.result.model == expected_model.value
    # assert response.result.type == 'text_completion'
    assert response.result.usage_total_tokens > 0
    assert 0 < response.result.cost_total < 0.1
    assert response.result.error_code is None
    assert response.result.error_type is None
    assert response.result.error_message is None


class TestDatasets(DatasetsBase):
    def __init__(self, cache) -> None:
        # define the datasets before calling __init__()
        self.dataset_1 = PickledDataLoader(
            description="Dataset description",
            dependencies=['SNOWFLAKE.SCHEMA.TABLE'],
            directory='.',
            cache=cache,
        )
        self.other_dataset_2 = PickledDataLoader(
            description="Other dataset description",
            dependencies=['dataset_1'],
            directory='.',
            cache=cache,
        )
        self.dataset_3_csv = CsvDataLoader(
            description="Other dataset description",
            dependencies=['other_dataset_2'],
            directory='.',
            cache=cache,
        )
        super().__init__()


class CustomAsyncMock(MagicMock):
    """Used to test Async Erorrs"""


class MockResponse:
    """Used to test Async Erorrs"""
    def __init__(self, status):
        self.status = status

    async def json(self):
        return {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


@pytest.fixture(scope='function')
def datasets_fake_cache():
    return TestDatasets(cache=True)


@pytest.fixture(scope='function')
def datasets_fake_no_cache():
    return TestDatasets(cache=False)


@pytest.fixture(scope='session')
def OPENAI_API_KEY() -> str:
    with open('/.openai_api_key', 'r') as handle:
        openai_api_key = handle.read().strip()
    return openai_api_key


@pytest.fixture(scope='session')
def OPENAI_URL_COMPLETION() -> str:
    return 'https://api.openai.com/v1/completions'


@pytest.fixture(scope='session')
def OPENAI_MODEL() -> InstructModels:
    return InstructModels.BABBAGE


@pytest.fixture(scope='session')
def babbage_model_payload__italy_capital(OPENAI_MODEL) -> dict:
    return {
        'model': OPENAI_MODEL.value,
        'prompt': "What is the capital of Italy?",
        'max_tokens': 50
    }
