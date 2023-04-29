"""Defines test fixtures for pytest unit-tests."""

import pytest
import re
from unittest.mock import MagicMock
from source.config.config import OPENAI_TOKEN
from source.service.openai import OpenAIResponse, InstructModels
from source.service.dataset_types import DatasetsBase, PickledDataLoader, CsvDataLoader


def is_valid_datetime_format(datetime_str: str) -> bool:
    """Checks that the str is in the expected datetime format of YYYY-MM-DD HH:MM:SS."""
    pattern = r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$"
    match = re.match(pattern, datetime_str)
    return bool(match)


def verify_openai_instruct_response(response: OpenAIResponse, expected_model: str) -> None:
    """Verifies the values in OpenAI insturct response are getting extracted correctly."""
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
    assert response.result.model_type == 'text_completion'
    assert response.result.usage_total_tokens > 0
    assert 0 < response.result.usage_prompt_tokens < response.result.usage_total_tokens
    assert 0 < response.result.usage_completion_tokens < response.result.usage_total_tokens
    assert 0 < response.result.cost_total < 0.1
    assert response.result.error_code is None
    assert response.result.error_type is None
    assert response.result.error_message is None


def verify_openai_instruct_response_on_error(response: OpenAIResponse) -> None:
    """
    Verifies the values in OpenAI response are getting extracted via. OpenAIResponse when there is
    an error.
    """
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
    assert response.result.model_type is None
    assert response.result.timestamp is None
    assert response.result.timestamp_utc is None
    assert response.result.usage_total_tokens is None
    assert response.result.usage_prompt_tokens is None
    assert response.result.usage_completion_tokens is None
    assert response.result.cost_total is None


def verify_openai_embedding_response(response: OpenAIResponse, expected_model: str) -> None:
    """Verifies the values in OpenAI embedding response are getting extracted correctly."""
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
    """Mock datasets to test functionality of Datasets classes."""

    def __init__(self, cache) -> None:  # noqa: ANN001
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
    """Used to test Async Erorrs."""


class MockResponse:
    """Used to test Async Erorrs."""

    def __init__(self, status: int):
        self.status = status

    async def json(self) -> dict:
        """Mocked function."""
        return {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):  # noqa: ANN001
        pass


@pytest.fixture()
def datasets_fake_cache() -> TestDatasets:
    """Returns fake dataset with cache turned on."""
    return TestDatasets(cache=True)


@pytest.fixture()
def datasets_fake_no_cache() -> TestDatasets:
    """Returns fake dataset with cache turned off."""
    return TestDatasets(cache=False)


@pytest.fixture(scope='session')
def openai_api_token() -> str:
    """Returns the OpenAI API Token."""
    return OPENAI_TOKEN


@pytest.fixture(scope='session')
def openai_url_completion() -> str:
    """The URL for OpenAI's completion api."""
    return 'https://api.openai.com/v1/completions'


@pytest.fixture(scope='session')
def openai_instruct_model() -> InstructModels:
    """OpenAI Model to use for testing. Goal is to minimize costs for unit tests."""
    return InstructModels.BABBAGE


@pytest.fixture(scope='session')
def babbage_model_payload__italy_capital(openai_instruct_model: InstructModels) -> dict:
    """Payload for asking OpenAI the capital of Italy so that we can verify it returns `France`."""
    return {
        'model': openai_instruct_model.value,
        'prompt': "What is the capital of Italy?",
        'max_tokens': 50,
    }
