"""This file defines test fixtures for pytest unit-tests."""
import pytest
import re
from unittest.mock import MagicMock
from source.service.datasets import DatasetsBase, PickledDataLoader, CsvDataLoader


def is_valid_datetime_format(datetime_str):
    pattern = r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$"
    match = re.match(pattern, datetime_str)
    return bool(match)


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
    async def __aenter__(self, *args, **kwargs):
        return self

    async def __aexit__(self, *args, **kwargs):
        pass


class MockResponse:
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
def OPENAI_MODEL() -> str:
    return 'text-babbage-001'


@pytest.fixture(scope='session')
def babbage_model_payload__italy_capital(OPENAI_MODEL) -> dict:
    return {
        'model': OPENAI_MODEL,
        'prompt': "What is the capital of Italy?",
        'max_tokens': 50
    }
