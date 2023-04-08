"""This file defines test fixtures for pytest unit-tests."""
import pytest
from source.service.datasets import DatasetsBase, PickledDataLoader, CsvDataLoader


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


@pytest.fixture(scope='function')
def datasets_fake_cache():
    return TestDatasets(cache=True)


@pytest.fixture(scope='function')
def datasets_fake_no_cache():
    return TestDatasets(cache=False)
