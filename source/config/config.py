"""Provides convenience functions for configuration settings."""

from helpsk.utility import open_yaml


CONFIG = open_yaml('source/config/config.yaml')


def dir_ouput():  # noqa: ANN201, D103
    return CONFIG['output']['directory']


def dir_data_raw():  # noqa: ANN201, D103
    return CONFIG['data']['raw_directory']


def dir_data_interim():  # noqa: ANN201, D103
    return CONFIG['data']['interim_directory']


def dir_data_external():  # noqa: ANN201, D103
    return CONFIG['data']['external_directory']


def dir_data_processed():  # noqa: ANN201, D103
    return CONFIG['data']['processed_directory']


def dir_notebooks():  # noqa: ANN201, D103
    return CONFIG['notebooks']['directory']


def openai_pricing():  # noqa: ANN201, D103
    return CONFIG['openai']['pricing']


def openai_token():  # noqa: ANN201, D103
    with open(CONFIG['openai']['token_path']) as handle:
        return handle.read().strip()
