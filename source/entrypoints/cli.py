"""
This file contains the functions for the command line interface. The makefile calls the commands
defined in this file.

For help in terminal, navigate to the project directory, run the docker container, and from within
the container run the following examples:
    - `python3.9 source/scripts/commands.py --help`
    - `python3.9 source/scripts/commands.py extract --help`
"""
import logging.config
import logging
import click

import source.service.etl as etl
from source.service.datasets import DATA


logging.config.fileConfig(
    "source/config/logging_to_file.conf",
    defaults={'logfilename': 'output/log.log'},
    disable_existing_loggers=False
)


@click.group()
def main():
    """
    Logic For Extracting and Transforming Datasets
    """
    pass


@main.command()
def extract():
    """This function downloads the credit data from openml.org."""
    credit_data = etl.extract()
    logging.info(
        f"Credit data downloaded with {credit_data.shape[0]} "
        f"rows and {credit_data.shape[1]} columns."
    )
    DATA.raw__credit.save(credit_data)


@main.command()
def transform():
    """This function transforms the credit data."""
    raw__credit = DATA.raw__credit.load()
    logging.info("Transforming credit data.")
    credit = etl.transform(raw__credit)
    DATA.credit.save(credit)


if __name__ == '__main__':
    main()
