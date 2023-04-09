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

# import source.library.openai as openai
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
    logging.info("Extracting Data")
    reddit = etl.extract()
    DATA.raw__reddit.save(reddit)


@main.command()
def transform():
    """This function transforms the reddit data."""
    raw__reddit = DATA.raw__reddit.load()
    logging.info("Transforming reddit data.")
    reddit = etl.transform(raw__reddit)
    DATA.reddit.save(reddit)

    # with open('/.openai_api_key', 'r') as handle:
    #     openai.API_KEY = handle.read().strip()

    # def generate_prompt(package: str) -> str:
    #     template = f"""
    #     Create a high-level description for the {package} python package. Be as accurate as
    #     possible and don't create descriptions for packages that don't exist. 
    #     """
    #     return template

    # packages = ['holoviews', 'bokeh', 'leather', 'pymc3', 'easyfinance']
    # prompts = [generate_prompt(x) for x in packages]
    # responses = openai.text_completion(model='text-babbage-001', prompts=prompts, max_tokens=200)



if __name__ == '__main__':
    main()
