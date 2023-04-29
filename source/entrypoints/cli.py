"""
Function definitions for the command line interface. The makefile calls the commands defined in
this file.

For help in terminal, navigate to the project directory, run the docker container, and from within
the container run the following examples:
    - `source/scripts/commands.py --help`
    - `source/scripts/commands.py extract --help`
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
    disable_existing_loggers=False,
)


@click.group()
def main() -> None:
    """Logic For Extracting and Transforming Datasets."""
    pass


@main.command()
def extract() -> None:
    """Extracts the data."""
    logging.info("Extracting Data")
    reddit = etl.extract()
    DATA.raw__reddit.save(reddit)


@main.command()
def transform() -> None:
    """Transforms the reddit data."""
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


@main.command()
def get_embeddings() -> None:
    """Gets the imbeddings for the reddit 'text' column."""
    reddit = DATA.reddit.load()
    embeddings = etl.get_embeddings(inputs=reddit['text'].tolist())
    DATA.reddit_embeddings.save(embeddings)


@main.command()
def using_acreate() -> None:
    """
    Shows an example of using `acreate()`, which also does asynchronous calls but does not have
    the retry logic that I implemented in my custom classes.
    """
    import openai
    import asyncio
    import source.config.config as config
    openai.api_key = config.OPENAI_TOKEN
    prompts = [
        "What is the capital of France? ",
        "What is the capital of Italy? ",
    ]
    # prompts *= 100
    # NOTE, if I try more than 20 parallel requests (uncomment above), i get the following error:
    # Too many parallel completions requested. You submitted 400 prompts, but you can currently
    # request up to at most a total of 20). Please contact us through our help center at
    # help.openai.com for further questions. (HINT: if you want to just evaluate probabilities
    # without generating new text, you can submit more prompts if you set 'max_tokens' to 0.)
    results = asyncio.run(openai.Completion.acreate(prompt=prompts, engine="text-ada-001"))
    print(len(results))
    print(results)


if __name__ == '__main__':
    main()
