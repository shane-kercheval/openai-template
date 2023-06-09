"""Logic for extracting and transforming the project data."""

import logging
import pandas as pd
from helpsk.logging import log_function_call, log_timer, Timer
from source.config.config import OPENAI_TOKEN
from source.service.openai import OpenAI, EmbeddingModels


@log_function_call
@log_timer
def extract() -> pd.DataFrame:
    """Loads the reddit data."""
    with Timer("Loading Reddit Dataset - Saving to /artifacts/data/raw/reddit.pkl"):
        logging.info("This dataset was copied from https://github.com/blueprints-for-text-analytics-python/blueprints-text/tree/master/data/reddit-selfposts")  # noqa
        reddit = pd.read_csv('data/external/reddit.tsv.zip', sep="\t")
        reddit.rename(columns={'selftext': 'post'}, inplace=True)
    logging.info("Filtering/returning posts from Volvo subreddit.")
    return reddit.query("subreddit == 'Volvo'").reset_index(drop=True)


@log_function_call
@log_timer
def transform(reddit__raw: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms the reddit data.

    Args:
        reddit__raw: the raw data to transform
    """
    reddit = reddit__raw.copy()
    assert not reddit['title'].isna().any()
    assert not (reddit['title'].str.strip() == '').any()
    assert not reddit['post'].isna().any()
    assert not (reddit['post'].str.strip() == '').any()
    reddit['text'] = (reddit['title'].str.strip() + ' - ' + reddit['post'].str.strip())
    assert not reddit['text'].isna().any()
    assert not (reddit['text'] == '').any()
    return reddit


@log_timer
def get_embeddings(inputs: list[str]) -> list[list]:
    """
    Gets the embeddings for the corresponding list of strings.

    Args:
        inputs: list of strings to get embeddings for
    """
    oai = OpenAI(api_key=OPENAI_TOKEN)
    logging.info(f"Getting embeddings for: {len(inputs)} inputs.")
    responses = oai.text_embeddings(model=EmbeddingModels.ADA, inputs=inputs)
    logging.info(f"Total Cost: ${responses.total_cost:.4f}")
    logging.info(f"Total Tokens: {responses.total_tokens:,}")
    for response in responses:
        if response.has_error:
            logging.info(f"Error: {response.response_status} - {response.response_reason}")
    # assert not responses.any_errors
    # assert not responses.any_missing_data
    return [x.result.embedding for x in responses]
