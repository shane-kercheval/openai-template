"""
This file contains the logic for extracting and transforming the project data.
"""
import logging
import pandas as pd
from helpsk.logging import log_function_call, log_timer, Timer


@log_function_call
@log_timer
def extract() -> pd.DataFrame:
    """This function loads the reddit ddata."""
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
    This function transforms the reddit data.

    Args:
        reddit__raw: the raw data to transform
    """
    reddit = reddit__raw.copy()
    assert not reddit['title'].isna().any() and not (reddit['title'].str.strip() == '').any()
    assert not reddit['post'].isna().any() and not (reddit['post'].str.strip() == '').any()
    reddit['text'] = (reddit['title'] + ' - ' + reddit['post']).str.strip()
    assert not reddit['text'].isna().any() and not (reddit['text'] == '').any()
    return reddit
