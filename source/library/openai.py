from dataclasses import dataclass
import asyncio  # for making API calls concurrently
import aiohttp  # for running API calls concurrently
import json
from pydantic import BaseModel, validator
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, \
    RetryError


API_KEY = None
RETRY_ATTEMPTS = 3
RETRY_MULTIPLIER = 1
RETRY_MAX = 10


class OpenAIResponse(BaseModel):
    response_status: int
    response_reason: str
    openai_result: dict

    @validator('response_reason')
    def clean_reason(cls, value: str):
        return value.strip()

    @validator('response_status')
    def clean_status(cls, value: int):
        if value < 0:
            raise ValueError(f"Invalid HTTP status code: {value}")
        return value


class RateLimitError(Exception):
    pass


@retry(
    stop=stop_after_attempt(RETRY_ATTEMPTS),
    wait=wait_exponential(multiplier=RETRY_MULTIPLIER, max=RETRY_MAX),
    retry=retry_if_exception_type(RateLimitError)
)
async def post_async(
        session: aiohttp.ClientSession,
        url: str,
        payload: dict) -> OpenAIResponse:
    """
    The post_async function is an asynchronous function that sends an HTTP POST request to a
    specified URL with a payload and returns an OpenAIResponse object.

    If the response status is 429 (Too Many Requests), the function raises a RateLimitError,
    which enables automatic retrying of the function via tenacity. The retry configuration
    parameters are set using constants defined by RETRY_ATTEMPTS, RETRY_MULTIPLIER, and RETRY_MAX).

    Parameters:
        session : The HTTP client session to use for the request.
        url : The URL to which the request is to be sent.
        payload : The payload data to send with the request.
    """
    HEADERS = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {API_KEY}'
    }
    async with session.post(url, headers=HEADERS, data=json.dumps(payload)) as response:
        result = await response.json()
        if response.status == 429:
            # 429 is `Too Many Requests`
            # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_handle_rate_limits.ipynb
            raise RateLimitError("Rate limit exceeded")

        return OpenAIResponse(
            response_status=response.status,
            response_reason=response.reason,
            openai_result=result
        )
