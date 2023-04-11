"""Helper functions and classes used to call the OpenAPI asynchronously, and parse the results."""
from abc import abstractmethod
from datetime import datetime
import asyncio
import aiohttp
import json
from pydantic import BaseModel, validator
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, \
    RetryError
from source.library.openai_pricing import cost


RETRY_ATTEMPTS = 3
RETRY_MULTIPLIER = 1
RETRY_MAX = 10


class OpenAIResultBase(BaseModel):
    """Class that parses the dictionary result returned by OpenAI"""
    result: dict | None = {}

    @validator('result', pre=True, always=True)
    def set_x_to_empty_dict(cls, value):
        """Ensure that if the value pass to result is None then set it to an empty dict."""
        return value or {}

    @property
    @abstractmethod
    def has_data():
        """
        Indicates whether or not OpenAI provides a response with valid data (e.g. text, embeddings,
        etc.)
        """

    @property
    def timestamp(self) -> int:
        """Returns the timestamp of the response provided by OpenAI."""
        return self.result.get('created')

    @property
    def timestamp_utc(self) -> str:
        """Return the timestamp in the format of YYYY-MM-DD HH:MM:SS"""
        if self.timestamp:
            dt = datetime.utcfromtimestamp(self.timestamp)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        else:
            return None

    @property
    def model(self) -> str:
        """Returns the name of the model used (e.g. `text-davinci-003`)."""
        return self.result.get('model')

    @property
    def type(self) -> str:
        """Returns the type of model used (e.g. `text_completion`)."""
        return self.result.get('object')

    @property
    def usage_total_tokens(self) -> int:
        """Returns the total number of tokens used (and charged for)."""
        return self.result.get('usage', {}).get('total_tokens')

    @property
    def cost_total(self) -> float:
        """Returns the total cost of the API call."""
        if self.model and self.usage_total_tokens:
            return cost(self.usage_total_tokens, model=self.model)
        else:
            return None

    @property
    def error_code(self) -> str:
        """Returns the error code, if there was an error from OpenAI"""
        if not self.result:
            return 'Uknown Error (error probably occurred outside OpenAI API call)'
        return self.result.get('error', {}).get('code')

    @property
    def error_type(self) -> str:
        """Returns the error type, if there was an error from OpenAI"""
        if not self.result:
            return 'Uknown Error (error probably occurred outside OpenAI API call)'
        return self.result.get('error', {}).get('type')

    @property
    def error_message(self) -> str:
        """Returns the error message, if there was an error from OpenAI"""
        if not self.result:
            return 'Uknown Error (error probably occurred outside OpenAI API call)'
        return self.result.get('error', {}).get('message')


class OpenAIInstructResult(OpenAIResultBase):
    """
    Class that parses the dictionary result returned by OpenAI, specific to an Instruct Model.
    """
    @property
    def choices(self) -> list[dict]:
        """If multiple prompts were given, this property gives access to all of the responses."""
        return self.result.get('choices', [dict(text='')])

    @property
    def reply(self) -> str:
        """This property gives the text/reply of the first (or only) response to the prompt(s)."""
        text = self.choices[0].get('text', '')
        if text:
            return text.strip()
        return ''

    @property
    def usage_prompt_tokens(self) -> int:
        """Returns the number of tokens used in the prompt."""
        return self.result.get('usage', {}).get('prompt_tokens')

    @property
    def usage_completion_tokens(self) -> int:
        """Returns the number of tokens used in the completion of the prompt."""
        return self.result.get('usage', {}).get('completion_tokens')

    @property
    def has_data(self) -> bool:
        """
        Returns the whether or not the result from OpenAI has data (e.g. not an empty string).
        """
        return bool(self.reply)


class OpenAIEmbeddingResult(OpenAIResultBase):
    """
    Class that parses the dictionary result returned by OpenAI, specific to an Embeddings Model.
    """
    @property
    def data(self) -> list[dict]:
        """
        If multiple inputs were given, this property gives access to all of the embeddings for each
        input.
        """
        return self.result.get('data', [dict(embedding=[])])

    @property
    def embedding(self) -> list[dict]:
        """This property gives the embedding of the first (or only) input(s)."""
        return self.data[0].get('embedding', [])

    @property
    def has_data(self) -> bool:
        """Indicates whether or not a non-empty array was returned."""
        return bool(self.embedding)

    @property
    def model(self) -> str:
        """Returns the name of the model used (e.g. `text-embedding-ada-002`)."""
        return super().model.replace('-v2', '')


class OpenAIResponse(BaseModel):
    """Class that wraps the http status/response and the result given by OpenAI."""
    response_status: int
    response_reason: str
    result: OpenAIResultBase

    @validator('response_reason')
    def clean_reason(cls, value: str):
        return value.strip()

    @validator('response_status')
    def clean_status(cls, value: int):
        """The http status of the API call (e.g. 200 for success)"""
        if value < 0:
            raise ValueError(f"Invalid HTTP status code: {value}")
        return value

    @property
    def has_error(self) -> bool:
        """Indicates whether or not the api call or the OpenAI result has an error."""
        return self.response_status != 200 or self.result.error_code is not None

    @property
    def has_data(self) -> bool:
        """Indicates whether or not the OpenAI result has any data (e.g. not an empty string)."""
        return self.result.has_data


class OpenAIResponses(BaseModel):
    """Class that wraps multiple http responses and OpenAI results"""
    responses: list[OpenAIResponse]

    @property
    def any_errors(self) -> bool:
        """Indicates if any of the api calls or the OpenAI results have an error."""
        return any(r.has_error for r in self.responses)

    @property
    def any_missing_data(self) -> bool:
        """Indicates if any of the OpenAI results are missing data (e.g. an empty string)."""
        return any(not r.has_data for r in self.responses)

    @property
    def total_tokens(self) -> int:
        """Returns the total number of tokens used across all requests (and charged for)."""
        return sum(r.result.usage_total_tokens for r in self.responses)

    @property
    def total_cost(self) -> float:
        """Returns the total cost across all of the API calls."""
        return sum(r.result.cost_total for r in self.responses)

    def __len__(self):
        return len(self.responses)

    def __iter__(self):
        for response in self.responses:
            yield response

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.responses[index]
        elif isinstance(index, slice):
            return self.responses[index.start:index.stop:index.step]
        else:
            raise TypeError("Invalid index type")


class OpenAIWrapperException(Exception):
    """Used to catch/indicate that an error was caused from one of our custom exceptions."""


class InvalidModelTypeError(OpenAIWrapperException):
    """Exception to indicate that a specific type of OpenAIModels was expected."""


class ExceededMaxTokensError(OpenAIWrapperException):
    """Exception to indicate that a specific type of OpenAIModels was expected."""


class RateLimitError(OpenAIWrapperException):
    """Exception to indicate that there was a OpenAI gave an error related to its Rate-Limit."""


class MissingApiKeyError(OpenAIWrapperException):
    """Exception to indicate that the OpenAI api key was not set."""


class CustomResponse(BaseModel):
    """Custom Reponse object used when there is an error and we need to return status/reason."""
    status: int
    reason: str


@retry(
    stop=stop_after_attempt(RETRY_ATTEMPTS),
    wait=wait_exponential(multiplier=RETRY_MULTIPLIER, max=RETRY_MAX),
    retry=retry_if_exception_type(RateLimitError)
)
async def _post_async_with_retry(
        session: aiohttp.ClientSession,
        url: str,
        api_key: str,
        payload: dict) -> tuple[aiohttp.ClientResponse, dict]:
    """
    Helper function that allows us to retry in case of rate-limit errors, and recover
    gracefully if we exceed the maximum attempts.
    """
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    async with session.post(url, headers=headers, data=json.dumps(payload)) as response:
        result = await response.json()

    if response.status == 429:
        # 429 is `Too Many Requests`
        # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_handle_rate_limits.ipynb
        raise RateLimitError("Rate limit exceeded")

    return response, result


async def post_async(
        session: aiohttp.ClientSession,
        url: str,
        api_key: str,
        payload: dict) -> tuple[aiohttp.ClientResponse, dict]:
    """
    The post_async function is an asynchronous function that sends an HTTP POST request to a
    specified URL with a payload and returns a ClientResponse object along with the json/dict
    returned by the OpenAI response.

    If the response status is 429 (Too Many Requests), the function raises a RateLimitError,
    which enables automatic retrying of the function via tenacity. The retry configuration
    parameters are set using constants defined by RETRY_ATTEMPTS, RETRY_MULTIPLIER, and RETRY_MAX).

    Parameters:
        session: The HTTP client session to use for the request.
        url: The URL to which the request is to be sent.
        payload: The payload data to send with the request.
    """
    try:
        return await _post_async_with_retry(
            session=session,
            url=url,
            api_key=api_key,
            payload=payload
        )
    except RetryError:
        return CustomResponse(status=429, reason='Too Many Requests'), None


async def gather_payloads(
        url: str,
        api_key: str,
        payloads: list[dict]) -> list[tuple[aiohttp.ClientResponse, dict]]:
    """
    This function makes calls to the url asynchronously via post_async for all of the payloads
    provided. and returns an OpenAIResponse object.

    If the response status is 429 (Too Many Requests), the function raises a RateLimitError,
    which enables automatic retrying of the function via tenacity. The retry configuration
    parameters are set using constants defined by RETRY_ATTEMPTS, RETRY_MULTIPLIER, and RETRY_MAX).

    Parameters:
        urls: The URL to which the request is to be sent.
        payloads: A list of payload data to send with each request.
    """
    async with aiohttp.ClientSession() as session:
        tasks = [
            post_async(session=session, url=url, api_key=api_key, payload=payload)
            for payload in payloads
        ]
        results = await asyncio.gather(*tasks)
        return results
